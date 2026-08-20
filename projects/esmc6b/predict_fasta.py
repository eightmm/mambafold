#!/usr/bin/env python3
"""Predict provisional MambaFold ESMC-6B structures from single-chain FASTA."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from mambafold.data.constants import (
    AA_3TO1,
    AA_TO_ID,
    ATOM_NAME_TO_ID,
    CA_ATOM_ID,
    MAX_ATOMS_PER_RES,
    PAIR_PAD_ID,
    PAIR_TO_ID,
    RESIDUE_ATOMS,
)
from mambafold.data.esm import ESMC_6B_DIM, ESMC_6B_REVISION, ESMEmbedder
from mambafold.data.types import ProteinBatch, ProteinExample
from mambafold.sampling import sample
from mambafold.structure_io import write_mmcif, write_pdb
from mambafold.train.distributed import enable_cuda_perf_flags
from mambafold.train.trainer import load_from_checkpoint
from projects.esmc6b.verify_artifact import verify_checkpoint

AA_1TO3 = {one: three for three, one in AA_3TO1.items()}
SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def validate_sequence(sequence: str) -> None:
    """Validate the published single-chain sequence contract."""
    invalid = sorted(set(sequence) - set(AA_1TO3))
    if invalid:
        raise ValueError(f"non-standard amino-acid codes: {''.join(invalid)}")
    if not 10 <= len(sequence) <= 1024:
        raise ValueError(f"sequence length must be in [10, 1024], got {len(sequence)}")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read and validate FASTA records, using the first header token as ID."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []

    def flush() -> None:
        nonlocal header, sequence
        if header is None:
            return
        seq = re.sub(r"\s+", "", "".join(sequence)).upper()
        if not seq:
            raise ValueError(f"empty FASTA record: {header}")
        validate_sequence(seq)
        name = SAFE_ID.sub("_", header.split()[0]).strip("._")
        if not name:
            raise ValueError(f"invalid FASTA header: {header}")
        records.append((name, seq))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header, sequence = line[1:].strip(), []
            if not header:
                raise ValueError("FASTA header cannot be empty")
        elif header is None:
            raise ValueError("FASTA sequence encountered before its header")
        else:
            sequence.append(line)
    flush()
    if not records:
        raise ValueError(f"no FASTA records found in {path}")
    names = [name for name, _ in records]
    if len(names) != len(set(names)):
        raise ValueError("FASTA IDs must be unique after filename sanitization")
    return records


def sequence_example(sequence: str, esm: torch.Tensor) -> ProteinExample:
    """Create inference-only atom-slot features from one canonical sequence."""
    validate_sequence(sequence)
    length = len(sequence)
    if tuple(esm.shape) != (length, ESMC_6B_DIM):
        raise ValueError(
            f"expected ESMC-6B embeddings [{length}, {ESMC_6B_DIM}], got {tuple(esm.shape)}"
        )

    atom_type = torch.full(
        (length, MAX_ATOMS_PER_RES),
        ATOM_NAME_TO_ID["PAD"],
        dtype=torch.long,
    )
    pair_type = torch.full(
        (length, MAX_ATOMS_PER_RES),
        PAIR_PAD_ID,
        dtype=torch.long,
    )
    atom_mask = torch.zeros((length, MAX_ATOMS_PER_RES), dtype=torch.bool)
    res_type = torch.empty(length, dtype=torch.long)
    for index, one_letter in enumerate(sequence):
        residue = AA_1TO3[one_letter]
        res_type[index] = AA_TO_ID[residue]
        for slot, atom_name in enumerate(RESIDUE_ATOMS[residue]):
            atom_type[index, slot] = ATOM_NAME_TO_ID[atom_name]
            pair_type[index, slot] = PAIR_TO_ID[(residue, atom_name)]
            atom_mask[index, slot] = True

    nterm = torch.zeros(length, dtype=torch.bool)
    cterm = torch.zeros(length, dtype=torch.bool)
    nterm[0], cterm[-1] = True, True
    return ProteinExample(
        res_type=res_type,
        atom_type=atom_type,
        pair_type=pair_type,
        coords=torch.zeros((length, MAX_ATOMS_PER_RES, 3), dtype=torch.float32),
        atom_mask=atom_mask,
        observed_mask=torch.zeros_like(atom_mask),
        res_seq_nums=torch.arange(1, length + 1, dtype=torch.long),
        seq_len=length,
        is_nterm=nterm,
        is_cterm=cterm,
        esm=esm.cpu(),
    )


def make_batch(x: torch.Tensor, example: ProteinExample, t_cur: float, device: str) -> ProteinBatch:
    """Build the model batch for one sampling step."""
    length = example.seq_len
    return ProteinBatch(
        res_type=example.res_type.unsqueeze(0).to(device),
        res_seq_nums=example.res_seq_nums.unsqueeze(0).to(device),
        atom_type=example.atom_type.unsqueeze(0).to(device),
        pair_type=example.pair_type.unsqueeze(0).to(device),
        res_mask=torch.ones(1, length, dtype=torch.bool, device=device),
        atom_mask=example.atom_mask.unsqueeze(0).to(device),
        valid_mask=torch.zeros_like(example.atom_mask).unsqueeze(0).to(device),
        ca_mask=example.atom_mask[:, CA_ATOM_ID].unsqueeze(0).to(device),
        chain_id=example.chain_id.unsqueeze(0).to(device),
        entity_id=example.entity_id.unsqueeze(0).to(device),
        sym_id=example.sym_id.unsqueeze(0).to(device),
        is_nterm=example.is_nterm.unsqueeze(0).to(device),
        is_cterm=example.is_cterm.unsqueeze(0).to(device),
        x_clean=example.coords.unsqueeze(0).to(device),
        x_t=x.unsqueeze(0),
        eps=torch.zeros_like(x).unsqueeze(0),
        t=torch.tensor([[[[t_cur]]]], dtype=torch.float32, device=device),
        esm=example.esm.unsqueeze(0).to(device),
    )


def write_prediction(
    coords: np.ndarray,
    example: ProteinExample,
    confidence: np.ndarray,
    output_dir: Path,
    name: str,
    output_format: str,
) -> dict[str, str]:
    """Write one prediction through the shared structure writers."""
    res_type = example.res_type.numpy()
    atom_mask = example.atom_mask.numpy().astype(bool)
    chain_id = example.chain_id.numpy()
    b_factors = (
        np.asarray(confidence, dtype=np.float32)[:, None] * 100.0 * atom_mask.astype(np.float32)
    )
    outputs: dict[str, str] = {}
    if output_format in ("pdb", "both"):
        output_path = output_dir / f"{name}.pdb"
        write_pdb(coords, res_type, atom_mask, b_factors, chain_id, output_path)
        outputs["pdb"] = output_path.name
    if output_format in ("cif", "both"):
        output_path = output_dir / f"{name}.cif"
        write_mmcif(
            coords,
            res_type,
            atom_mask,
            b_factors,
            chain_id,
            output_path,
            entry_id=name,
        )
        outputs["cif"] = output_path.name
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True, help="single-chain protein FASTA")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="verified mambafold-esmc6b-170k-ema.pt artifact",
    )
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--n_steps", type=int, default=50, help="SDE integration steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-format",
        choices=("pdb", "cif", "both"),
        default="both",
        help="structure file format (default: both)",
    )
    args = parser.parse_args()

    if args.n_steps < 1:
        raise SystemExit("--n_steps must be positive")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output path: {args.out}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this MambaFold prerelease")

    try:
        manifest = verify_checkpoint(args.checkpoint)
        records = read_fasta(args.fasta)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    enable_cuda_perf_flags()
    model = load_from_checkpoint(args.checkpoint, args.device, use_ema=True).eval()
    embedder = ESMEmbedder("esmc-6b", device=args.device)
    # Fail on missing/incompatible PLM weights before reserving the requested
    # output path, so a dependency error does not make a clean retry look like
    # an overwrite attempt.
    embedder._get_client()
    args.out.mkdir(parents=True)
    output_rows = []
    for index, (name, sequence) in enumerate(records, start=1):
        embeddings = embedder([sequence], max_length=len(sequence)).squeeze(0)
        example = sequence_example(sequence, embeddings)
        with torch.no_grad():
            _, coords, _, _, confidence = sample(
                model,
                example,
                lambda x, t: make_batch(x, example, t, args.device),
                n_steps=args.n_steps,
                seed=args.seed,
                device=args.device,
                sampler="sde",
                sde_tau=0.01,
                sde_eps=0.01,
                sde_w_cutoff=0.99,
                sde_log_timesteps=True,
                record_trajectory=False,
            )
        outputs = write_prediction(
            coords,
            example,
            confidence,
            args.out,
            name,
            args.output_format,
        )
        output_rows.append({"id": name, "length": len(sequence), **outputs})
        print(f"[{index}/{len(records)}] {name} L={len(sequence)} -> {', '.join(outputs.values())}")

    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "project_id": manifest["project_id"],
                "project_status": manifest["status"],
                "release_status": manifest["release_status"],
                "source_tag": manifest["source_tag"],
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "conditioning_model": "biohub/ESMC-6B",
                "conditioning_revision": ESMC_6B_REVISION,
                "fasta": str(args.fasta),
                "sampler": "sde",
                "n_steps": args.n_steps,
                "seed": args.seed,
                "output_format": args.output_format,
                "records": output_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
