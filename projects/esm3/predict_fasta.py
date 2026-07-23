#!/usr/bin/env python3
"""Predict frozen MambaFold ESM3 structures from standard single-chain FASTA."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from verify_artifact import verify_checkpoint  # noqa: E402

from mambafold.data.constants import (  # noqa: E402
    AA_3TO1,
    AA_TO_ID,
    ATOM_NAME_TO_ID,
    CA_ATOM_ID,
    MAX_ATOMS_PER_RES,
    PAIR_PAD_ID,
    PAIR_TO_ID,
    RESIDUE_ATOMS,
)
from mambafold.data.esm import ESMEmbedder  # noqa: E402
from mambafold.data.types import ProteinBatch, ProteinExample  # noqa: E402
from mambafold.sampling import sample  # noqa: E402
from mambafold.train.distributed import enable_cuda_perf_flags  # noqa: E402
from mambafold.train.trainer import load_from_checkpoint  # noqa: E402

AA_1TO3 = {one: three for three, one in AA_3TO1.items()}
SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read a non-empty FASTA file, preserving the first header token as ID."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []

    def flush() -> None:
        nonlocal header, sequence
        if header is None:
            return
        seq = "".join(sequence).replace(" ", "").upper()
        if not seq:
            raise ValueError(f"empty FASTA record: {header}")
        name = SAFE_ID.sub("_", header.split()[0]).strip("._")
        if not name:
            raise ValueError(f"invalid FASTA header: {header}")
        records.append((name, seq))

    for raw_line in path.read_text().splitlines():
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
    invalid = sorted(set(sequence) - set(AA_1TO3))
    if invalid:
        raise ValueError(f"non-standard amino-acid codes: {''.join(invalid)}")
    length = len(sequence)
    if not 10 <= length <= 1024:
        raise ValueError(f"sequence length must be in [10, 1024], got {length}")
    if tuple(esm.shape) != (length, 1536):
        raise ValueError(f"expected ESM3 embeddings [{length}, 1536], got {tuple(esm.shape)}")

    atom_type = torch.full((length, MAX_ATOMS_PER_RES), ATOM_NAME_TO_ID["PAD"], dtype=torch.long)
    pair_type = torch.full((length, MAX_ATOMS_PER_RES), PAIR_PAD_ID, dtype=torch.long)
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


def write_pdb(
    coords: np.ndarray,
    example: ProteinExample,
    confidence: np.ndarray,
    path: Path,
) -> None:
    """Write one all-atom single-chain PDB; B-factors are predicted pLDDT × 100."""
    lines: list[str] = []
    serial = 1
    for residue_index, res_id in enumerate(example.res_type.tolist(), start=1):
        residue = next(name for name, value in AA_TO_ID.items() if value == res_id)
        for slot, atom_name in enumerate(RESIDUE_ATOMS[residue]):
            if not example.atom_mask[residue_index - 1, slot]:
                continue
            x, y, z = (float(v) for v in coords[residue_index - 1, slot])
            b_factor = max(0.0, min(100.0, float(confidence[residue_index - 1]) * 100.0))
            atom_field = atom_name if len(atom_name) >= 4 else f" {atom_name:<3s}"
            lines.append(
                f"ATOM  {serial:>5d} {atom_field:<4s} {residue:>3s} A{residue_index:>4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}          {atom_name[0]:>2s}\n"
            )
            serial += 1
    lines.extend((f"TER   {serial:>5d}      {residue:>3s} A{example.seq_len:>4d}\n", "END\n"))
    path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True, help="single-chain protein FASTA")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="verified frozen ESM3 v1 checkpoint"
    )
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--n_steps", type=int, default=50, help="SDE integration steps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.n_steps < 1:
        raise SystemExit("--n_steps must be positive")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output path: {args.out}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this MambaFold release")

    try:
        manifest = verify_checkpoint(args.checkpoint)
        records = read_fasta(args.fasta)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    enable_cuda_perf_flags()
    args.out.mkdir(parents=True)
    model = load_from_checkpoint(args.checkpoint, args.device, use_ema=True).eval()
    embedder = ESMEmbedder("esm3-open", device=args.device)
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
            )
        output_path = args.out / f"{name}.pdb"
        write_pdb(coords, example, confidence, output_path)
        output_rows.append({"id": name, "length": len(sequence), "pdb": output_path.name})
        print(f"[{index}/{len(records)}] {name} L={len(sequence)} -> {output_path}")

    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "project_id": manifest["project_id"],
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "fasta": str(args.fasta),
                "sampler": "sde",
                "n_steps": args.n_steps,
                "seed": args.seed,
                "records": output_rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
