#!/usr/bin/env python
"""Generate structures for an external benchmark set from a trained checkpoint.

With the transformer control arm cancelled, this is the only path from a
checkpoint to a number. It does three things and deliberately not a fourth:

    sequence -> ESMC-6B lookup -> sample -> PDB/mmCIF -> reference-free geometry

Accuracy scoring against deposited structures is NOT here, because the
reference structures are not in this repository — `benchmarks/external_testsets`
holds FASTA only. Scoring is a separate step against a separate download, and
pretending otherwise would produce a script that silently reports nothing.

What is here instead is reference-free stereochemistry, which needs no ground
truth. Bond lengths, consecutive CA spacing and steric clashes say something
real about whether the all-atom output is physical, and can be read before
reference structures are staged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambafold.confidence import expected_plddt, load_plddt_checkpoint  # noqa: E402
from mambafold.data.constants import (  # noqa: E402
    AA_3TO1,
    AA_TO_ID,
    ATOM_NAME_TO_ID,
    CA_ATOM_ID,
    COORD_SCALE,
    MAX_ATOMS_PER_RES,
    PAIR_PAD_ID,
    PAIR_TO_ID,
    RESIDUE_ATOMS,
)
from mambafold.data.sequence_cache import sequence_embedding_path  # noqa: E402
from mambafold.data.types import ProteinExample  # noqa: E402
from mambafold.losses.geometry import stereochemical_losses  # noqa: E402
from mambafold.sampling import prepare_inference_batch, sample  # noqa: E402
from mambafold.structure_io import write_pdb  # noqa: E402
from mambafold.train.trainer import build_model  # noqa: E402

AA_1TO3 = {one: three for three, one in AA_3TO1.items()}

# Ideal backbone bond lengths in Angstrom (Engh & Huber). Used only as a
# reference-free sanity scale, never as a training constraint.
IDEAL_BOND_A = {("N", "CA"): 1.458, ("CA", "C"): 1.525, ("C", "O"): 1.231}
IDEAL_CA_CA_A = 3.80


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name = line[1:].split()[0]
            buf = []
        elif line.strip():
            buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_target_ids(path: Path) -> list[str]:
    target_ids = [
        line.split("#", 1)[0].strip()
        for line in path.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if not target_ids:
        raise ValueError(f"target id file is empty: {path}")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError(f"target id file contains duplicates: {path}")
    return target_ids


def benchmark_seeds(seed: int, target_names: list[str]) -> list[int]:
    """Derive stable, independent RNG streams from the benchmark seed and target ID."""
    seeds = []
    for name in target_names:
        digest = hashlib.sha256(f"{seed}\0{name}".encode()).digest()
        seeds.append(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
    return seeds


def example_from_sequence(sequence: str, esm: torch.Tensor | None) -> ProteinExample:
    """Build an inference example from sequence alone.

    Coordinates are zero and every canonical slot is marked present: the sampler
    replaces `x_t` with noise, and `atom_mask` here means "this residue type has
    this atom", which is what decides the shape of the answer.
    """
    length = len(sequence)
    res_type = torch.zeros(length, dtype=torch.long)
    atom_type = torch.full((length, MAX_ATOMS_PER_RES), ATOM_NAME_TO_ID["PAD"], dtype=torch.long)
    pair_type = torch.full((length, MAX_ATOMS_PER_RES), PAIR_PAD_ID, dtype=torch.long)
    atom_mask = torch.zeros(length, MAX_ATOMS_PER_RES, dtype=torch.bool)
    for index, letter in enumerate(sequence):
        name3 = AA_1TO3.get(letter, "UNK")
        res_type[index] = AA_TO_ID.get(name3, AA_TO_ID["UNK"])
        for slot, atom in enumerate(RESIDUE_ATOMS.get(name3, RESIDUE_ATOMS["UNK"])):
            if slot >= MAX_ATOMS_PER_RES:
                break
            atom_type[index, slot] = ATOM_NAME_TO_ID.get(atom, ATOM_NAME_TO_ID["PAD"])
            pair_type[index, slot] = PAIR_TO_ID.get((name3, atom), PAIR_PAD_ID)
            atom_mask[index, slot] = True
    zeros_l = torch.zeros(length, dtype=torch.long)
    is_nterm = torch.zeros(length, dtype=torch.bool)
    is_cterm = torch.zeros(length, dtype=torch.bool)
    if length:
        is_nterm[0] = True
        is_cterm[-1] = True
    return ProteinExample(
        res_type=res_type,
        atom_type=atom_type,
        pair_type=pair_type,
        coords=torch.zeros(length, MAX_ATOMS_PER_RES, 3),
        atom_mask=atom_mask,
        observed_mask=atom_mask.clone(),
        res_seq_nums=torch.arange(length),
        seq_len=length,
        chain_id=zeros_l,
        entity_id=zeros_l.clone(),
        sym_id=zeros_l.clone(),
        is_nterm=is_nterm,
        is_cterm=is_cterm,
        esm=esm,
    )


def geometry_report(coords: np.ndarray, sequence: str) -> dict:
    """Reference-free stereochemistry of one predicted all-atom structure.

    These quantities need no deposited structure to check. Clash diagnostics
    use the exact same topology, radii and thresholds as geometric fine-tuning.
    """
    length = len(sequence)
    out: dict[str, float] = {}

    for (a_name, b_name), ideal in IDEAL_BOND_A.items():
        slots = []
        for index, letter in enumerate(sequence):
            atoms = RESIDUE_ATOMS.get(AA_1TO3.get(letter, "UNK"), [])
            if a_name in atoms and b_name in atoms:
                slots.append((index, atoms.index(a_name), atoms.index(b_name)))
        if not slots:
            continue
        i = np.array([s[0] for s in slots])
        d = np.linalg.norm(
            coords[i, [s[1] for s in slots]] - coords[i, [s[2] for s in slots]], axis=-1
        )
        out[f"bond_{a_name}-{b_name}_mean_A"] = float(d.mean())
        out[f"bond_{a_name}-{b_name}_rmsd_from_ideal_A"] = float(np.sqrt(((d - ideal) ** 2).mean()))

    ca = coords[:, CA_ATOM_ID, :]
    if length > 1:
        step = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
        out["ca_ca_mean_A"] = float(step.mean())
        out["ca_ca_rmsd_from_ideal_A"] = float(np.sqrt(((step - IDEAL_CA_CA_A) ** 2).mean()))
        out["ca_ca_frac_within_0.5A"] = float((np.abs(step - IDEAL_CA_CA_A) < 0.5).mean())

    if length:
        example = example_from_sequence(sequence, esm=None)
        pred = torch.from_numpy(coords).float().unsqueeze(0) / COORD_SCALE
        atom_mask = example.atom_mask.unsqueeze(0)
        clash = stereochemical_losses(
            pred,
            pred,
            atom_mask,
            atom_mask,
            torch.ones(1, length, dtype=torch.bool),
            example.res_type.unsqueeze(0),
            example.atom_type.unsqueeze(0),
            example.res_seq_nums.unsqueeze(0),
            example.chain_id.unsqueeze(0),
        )
        for key in (
            "clashes_per_1000_atoms",
            "soft_clashes_per_1000_atoms",
            "mean_clash_overlap_A",
        ):
            out[key] = float(clash[key].item())
        out["clash_surrogate_A_per_atom"] = float(clash["clash"].item())
        out["radius_of_gyration_A"] = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run_a_mamba.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--plddt_checkpoint",
        help="optional standalone pLDDT head; writes residue scores to PDB B-factors",
    )
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--esm_dir", default="data/casp_esmc6b")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--method", default="ode", choices=("ode", "sde"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sde_tau", type=float, default=0.01)
    ap.add_argument("--sde_eps", type=float, default=0.01)
    ap.add_argument("--sde_w_cutoff", type=float, default=0.99)
    ap.add_argument(
        "--sde_log_timesteps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--max_batch_residues", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="0 = every target")
    ap.add_argument(
        "--target_ids",
        type=Path,
        help="optional frozen target-id subset; unknown IDs are rejected",
    )
    ap.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    fasta_path = Path(args.fasta)
    cfg = yaml.safe_load(config_path.read_text())
    out_dir = Path(args.out_dir)
    (out_dir / "structures").mkdir(parents=True, exist_ok=True)

    targets = read_fasta(fasta_path)
    selected_target_ids = None
    if args.target_ids is not None:
        selected_target_ids = read_target_ids(args.target_ids)
        unknown = sorted(set(selected_target_ids) - set(targets))
        if unknown:
            raise ValueError(f"target id file contains IDs absent from FASTA: {unknown}")
        targets = {name: targets[name] for name in selected_target_ids}
    if args.limit:
        targets = dict(sorted(targets.items())[: args.limit])

    esm_dir = Path(args.esm_dir)
    usable, missing = {}, []
    for name, seq in targets.items():
        path = sequence_embedding_path(esm_dir, seq)
        if path.exists():
            usable[name] = (seq, torch.from_numpy(np.load(path)).float())
        else:
            missing.append(name)
    print(
        f"[rollout] {len(targets)} targets, {len(usable)} with embeddings, {len(missing)} missing",
        flush=True,
    )
    if missing:
        print(
            f"[rollout] missing: {', '.join(sorted(missing)[:8])}"
            f"{' ...' if len(missing) > 8 else ''}",
            flush=True,
        )
    if not usable:
        print(
            "[rollout] nothing to sample; run pipeline/14_embed_external_testsets.py",
            file=sys.stderr,
        )
        return 2

    state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model_config = state.get("args", cfg) if isinstance(state, dict) else cfg
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint args must be a configuration mapping")
    model = build_model(model_config, args.device)
    key = "ema" if (args.use_ema and "ema" in state) else "model"
    weights = state.get(key, state)
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    weights = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.eval()
    print(f"[rollout] loaded '{key}' weights from {checkpoint_path}", flush=True)

    confidence_head = None
    confidence_temperature = 1.0
    confidence_metadata = None
    confidence_path = Path(args.plddt_checkpoint) if args.plddt_checkpoint else None
    if confidence_path is not None:
        confidence_head, confidence_temperature, confidence_metadata = load_plddt_checkpoint(
            confidence_path,
            map_location=args.device,
        )
        confidence_head.to(args.device).eval()
        print(
            f"[rollout] loaded pLDDT head step={confidence_metadata['step']} "
            f"from {confidence_path}",
            flush=True,
        )

    # Group by length so a short target is not padded up to the longest one.
    order = sorted(usable, key=lambda n: len(usable[n][0]))
    batches: list[list[str]] = []
    current: list[str] = []
    for name in order:
        longest = max(len(usable[n][0]) for n in current + [name])
        if current and longest * (len(current) + 1) > args.max_batch_residues:
            batches.append(current)
            current = []
        current.append(name)
    if current:
        batches.append(current)

    results, started = {}, time.time()
    for index, names in enumerate(batches, 1):
        examples = [example_from_sequence(*usable[n]) for n in names]
        batch = prepare_inference_batch(examples, args.device, length_bin=cfg.get("length_bin", 0))
        # Give each target a stable independent stream. This remains invariant
        # to batching and avoids Python's process-randomized hash().
        seeds = benchmark_seeds(args.seed, names)
        out = sample(
            model,
            batch,
            seeds,
            n_steps=args.n_steps,
            method=args.method,
            sde_tau=args.sde_tau,
            sde_eps=args.sde_eps,
            sde_w_cutoff=args.sde_w_cutoff,
            sde_log_timesteps=args.sde_log_timesteps,
            return_trunk_latent=confidence_head is not None,
        )
        batch_plddt = None
        if confidence_head is not None:
            if out.trunk_latent is None:
                raise RuntimeError("sampler did not return the latent required by pLDDT head")
            with torch.inference_mode():
                confidence_logits = confidence_head(
                    out.trunk_latent.to(args.device),
                    out.residue_mask.to(args.device),
                )
                batch_plddt = (
                    expected_plddt(
                        confidence_logits,
                        temperature=confidence_temperature,
                    )
                    .float()
                    .cpu()
                )
        for row, name in enumerate(names):
            seq = usable[name][0]
            coords = out.final_aa[row, : len(seq)].numpy()
            example = examples[row]
            path = out_dir / "structures" / f"{name}.pdb"
            atom_mask_np = example.atom_mask.numpy()
            residue_plddt = (
                np.zeros(len(seq), dtype=np.float32)
                if batch_plddt is None
                else batch_plddt[row, : len(seq)].numpy()
            )
            write_pdb(
                coords_A=coords,
                res_type_ids=example.res_type.numpy(),
                atom_mask=atom_mask_np,
                b_factors=np.broadcast_to(residue_plddt[:, None], atom_mask_np.shape).copy(),
                chain_id=example.chain_id.numpy(),
                path=path,
            )
            results[name] = {
                "length": len(seq),
                **(
                    {
                        "mean_plddt": float(residue_plddt.mean()),
                        "min_plddt": float(residue_plddt.min()),
                        "max_plddt": float(residue_plddt.max()),
                    }
                    if batch_plddt is not None
                    else {}
                ),
                **geometry_report(coords, seq),
            }
        print(
            f"  batch {index}/{len(batches)}  {len(names)} targets  {time.time() - started:.0f}s",
            flush=True,
        )

    def mean(key: str) -> float | None:
        vals = [r[key] for r in results.values() if key in r]
        return float(np.mean(vals)) if vals else None

    summary = {
        "schema_version": 3,
        "fasta": str(fasta_path),
        "fasta_sha256": file_sha256(fasta_path),
        "target_ids_file": str(args.target_ids) if args.target_ids else None,
        "target_ids_file_sha256": file_sha256(args.target_ids) if args.target_ids else None,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "plddt_checkpoint": str(confidence_path) if confidence_path else None,
        "plddt_checkpoint_sha256": (file_sha256(confidence_path) if confidence_path else None),
        "plddt_head_step": (
            confidence_metadata["step"] if confidence_metadata is not None else None
        ),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "weights": key,
        "n_steps": args.n_steps,
        "method": args.method,
        "seed": args.seed,
        "seed_scheme": "sha256(base_seed\\0target_id)-low63",
        "sde_tau": args.sde_tau,
        "sde_eps": args.sde_eps,
        "sde_w_cutoff": args.sde_w_cutoff,
        "sde_log_timesteps": args.sde_log_timesteps,
        "max_batch_residues": args.max_batch_residues,
        "targets": len(targets),
        "sampled": len(results),
        "missing_embeddings": sorted(missing),
        "geometry_means": {
            k: mean(k)
            for k in (
                "ca_ca_mean_A",
                "ca_ca_rmsd_from_ideal_A",
                "ca_ca_frac_within_0.5A",
                "bond_N-CA_rmsd_from_ideal_A",
                "bond_CA-C_rmsd_from_ideal_A",
                "bond_C-O_rmsd_from_ideal_A",
                "clashes_per_1000_atoms",
                "soft_clashes_per_1000_atoms",
                "mean_clash_overlap_A",
                "clash_surrogate_A_per_atom",
                "radius_of_gyration_A",
            )
        },
        "note": (
            "Reference-free geometry only. Accuracy against deposited structures "
            "requires reference PDBs, which this repository does not hold; report "
            "coverage per benchmarks/BENCHMARK_POLICY.md when they are added."
        ),
        "per_target": results,
    }
    (out_dir / "rollout.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[rollout] {len(results)}/{len(targets)} sampled -> {out_dir}")
    for k, v in summary["geometry_means"].items():
        if v is not None:
            print(f"  {k:34} {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
