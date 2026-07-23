"""Score local geometry of predicted/GT PDB pairs.

Inputs are `<target>_pred.pdb` and `<target>_gt.pdb` pairs from
`benchmarks/run_inference.py`. Metrics are intentionally simple:

- backbone/CB bond MAE over N-CA, CA-C, C-O, CA-CB, and peptide C-N.
- hard heavy-atom clashes below a distance threshold, excluding same-residue
  and adjacent-residue pairs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

IDEAL_A = {
    "N_CA": 1.458,
    "CA_C": 1.525,
    "C_O": 1.229,
    "CA_CB": 1.530,
    "C_N": 1.329,
}


def parse_pdb(path: Path) -> dict[str, Any]:
    residues: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    atoms: list[tuple[tuple[str, int, str], str, np.ndarray]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        chain = line[21]
        resseq = int(line[22:26])
        icode = line[26].strip()
        res_id = (chain, resseq, icode)
        xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float32,
        )
        residues.setdefault(res_id, {})[atom] = xyz
        atoms.append((res_id, atom, xyz))
    return {"residues": residues, "atoms": atoms}


def _bond_errors(pdb: dict[str, Any]) -> list[float]:
    residues = pdb["residues"]
    errors: list[float] = []
    for res_id, atom_map in residues.items():
        for a, b, ideal in (
            ("N", "CA", IDEAL_A["N_CA"]),
            ("CA", "C", IDEAL_A["CA_C"]),
            ("C", "O", IDEAL_A["C_O"]),
            ("CA", "CB", IDEAL_A["CA_CB"]),
        ):
            if a in atom_map and b in atom_map:
                errors.append(abs(float(np.linalg.norm(atom_map[a] - atom_map[b])) - ideal))

    by_chain: dict[str, list[tuple[int, str, dict[str, np.ndarray]]]] = {}
    for (chain, resseq, icode), atom_map in residues.items():
        by_chain.setdefault(chain, []).append((resseq, icode, atom_map))
    for chain_res in by_chain.values():
        chain_res.sort(key=lambda x: (x[0], x[1]))
        for (ri, _, ai), (rj, _, aj) in zip(chain_res[:-1], chain_res[1:]):
            if rj - ri != 1 or "C" not in ai or "N" not in aj:
                continue
            errors.append(abs(float(np.linalg.norm(ai["C"] - aj["N"])) - IDEAL_A["C_N"]))
    return errors


def _clash_count(pdb: dict[str, Any], threshold: float, block: int = 2048) -> tuple[int, int]:
    atoms = pdb["atoms"]
    n = len(atoms)
    if n < 2:
        return 0, n
    coords = np.stack([a[2] for a in atoms]).astype(np.float32)
    res_keys = np.array([f"{a[0][0]}:{a[0][1]}:{a[0][2]}" for a in atoms], dtype=object)
    res_nums = np.array([a[0][1] for a in atoms], dtype=np.int32)
    chains = np.array([a[0][0] for a in atoms], dtype=object)
    clashes = 0
    thr2 = threshold * threshold
    for start in range(0, n, block):
        end = min(start + block, n)
        d = coords[start:end, None, :] - coords[None, :, :]
        d2 = np.einsum("bij,bij->bi", d, d)
        rows = np.arange(start, end)[:, None]
        cols = np.arange(n)[None, :]
        upper = rows < cols
        same_res = res_keys[start:end, None] == res_keys[None, :]
        adjacent = (chains[start:end, None] == chains[None, :]) & (
            np.abs(res_nums[start:end, None] - res_nums[None, :]) <= 1
        )
        mask = upper & (~same_res) & (~adjacent) & (d2 < thr2)
        clashes += int(mask.sum())
    return clashes, n


def score_one(path: Path, clash_threshold: float) -> dict[str, float]:
    pdb = parse_pdb(path)
    bond_errors = np.asarray(_bond_errors(pdb), dtype=np.float32)
    clashes, n_atoms = _clash_count(pdb, clash_threshold)
    if len(bond_errors) == 0:
        bond_mae = bond_p95 = bond_bad_frac = float("nan")
    else:
        bond_mae = float(bond_errors.mean())
        bond_p95 = float(np.percentile(bond_errors, 95))
        bond_bad_frac = float((bond_errors > 0.10).mean())
    return {
        "n_atoms": int(n_atoms),
        "n_bonds": int(len(bond_errors)),
        "bond_mae_A": bond_mae,
        "bond_p95_A": bond_p95,
        "bond_bad_frac_gt_0p10A": bond_bad_frac,
        "clashes": int(clashes),
        "clashes_per_1k_atoms": float(clashes * 1000.0 / max(n_atoms, 1)),
    }


def aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = [
        "bond_mae_A",
        "bond_p95_A",
        "bond_bad_frac_gt_0p10A",
        "clashes_per_1k_atoms",
    ]
    for key in keys:
        vals = np.asarray(
            [r[prefix][key] for r in rows if not math.isnan(r[prefix][key])], dtype=np.float64
        )
        out[f"{prefix}_{key}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{prefix}_{key}_median"] = float(np.median(vals)) if len(vals) else float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--clash_threshold", type=float, default=1.5)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    rows: list[dict[str, Any]] = []
    for gt_path in sorted(in_dir.glob("*_gt.pdb")):
        pid = gt_path.name.removesuffix("_gt.pdb")
        pred_path = in_dir / f"{pid}_pred.pdb"
        if not pred_path.exists():
            continue
        pred = score_one(pred_path, args.clash_threshold)
        gt = score_one(gt_path, args.clash_threshold)
        rows.append({"pdb_id": pid, "pred": pred, "gt": gt})

    summary = {
        "n": len(rows),
        "clash_threshold_A": args.clash_threshold,
        **aggregate(rows, "pred"),
        **aggregate(rows, "gt"),
        "rows": rows,
    }
    out_path = Path(args.out) if args.out else in_dir / "local_geometry.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
