"""Score local geometry of predicted/GT PDB pairs.

Inputs are `<target>_pred.pdb` and `<target>_gt.pdb` pairs from
`benchmarks/run_inference.py`. Metrics are intentionally simple:

- backbone/CB bond MAE over N-CA, CA-C, C-O, CA-CB, and peptide C-N.
- hard heavy-atom clashes below a distance threshold, excluding same-residue
  and adjacent-residue pairs.
- C-alpha stereocentre handedness from the signed N-CA/C-CA/CB-CA volume.
- minimum distance between sequence-distant C-alpha backbone segments, which
  catches X-shaped trace crossings that endpoint-only C-alpha distances miss.
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
NONLOCAL_CA_SEQ_SEP = 12
NONLOCAL_CA_POINT_FLOOR_A = 3.6
NONLOCAL_CA_SEGMENT_FLOOR_A = 2.5
NONLOCAL_CA_SEGMENT_MAX_EDGE_A = 6.0


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


def _ca_chiral_volumes(pdb: dict[str, Any]) -> list[float]:
    """Return normalized C-alpha signed volumes for non-glycine residues.

    With the atom ordering used here, standard L-amino acids have positive
    ``cross(N-CA, C-CA) dot (CB-CA)``. Reflection changes only the sign.
    """
    volumes: list[float] = []
    for atom_map in pdb["residues"].values():
        if not all(atom in atom_map for atom in ("N", "CA", "C", "CB")):
            continue
        ca = atom_map["CA"]
        vectors = [atom_map[atom] - ca for atom in ("N", "C", "CB")]
        norms = [float(np.linalg.norm(vector)) for vector in vectors]
        if min(norms) <= 1e-8:
            volumes.append(0.0)
            continue
        n, c, cb = (vector / norm for vector, norm in zip(vectors, norms))
        volumes.append(float(np.dot(np.cross(n, c), cb)))
    return volumes


def _nonlocal_ca_metrics(
    pdb: dict[str, Any],
    *,
    seq_sep: int = NONLOCAL_CA_SEQ_SEP,
) -> dict[str, float | int]:
    """Measure gross self-overlap between sequence-distant C-alpha atoms."""
    ca_rows = [
        (chain, resseq, atoms["CA"])
        for (chain, resseq, _icode), atoms in pdb["residues"].items()
        if "CA" in atoms
    ]
    if len(ca_rows) < 2:
        return {
            "nonlocal_ca_pairs": 0,
            "nonlocal_ca_min_A": float("nan"),
            "nonlocal_ca_clashes_lt_2A": 0,
            "nonlocal_ca_clashes_lt_3A": 0,
            "nonlocal_ca_clashes_lt_3p6A": 0,
            "nonlocal_ca_penetration_rms_A": 0.0,
        }
    coords = np.stack([row[2] for row in ca_rows]).astype(np.float32)
    chains = np.asarray([row[0] for row in ca_rows], dtype=object)
    resseq = np.asarray([row[1] for row in ca_rows], dtype=np.int32)
    delta = coords[:, None, :] - coords[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    upper = np.triu(np.ones(distance.shape, dtype=bool), k=1)
    same_chain = chains[:, None] == chains[None, :]
    separation = np.abs(resseq[:, None] - resseq[None, :])
    candidate = upper & ((~same_chain) | (separation > seq_sep))
    values = distance[candidate]
    if len(values) == 0:
        return {
            "nonlocal_ca_pairs": 0,
            "nonlocal_ca_min_A": float("nan"),
            "nonlocal_ca_clashes_lt_2A": 0,
            "nonlocal_ca_clashes_lt_3A": 0,
            "nonlocal_ca_clashes_lt_3p6A": 0,
            "nonlocal_ca_penetration_rms_A": 0.0,
        }
    penetration = np.maximum(NONLOCAL_CA_POINT_FLOOR_A - values, 0.0)
    violating = penetration > 0
    penetration_rms = (
        float(np.sqrt(np.mean(np.square(penetration[violating])))) if violating.any() else 0.0
    )
    return {
        "nonlocal_ca_pairs": int(len(values)),
        "nonlocal_ca_min_A": float(values.min()),
        "nonlocal_ca_clashes_lt_2A": int((values < 2.0).sum()),
        "nonlocal_ca_clashes_lt_3A": int((values < 3.0).sum()),
        "nonlocal_ca_clashes_lt_3p6A": int(violating.sum()),
        "nonlocal_ca_penetration_rms_A": penetration_rms,
    }


def _point_to_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """Vectorized point-to-finite-segment distance."""
    direction = end - start
    denom = np.einsum("ij,ij->i", direction, direction)
    projection = np.einsum("ij,ij->i", point - start, direction)
    fraction = np.divide(
        projection,
        denom,
        out=np.zeros_like(projection),
        where=denom > 1e-12,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start + fraction[:, None] * direction
    return np.linalg.norm(point - closest, axis=-1)


def _segment_pair_distances(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
) -> np.ndarray:
    """Exact minimum distances for matched pairs of finite 3-D segments.

    A constrained quadratic reaches its minimum either at an interior
    line-line stationary point or on one of the four boundaries.  The four
    endpoint-to-segment distances cover the boundaries; the fifth candidate
    covers a valid interior solution.  This formulation stays robust for
    parallel and zero-length segments.
    """
    candidates = np.stack(
        (
            _point_to_segment_distance(p0, q0, q1),
            _point_to_segment_distance(p1, q0, q1),
            _point_to_segment_distance(q0, p0, p1),
            _point_to_segment_distance(q1, p0, p1),
        ),
        axis=-1,
    )
    best = candidates.min(axis=-1)

    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = np.einsum("ij,ij->i", u, u)
    b = np.einsum("ij,ij->i", u, v)
    c = np.einsum("ij,ij->i", v, v)
    d = np.einsum("ij,ij->i", u, w)
    e = np.einsum("ij,ij->i", v, w)
    denominator = a * c - b * b
    valid = denominator > 1e-12
    s = np.divide(
        b * e - c * d,
        denominator,
        out=np.zeros_like(denominator),
        where=valid,
    )
    t = np.divide(
        a * e - b * d,
        denominator,
        out=np.zeros_like(denominator),
        where=valid,
    )
    interior = valid & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
    if interior.any():
        delta = w[interior] + s[interior, None] * u[interior]
        delta -= t[interior, None] * v[interior]
        best[interior] = np.minimum(best[interior], np.linalg.norm(delta, axis=-1))
    return best


def _nonlocal_ca_segment_metrics(
    pdb: dict[str, Any],
    *,
    seq_sep: int = NONLOCAL_CA_SEQ_SEP,
    max_edge_A: float = NONLOCAL_CA_SEGMENT_MAX_EDGE_A,
    block: int = 65_536,
) -> dict[str, float | int]:
    """Measure near-crossings of sequence-distant C-alpha trace segments."""
    by_chain: dict[str, list[tuple[int, str, np.ndarray]]] = {}
    for (chain, resseq, icode), atoms in pdb["residues"].items():
        if "CA" in atoms:
            by_chain.setdefault(chain, []).append((resseq, icode, atoms["CA"]))

    segments: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for chain, residues in by_chain.items():
        residues.sort(key=lambda row: (row[0], row[1]))
        for left, right in zip(residues[:-1], residues[1:]):
            if right[0] - left[0] == 1 and float(np.linalg.norm(right[2] - left[2])) <= max_edge_A:
                segments.append((chain, left[0], left[2], right[2]))

    empty = {
        "nonlocal_ca_segment_pairs": 0,
        "nonlocal_ca_segment_min_A": float("nan"),
        "nonlocal_ca_segment_clashes_lt_0p5A": 0,
        "nonlocal_ca_segment_clashes_lt_1A": 0,
        "nonlocal_ca_segment_clashes_lt_2A": 0,
        "nonlocal_ca_segment_clashes_lt_2p5A": 0,
        "nonlocal_ca_segment_clashes_lt_3A": 0,
        "nonlocal_ca_segment_penetration_rms_A": 0.0,
    }
    if len(segments) < 2:
        return empty

    chains = np.asarray([row[0] for row in segments], dtype=object)
    starts = np.asarray([row[1] for row in segments], dtype=np.int32)
    p0 = np.stack([row[2] for row in segments]).astype(np.float32)
    p1 = np.stack([row[3] for row in segments]).astype(np.float32)
    rows, cols = np.triu_indices(len(segments), k=1)
    # Each segment spans start -> start+1.  Require every endpoint pair to be
    # farther than seq_sep, not merely the two segment starts.
    endpoint_separation = np.maximum(np.abs(starts[rows] - starts[cols]) - 1, 0)
    candidate = (chains[rows] != chains[cols]) | (endpoint_separation > seq_sep)
    rows, cols = rows[candidate], cols[candidate]
    if len(rows) == 0:
        return empty

    minima: list[np.ndarray] = []
    for offset in range(0, len(rows), block):
        i = rows[offset : offset + block]
        j = cols[offset : offset + block]
        minima.append(_segment_pair_distances(p0[i], p1[i], p0[j], p1[j]))
    values = np.concatenate(minima)
    penetration = np.maximum(NONLOCAL_CA_SEGMENT_FLOOR_A - values, 0.0)
    violating = penetration > 0.0
    return {
        "nonlocal_ca_segment_pairs": int(len(values)),
        "nonlocal_ca_segment_min_A": float(values.min()),
        "nonlocal_ca_segment_clashes_lt_0p5A": int((values < 0.5).sum()),
        "nonlocal_ca_segment_clashes_lt_1A": int((values < 1.0).sum()),
        "nonlocal_ca_segment_clashes_lt_2A": int((values < 2.0).sum()),
        "nonlocal_ca_segment_clashes_lt_2p5A": int(violating.sum()),
        "nonlocal_ca_segment_clashes_lt_3A": int((values < 3.0).sum()),
        "nonlocal_ca_segment_penetration_rms_A": (
            float(np.sqrt(np.mean(np.square(penetration[violating])))) if violating.any() else 0.0
        ),
    }


def score_one(path: Path, clash_threshold: float) -> dict[str, float | int]:
    pdb = parse_pdb(path)
    bond_errors = np.asarray(_bond_errors(pdb), dtype=np.float32)
    clashes, n_atoms = _clash_count(pdb, clash_threshold)
    chiral_volumes = np.asarray(_ca_chiral_volumes(pdb), dtype=np.float32)
    if len(bond_errors) == 0:
        bond_mae = bond_p95 = bond_bad_frac = float("nan")
    else:
        bond_mae = float(bond_errors.mean())
        bond_p95 = float(np.percentile(bond_errors, 95))
        bond_bad_frac = float((bond_errors > 0.10).mean())
    if len(chiral_volumes) == 0:
        chirality_wrong_frac = chirality_degenerate_frac = chirality_median = float("nan")
    else:
        chirality_wrong_frac = float((chiral_volumes <= 0.0).mean())
        chirality_degenerate_frac = float((np.abs(chiral_volumes) < 0.1).mean())
        chirality_median = float(np.median(chiral_volumes))
    return {
        "n_atoms": int(n_atoms),
        "n_bonds": int(len(bond_errors)),
        "bond_mae_A": bond_mae,
        "bond_p95_A": bond_p95,
        "bond_bad_frac_gt_0p10A": bond_bad_frac,
        "clashes": int(clashes),
        "clashes_per_1k_atoms": float(clashes * 1000.0 / max(n_atoms, 1)),
        "n_ca_chiral_centres": int(len(chiral_volumes)),
        "ca_chirality_wrong_frac": chirality_wrong_frac,
        "ca_chirality_degenerate_frac_lt_0p1": chirality_degenerate_frac,
        "ca_chirality_volume_median": chirality_median,
        **_nonlocal_ca_metrics(pdb),
        **_nonlocal_ca_segment_metrics(pdb),
    }


def aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = [
        "bond_mae_A",
        "bond_p95_A",
        "bond_bad_frac_gt_0p10A",
        "clashes_per_1k_atoms",
        "ca_chirality_wrong_frac",
        "ca_chirality_degenerate_frac_lt_0p1",
        "ca_chirality_volume_median",
        "nonlocal_ca_min_A",
        "nonlocal_ca_clashes_lt_2A",
        "nonlocal_ca_clashes_lt_3A",
        "nonlocal_ca_clashes_lt_3p6A",
        "nonlocal_ca_penetration_rms_A",
        "nonlocal_ca_segment_min_A",
        "nonlocal_ca_segment_clashes_lt_0p5A",
        "nonlocal_ca_segment_clashes_lt_1A",
        "nonlocal_ca_segment_clashes_lt_2A",
        "nonlocal_ca_segment_clashes_lt_2p5A",
        "nonlocal_ca_segment_clashes_lt_3A",
        "nonlocal_ca_segment_penetration_rms_A",
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
        "nonlocal_ca_metric_definition": {
            "sequence_separation_gt": NONLOCAL_CA_SEQ_SEP,
            "point_penetration_floor_A": NONLOCAL_CA_POINT_FLOOR_A,
            "segment_penetration_floor_A": NONLOCAL_CA_SEGMENT_FLOOR_A,
            "segment_max_edge_A": NONLOCAL_CA_SEGMENT_MAX_EDGE_A,
        },
        **aggregate(rows, "pred"),
        **aggregate(rows, "gt"),
        "rows": rows,
    }
    out_path = Path(args.out) if args.out else in_dir / "local_geometry.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
