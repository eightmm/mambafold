#!/usr/bin/env python
"""Prepare official CASP target structures as MambaFold npz records.

Inputs are official Prediction Center target tarballs containing PDB files.
Output layout matches benchmark inference lookup:

    <out_dir>/<target_id[1:3]>/<target_id>.npz

Example:
    PYTHONPATH=src uv run python scripts/prepare_casp_targets.py \
        --tar data/casp_official/raw/casp14.targets.T.public_11.29.2020.tar.gz \
        --tar data/casp_official/raw/casp15.targets.TS-domains.public_12.20.2022.tar.gz \
        --out_dir data/casp_official/npz \
        --ids_out data/casp_official/casp14_15_ids.txt
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Bio.PDB import PDBParser  # noqa: E402
from pdb_to_npz import convert  # noqa: E402


def target_id_from_member(name: str) -> str:
    return Path(name).name.removesuffix(".pdb").lower()


def parse_pdb_bytes(payload: bytes, target_id: str):
    text = payload.decode("utf-8", errors="replace")
    parser = PDBParser(QUIET=True)
    return parser.get_structure(target_id, io.StringIO(text))


def convert_tar(
    tar_path: Path,
    out_dir: Path,
    allowed_ids: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    failures: list[str] = []
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".pdb")]
        for member in sorted(members, key=lambda m: m.name):
            tid = target_id_from_member(member.name)
            if allowed_ids is not None and tid not in allowed_ids:
                continue
            out_path = out_dir / tid[1:3] / f"{tid}.npz"
            if out_path.exists() and out_path.stat().st_size > 0:
                ok.append(tid)
                continue
            try:
                fh = tf.extractfile(member)
                if fh is None:
                    raise ValueError("missing tar member payload")
                structure = parse_pdb_bytes(fh.read(), tid)
                arrays = convert(structure, tid)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_path, **arrays)
                ok.append(tid)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{tid}\t{type(exc).__name__}: {exc}")
    return ok, failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", action="append", required=True, help="CASP target .tar.gz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ids_out", required=True)
    ap.add_argument(
        "--ids_file",
        default=None,
        help="Optional target ID allow-list. Requested IDs must all be present.",
    )
    ap.add_argument("--fail_out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    failures: list[str] = []
    allowed_ids = None
    if args.ids_file:
        allowed_ids = {
            line.strip().lower()
            for line in Path(args.ids_file).read_text().splitlines()
            if line.strip()
        }

    for raw_tar in args.tar:
        tar_path = Path(raw_tar)
        ok, fail = convert_tar(tar_path, out_dir, allowed_ids)
        ids.extend(ok)
        failures.extend(fail)
        print(f"{tar_path}: ok={len(ok)} fail={len(fail)}", flush=True)

    ids = sorted(set(ids))
    Path(args.ids_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ids_out).write_text("\n".join(ids) + ("\n" if ids else ""))

    fail_out = (
        Path(args.fail_out) if args.fail_out else Path(args.ids_out).with_suffix(".failed.tsv")
    )
    fail_out.write_text("target_id\terror\n" + "\n".join(failures) + ("\n" if failures else ""))

    print(f"wrote ids: {args.ids_out} n={len(ids)}")
    print(f"failures: {fail_out} n={len(failures)}")

    if allowed_ids is not None and set(ids) != allowed_ids:
        missing = sorted(allowed_ids - set(ids))
        unexpected = sorted(set(ids) - allowed_ids)
        raise RuntimeError(
            f"Target allow-list mismatch: missing={missing}, unexpected={unexpected}"
        )


if __name__ == "__main__":
    main()
