"""Compare two (or more) benchmark runs side-by-side and print a markdown report.

Reads scores.json files produced by benchmarks/score.py and emits:
  • A table of overall mean metrics per run, split by mono / multi
  • A length-bin breakdown (≤256, 256-512, 512-1024, 1024-2048)
  • Top-K largest changes per metric for the FIRST two runs (regressions + wins)

Pure stdlib + numpy. Runs in either venv (no DockQ etc. needed here).

Usage:
  python benchmarks/compare.py \
      benchmarks/results/<phase2>_t1/scores.json \
      benchmarks/results/<phase3>_t1/scores.json \
      [--out benchmarks/results/compare_p2_p3.md] \
      [--top_k 10]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

L_BINS = [(0, 256), (256, 512), (512, 1024), (1024, 2048)]


def _is_finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _get(row: dict, key: str):
    """Read a metric, including nested dockq.dockq_mean."""
    if key == "dockq":
        d = row.get("dockq")
        return d.get("dockq_mean") if isinstance(d, dict) else None
    return row.get(key)


METRICS_MONO = [
    ("ca_lddt",   "↑"),
    ("tm_score",  "↑"),
    ("ca_rmsd",   "↓"),
]
METRICS_MULTI = [
    ("ca_lddt",        "↑"),
    ("tm_score",       "↑"),  # mean over chains for multimers
    ("interface_lddt", "↑"),
    ("dockq",          "↑"),
    ("ca_rmsd",        "↓"),
]


def mean_or_nan(values):
    vs = [float(v) for v in values if _is_finite(v)]
    return float(np.mean(vs)) if vs else float("nan")


def fmt(v, fmt_spec=".3f"):
    if not _is_finite(v):
        return " nan "
    return f"{v:{fmt_spec}}"


def load_run(path: Path) -> tuple[str, list[dict]]:
    rows = json.loads(path.read_text())
    label = path.parent.name  # uses run dir name as label
    return label, rows


def overall_table(runs: list[tuple[str, list[dict]]]) -> str:
    lines = []
    lines.append(f"## Overall ({len(runs)} runs)")
    lines.append("")

    for kind, metrics, fname in [
        ("monomer",  METRICS_MONO,  lambda r: r["n_chains"] == 1),
        ("multimer", METRICS_MULTI, lambda r: r["n_chains"]  > 1),
    ]:
        header = ["run", "N"] + [f"{m}{a}" for m, a in metrics]
        lines.append(f"### {kind}")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for label, rows in runs:
            sub = [r for r in rows if fname(r)]
            cells = [label, str(len(sub))]
            for m, _ in metrics:
                cells.append(fmt(mean_or_nan(_get(r, m) for r in sub)))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def lbin_table(runs: list[tuple[str, list[dict]]], focus_metric: str = "ca_lddt") -> str:
    lines = [f"## Length bin × run ({focus_metric}, monomer+multimer combined)", ""]
    header = ["run"] + [f"L≤{hi}" for _, hi in L_BINS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for label, rows in runs:
        cells = [label]
        for lo, hi in L_BINS:
            sub = [r for r in rows if lo < r["n_residues"] <= hi]
            cells.append(fmt(mean_or_nan(_get(r, focus_metric) for r in sub)) + f" (n={len(sub)})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def top_changes(runs: list[tuple[str, list[dict]]], top_k: int) -> str:
    if len(runs) < 2:
        return ""
    (la, ra), (lb, rb) = runs[0], runs[1]
    by_id_a = {r["pdb_id"]: r for r in ra}
    by_id_b = {r["pdb_id"]: r for r in rb}
    common = sorted(set(by_id_a) & set(by_id_b))

    out = [f"## Top {top_k} per-target changes  ({lb} − {la})", ""]

    for metric, arrow in [("ca_lddt", "↑"), ("interface_lddt", "↑"), ("dockq", "↑"), ("ca_rmsd", "↓")]:
        diffs = []
        for pid in common:
            va = _get(by_id_a[pid], metric)
            vb = _get(by_id_b[pid], metric)
            if not (_is_finite(va) and _is_finite(vb)):
                continue
            diffs.append((pid, float(vb) - float(va), float(va), float(vb), by_id_a[pid]["n_chains"]))
        if not diffs:
            continue
        # For "↑" metrics: positive Δ is good (improvement); negative is regression
        # For "↓" metrics: invert
        sign = +1 if arrow == "↑" else -1
        diffs_signed = [(p, sign * d, va, vb, c) for (p, d, va, vb, c) in diffs]
        diffs_signed.sort(key=lambda t: t[1], reverse=True)

        out.append(f"### {metric}{arrow}")
        out.append("**Improved:**")
        out.append("| pdb | n_chains | A | B | Δ |")
        out.append("|---|---|---|---|---|")
        for p, _signed, va, vb, c in diffs_signed[:top_k]:
            d = vb - va
            out.append(f"| {p} | {c} | {va:.3f} | {vb:.3f} | {d:+.3f} |")
        out.append("")
        out.append("**Regressed:**")
        out.append("| pdb | n_chains | A | B | Δ |")
        out.append("|---|---|---|---|---|")
        for p, _signed, va, vb, c in diffs_signed[-top_k:][::-1]:
            d = vb - va
            out.append(f"| {p} | {c} | {va:.3f} | {vb:.3f} | {d:+.3f} |")
        out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", nargs="+", type=Path,
                    help="Two or more scores.json paths produced by score.py")
    ap.add_argument("--out", type=Path, default=None, help="write markdown here (default: stdout)")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--lbin_metric", default="ca_lddt")
    args = ap.parse_args()

    runs = [load_run(p) for p in args.scores]

    parts = [
        f"# MambaFold benchmark comparison",
        "",
        "Run sources:",
        *[f"- `{label}` ({len(rows)} targets) — `{path}`" for (label, rows), path in zip(runs, args.scores)],
        "",
        overall_table(runs),
        lbin_table(runs, focus_metric=args.lbin_metric),
        top_changes(runs, args.top_k),
    ]
    text = "\n".join(parts)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"[done] markdown → {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
