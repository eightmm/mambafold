"""Parse CASP15 multimer reference structures into a benchmark manifest.

Reads `benchmarks/casp15/raw/oligo/*.pdb` (heteromers H*.pdb + homomers
T*o.pdb) and emits:

  benchmarks/casp15/manifest.tsv     # one row per target
      target_id, kind, n_chains, total_residues, chain_ids, chain_lengths,
      stoichiometry, gt_pdb_path

  benchmarks/casp15/sequences/<target>.fasta
      one FASTA per target, with one record per chain (chain id in header)

  benchmarks/casp15/targets_protein_multimer.txt
      list of target IDs we'll evaluate on (filters out RNA, alt versions, etc.)

Heuristics:
  - kind = "heteromer" if the PDB has 2+ distinct sequences, else "homomer"
  - stoichiometry e.g. "A2" (homo-dimer of one entity), "A1B1" (hetero-dimer)
  - Skip "v1" / "v2" alt versions for now (use the canonical one only) —
    these are alternative reference assemblies that CASP scores separately

Run after `download.sh`. Pure stdlib + numpy.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

# 3-letter → 1-letter amino acid (covers the 20 standard AAs only;
# non-standard residues become "X")
_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",  # selenomethionine → methionine
    "SEC": "U", "PYL": "O",
}

REPO = Path(__file__).resolve().parents[2]
RAW = Path(__file__).resolve().parent / "raw" / "oligo"
OUT_DIR = Path(__file__).resolve().parent
SEQ_DIR = OUT_DIR / "sequences"
MANIFEST = OUT_DIR / "manifest.tsv"
TARGETS_FILE = OUT_DIR / "targets_protein_multimer.txt"

# Skip alternative versions (CASP "vN" suffix) — keep the canonical one only.
ALT_RE = re.compile(r"v\d+$")


def parse_pdb_chains(pdb_path: Path):
    """Return OrderedDict {chain: sequence} based on Cα-bearing standard residues.

    Iterates PDB ATOM records, picking the residue's 3-letter name once per
    (chain, resseq, icode) triple and concatenating to a one-letter sequence.
    Anything outside the 20 standard set becomes 'X'.
    """
    chains: OrderedDict[str, list[str]] = OrderedDict()
    seen_res: set = set()
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        if atom != "CA":  # one residue → one letter, anchored on Cα
            continue
        res_name = line[17:20].strip()
        chain = line[21]
        res_seq = line[22:26].strip()
        icode = line[26]
        key = (chain, res_seq, icode)
        if key in seen_res:
            continue
        seen_res.add(key)
        chains.setdefault(chain, []).append(_AA3TO1.get(res_name, "X"))
    return OrderedDict((c, "".join(s)) for c, s in chains.items())


def stoichiometry_str(chain_seqs: dict[str, str]) -> str:
    """Return e.g. 'A2' (homo-dimer), 'A1B1' (hetero-dimer), 'A2B2' (...)."""
    # Group chains by sequence identity → entities
    seen: list[tuple[str, int]] = []  # [(seq, count), ...] in encounter order
    for seq in chain_seqs.values():
        for i, (s, n) in enumerate(seen):
            if s == seq:
                seen[i] = (s, n + 1)
                break
        else:
            seen.append((seq, 1))
    return "".join(f"{chr(ord('A') + i)}{n}" for i, (_, n) in enumerate(seen))


def main():
    SEQ_DIR.mkdir(exist_ok=True)
    pdbs = sorted(RAW.glob("*.pdb"))
    print(f"[scan] {len(pdbs)} PDB files in {RAW.relative_to(REPO)}")

    rows: list[dict] = []
    skipped_alt = 0
    skipped_empty = 0
    for pdb in pdbs:
        target = pdb.stem
        # T-suffix 'o' = oligomer marker; strip for the canonical id but keep
        # both the file name (with 'o') and the cleaned id.
        canonical = target.rstrip("o") if target.startswith("T") and target.endswith("o") else target

        # Skip "vN" alt versions (use canonical only)
        if ALT_RE.search(canonical):
            skipped_alt += 1
            continue

        chain_seqs = parse_pdb_chains(pdb)
        if not chain_seqs:
            skipped_empty += 1
            continue

        chain_ids = list(chain_seqs.keys())
        chain_lens = [len(s) for s in chain_seqs.values()]
        total_L = sum(chain_lens)

        # Need ≥ 2 chains and at least one chain ≥ 10 residues (filter dust)
        if len(chain_ids) < 2 or total_L < 20:
            continue

        # Distinct entities
        n_entities = len({s for s in chain_seqs.values()})
        kind = "homomer" if n_entities == 1 else "heteromer"

        # Per-chain FASTA
        fa_lines = []
        for c, s in chain_seqs.items():
            fa_lines.append(f">{canonical}|{c}")
            fa_lines.append(s)
        (SEQ_DIR / f"{canonical}.fasta").write_text("\n".join(fa_lines) + "\n")

        rows.append({
            "target_id": canonical,
            "kind": kind,
            "n_chains": len(chain_ids),
            "total_residues": total_L,
            "chain_ids": ",".join(chain_ids),
            "chain_lengths": ",".join(str(x) for x in chain_lens),
            "stoichiometry": stoichiometry_str(chain_seqs),
            "gt_pdb_path": str(pdb.resolve().relative_to(REPO)),
        })

    rows.sort(key=lambda r: (r["kind"], r["target_id"]))

    cols = ["target_id", "kind", "n_chains", "total_residues",
            "stoichiometry", "chain_ids", "chain_lengths", "gt_pdb_path"]
    MANIFEST.write_text(
        "\t".join(cols) + "\n" +
        "\n".join("\t".join(str(r[c]) for c in cols) for r in rows) + "\n"
    )

    TARGETS_FILE.write_text("\n".join(r["target_id"] for r in rows) + "\n")

    n_homo = sum(1 for r in rows if r["kind"] == "homomer")
    n_hetero = sum(1 for r in rows if r["kind"] == "heteromer")
    print(f"[done] kept {len(rows)}: heteromers={n_hetero} homomers={n_homo}  "
          f"(skipped alt={skipped_alt} empty={skipped_empty})")
    print(f"  manifest : {MANIFEST.relative_to(REPO)}")
    print(f"  targets  : {TARGETS_FILE.relative_to(REPO)}")
    print(f"  fasta    : {SEQ_DIR.relative_to(REPO)}/<target>.fasta")

    print("\nStoichiometry distribution:")
    from collections import Counter
    sc = Counter(r["stoichiometry"] for r in rows)
    for s, n in sc.most_common():
        print(f"  {s}: {n}")
    print("\nChain count distribution:")
    cc = Counter(r["n_chains"] for r in rows)
    for c in sorted(cc):
        print(f"  {c} chains: {cc[c]}")


if __name__ == "__main__":
    main()
