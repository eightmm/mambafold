#!/usr/bin/env python3
"""Paired K=1/2/3 development rollouts from a frozen EMA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.audit_sequence_overlap import file_digest, read_fasta, sequence_digest  # noqa: E402

LENGTH_BINS = ((64, 128), (129, 256), (257, 512), (513, 1024))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def set_loop_budget(model, loops: int) -> None:
    if type(loops) is not int or loops not in (1, 2, 3):
        raise ValueError("comparison requires K in {1,2,3}")
    branch = getattr(model, "latent_residual", None)
    if branch is None or model.latent_position != "post":
        raise ValueError("comparison requires a post-trunk latent branch")
    # K has no parameter keys; the depth grid reads this value on every forward.
    branch.n_loops = loops


def case_order(index: int) -> tuple[int, ...]:
    return ((2, 1, 3), (1, 3, 2), (3, 2, 1))[index % 3]


def prepare(args) -> None:
    import torch

    from mambafold.data.constants import AA_3TO1, AA_TO_ID, CA_ATOM_ID
    from mambafold.data.dataset import RCSBDataset
    from mambafold.structure_io import write_pdb

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs").mkdir(exist_ok=True)
    (out / "references").mkdir(exist_ok=True)
    training = set()
    training_sources = []
    for path in args.training:
        records = read_fasta(path)
        training.update(sequence_digest(record.sequence) for record in records)
        training_sources.append(
            dict(path=str(path), records=len(records), sha256=file_digest(path))
        )
        print(f"Training inventory: {path.name}, records={len(records)}", flush=True)
        del records
    dataset = RCSBDataset(str(args.data), max_length=1024, file_list=str(args.ids))
    candidates = sorted(
        dataset.files, key=lambda p: hashlib.sha256(f"{args.panel_seed}:{p.name}".encode()).digest()
    )
    counts = Counter()
    selected = []
    seen = set()
    rejected = Counter()
    files_examined = 0
    for path in candidates[: args.max_entries]:
        files_examined += 1
        with np.load(path) as raw:
            origin = -1
            for chain in raw["chains"]:
                if chain["mol_type"] != 0:
                    continue
                origin += 1
                start = int(chain["res_idx"])
                residues = raw["residues"][start : start + int(chain["res_num"])]
                names = [
                    str(r["name"])
                    for r in residues
                    if r["is_standard"] and r["name"] in AA_TO_ID and r["name"] != "UNK"
                ]
                sequence = "".join(AA_3TO1[name] for name in names)
                length = len(sequence)
                bin_index = next(
                    (i for i, (lo, hi) in enumerate(LENGTH_BINS) if lo <= length <= hi), None
                )
                if bin_index is None or counts[bin_index] >= args.per_bin:
                    continue
                digest = sequence_digest(sequence)
                if digest in seen:
                    rejected["duplicate_sequence"] += 1
                    continue
                if digest in training:
                    rejected["exact_training_match"] += 1
                    continue
                reference = dataset._canonicalize(raw, path, only_chain_origin=origin)
                if reference is None or reference.seq_len != length:
                    rejected["canonicalization"] += 1
                    continue
                observed = reference.atom_mask & reference.observed_mask
                if observed[:, CA_ATOM_ID].float().mean().item() < 0.9:
                    rejected["observed_ca_below_90pct"] += 1
                    continue
                target = f"{path.stem}_ch{origin}"
                gt = out / "references" / f"{target}.pdb"
                write_pdb(
                    reference.coords.numpy(),
                    reference.res_type.numpy(),
                    observed.numpy(),
                    np.zeros((length, 14)),
                    np.zeros(length, dtype=np.int64),
                    gt,
                )
                selected.append(
                    dict(
                        target=target,
                        sequence=sequence,
                        length=length,
                        sequence_sha256=digest,
                        bin_index=bin_index,
                        source=str(path),
                        source_sha256=file_digest(path),
                        reference_sha256=file_digest(gt),
                    )
                )
                seen.add(digest)
                counts[bin_index] += 1
                print(f"SELECT {target} L={length} bins={dict(counts)}", flush=True)
        if len(selected) == args.per_bin * len(LENGTH_BINS):
            break
    write_json(
        out / "selection_scan.json",
        dict(
            files_examined=files_examined,
            max_entries=args.max_entries,
            rejections=dict(rejected),
            bin_counts=dict(counts),
        ),
    )
    if len(selected) != args.per_bin * len(LENGTH_BINS):
        raise RuntimeError(
            f"incomplete length-stratified panel: {dict(counts)}; no rollout submitted"
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if checkpoint.get("step") != 10000 or "ema" not in checkpoint:
        raise ValueError("expected the preserved 10k EMA checkpoint")
    torch.save(
        dict(
            model=checkpoint["ema"],
            args=checkpoint["args"],
            step=checkpoint["step"],
            mamba3_runtime=checkpoint.get("mamba3_runtime"),
        ),
        out / "ema.pt",
    )
    panel = dict(
        schema_version=1,
        checkpoint_step=10000,
        weights="EMA",
        checkpoint_sha256=file_digest(args.checkpoint),
        ema_sha256=file_digest(out / "ema.pt"),
        heldout_ids_sha256=file_digest(args.ids),
        training_sources=training_sources,
        panel_seed=args.panel_seed,
        length_bins=LENGTH_BINS,
        targets=selected,
        base_seeds=[0, 1],
        loops=[1, 2, 3],
        n_steps=500,
        method="sde",
        sde_tau=0.01,
        scope="post-cutoff development pilot; exact training matches excluded; homology annotated",
    )
    write_json(out / "panel.json", panel)
    (out / "panel.fasta").write_text(
        "".join(f">{r['target']}\n{r['sequence']}\n" for r in selected)
    )
    query = out / "queries"
    query.mkdir(exist_ok=True)
    (query / "panel.fasta").write_bytes((out / "panel.fasta").read_bytes())
    print(f"Prepared {len(selected)} targets, two seeds, three loop budgets", flush=True)


def condition(args) -> None:
    from mambafold.data.sequence_cache import sequence_embedding_path

    panel = json.loads((args.out / "panel.json").read_text())
    original = args.out / "selection_panel.json"
    if original.exists():
        raise FileExistsError("conditioning already finalized")
    for target in panel["targets"]:
        embedding = next(
            (
                p
                for root in args.esm
                if (p := sequence_embedding_path(root, target["sequence"])).is_file()
            ),
            None,
        )
        if embedding is None:
            raise FileNotFoundError(f"missing conditioning: {target['target']}")
        esm = np.load(embedding, mmap_mode="r")
        if esm.shape != (target["length"], 2560) or not np.isfinite(esm).all():
            raise ValueError(f"invalid conditioning: {target['target']}")
        packed = args.out / "inputs" / f"{target['target']}.npz"
        np.savez(packed, esm=np.asarray(esm, dtype=np.float32))
        target["embedding_sha256"] = file_digest(embedding)
        target["packed_sha256"] = file_digest(packed)
    original.write_bytes((args.out / "panel.json").read_bytes())
    write_json(args.out / "panel.json", panel)


def infer(args) -> None:
    import torch

    from mambafold.sampling import prepare_inference_batch, sample
    from mambafold.structure_io import write_pdb
    from mambafold.train.trainer import build_model
    from scripts.rollout import benchmark_seeds, example_from_sequence

    out = args.out
    panel = json.loads((out / "panel.json").read_text())
    if file_digest(out / "ema.pt") != panel["ema_sha256"]:
        raise ValueError("frozen EMA changed")
    state = torch.load(out / "ema.pt", map_location="cpu", mmap=True, weights_only=False)
    model = build_model(state["args"], args.device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    del state
    device = torch.device(args.device)
    identity = dict(
        device=torch.cuda.get_device_name(device),
        torch=torch.__version__,
        n_steps=panel["n_steps"],
        model_evaluations=panel["n_steps"] + 1,
        precision="bf16 autocast",
        batch_size=1,
        timing_scope=(
            "sampler wall time including final CPU copy; "
            "excludes loading, JIT/warmup, PDB and scoring"
        ),
    )
    write_json(out / "device.json", identity)
    rows = []
    counter = 0
    for target in panel["targets"]:
        path = out / "inputs" / f"{target['target']}.npz"
        if file_digest(path) != target["packed_sha256"]:
            raise ValueError("frozen conditioning changed")
        with np.load(path) as data:
            esm = torch.from_numpy(data["esm"].copy())
        example = example_from_sequence(target["sequence"], esm)
        assert torch.count_nonzero(example.coords) == 0
        batch = prepare_inference_batch([example], device, length_bin=64)
        for base_seed in panel["base_seeds"]:
            seed = benchmark_seeds(base_seed, [target["target"]])[0]
            for loops in case_order(counter):
                set_loop_budget(model, loops)
                torch.cuda.empty_cache()
                calls = Counter()
                handles = [
                    model.latent_residual.stack.register_forward_hook(
                        lambda *unused: calls.update(stack=1)
                    ),
                    model.atom_decoder.register_forward_hook(
                        lambda *unused: calls.update(decoder=1)
                    ),
                ]
                try:
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        model(batch)
                finally:
                    for handle in handles:
                        handle.remove()
                if calls != Counter(stack=loops, decoder=1):
                    raise RuntimeError(f"loop/readout override failed: {loops=} {calls=}")
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    model(batch)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                result = sample(
                    model,
                    batch,
                    [seed],
                    n_steps=panel["n_steps"],
                    method="sde",
                    sde_tau=panel["sde_tau"],
                    return_trunk_latent=False,
                )
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                row = dict(
                    target=target["target"],
                    base_seed=base_seed,
                    seed=seed,
                    K=loops,
                    length=target["length"],
                    seconds=seconds,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated(device) / 1024**3,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved(device) / 1024**3,
                    observed_stack_calls=calls["stack"],
                    observed_decoder_calls=calls["decoder"],
                )
                pairs = out / f"K{loops}" / "pairs"
                pairs.mkdir(parents=True, exist_ok=True)
                name = f"{target['target']}_seed{base_seed}"
                prediction = pairs / f"{name}_pred.pdb"
                write_pdb(
                    result.final_aa[0, : target["length"]].numpy(),
                    example.res_type.numpy(),
                    example.atom_mask.numpy(),
                    np.zeros((target["length"], 14)),
                    np.zeros(target["length"], dtype=np.int64),
                    prediction,
                )
                reference = pairs / f"{name}_gt.pdb"
                if not reference.exists():
                    reference.symlink_to((out / "references" / f"{target['target']}.pdb").resolve())
                row["prediction_sha256"] = file_digest(prediction)
                rows.append(row)
                write_json(out / "timing.json", dict(device=identity, rows=rows))
                print(json.dumps(row), flush=True)
                del result
            counter += 1
        del batch
    expected = len(panel["targets"]) * len(panel["base_seeds"])
    for loops in panel["loops"]:
        selected = [row for row in rows if row["K"] == loops]
        if len(selected) != expected:
            raise RuntimeError("incomplete inference coverage")
        write_json(
            out / f"K{loops}" / "pairs/manifest.json",
            dict(
                panel_sha256=file_digest(out / "panel.json"),
                K=loops,
                expected=expected,
                rows=selected,
            ),
        )


def summarize(args) -> None:
    out = args.out
    timing = json.loads((out / "timing.json").read_text())
    panel = json.loads((out / "panel.json").read_text())
    summary = {}
    target_metrics = {}
    expected = {(t["target"], seed) for t in panel["targets"] for seed in panel["base_seeds"]}
    for loops in panel["loops"]:
        scored = json.loads((out / f"K{loops}/scores/summary.json").read_text())
        if not scored["evaluation_complete"]:
            raise ValueError("refusing an incomplete scored comparison")
        rows = [row for row in timing["rows"] if row["K"] == loops]
        if len(rows) != len(expected) or {(r["target"], r["base_seed"]) for r in rows} != expected:
            raise ValueError("incomplete or duplicate timing coverage")
        expected_ids = {f"{target}_seed{seed}" for target, seed in expected}
        if (
            len(scored["rows"]) != len(expected_ids)
            or {r["target_id"] for r in scored["rows"]} != expected_ids
        ):
            raise ValueError("incomplete or duplicate scored coverage")
        summary[str(loops)] = dict(
            targets=len(panel["targets"]),
            samples=len(rows),
            seconds_mean=statistics.mean(r["seconds"] for r in rows),
            seconds_median=statistics.median(r["seconds"] for r in rows),
            peak_allocated_GiB=max(r["peak_allocated_GiB"] for r in rows),
            peak_reserved_GiB=max(r["peak_reserved_GiB"] for r in rows),
            metrics=scored["metrics"],
        )
        target_metrics[loops] = {}
        for target in panel["targets"]:
            target_ids = {f"{target['target']}_seed{seed}" for seed in panel["base_seeds"]}
            matches = [r for r in scored["rows"] if r["target_id"] in target_ids]
            if len(matches) != len(panel["base_seeds"]):
                raise ValueError("missing paired target/seed")
            target_metrics[loops][target["target"]] = {
                metric: statistics.mean(row[metric] for row in matches)
                for metric in ("tm_score", "lddt", "bb_lddt")
            }
    rng = np.random.default_rng(107113)
    paired = {}
    for loops in (1, 3):
        paired[str(loops)] = {}
        for metric in ("tm_score", "lddt", "bb_lddt"):
            delta = np.array(
                [
                    target_metrics[loops][t["target"]][metric]
                    - target_metrics[2][t["target"]][metric]
                    for t in panel["targets"]
                ]
            )
            samples = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
            paired[str(loops)][metric] = dict(
                delta_vs_K2=float(delta.mean()),
                protein_bootstrap_95pct=np.quantile(samples, [0.025, 0.975]).tolist(),
                proteins_improved=int((delta > 0).sum()),
                proteins=len(delta),
            )
    write_json(
        out / "comparison.json",
        dict(
            device=timing["device"],
            checkpoint_step=10000,
            panel_sha256=file_digest(out / "panel.json"),
            summary=summary,
            paired=paired,
            scope=(
                "small, early-checkpoint development pilot; no generalization or convergence claim"
            ),
        ),
    )
    print(json.dumps(dict(summary=summary, paired=paired), indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "condition", "infer", "summarize"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/rcsb_heldout_post_cutoff"))
    parser.add_argument("--ids", type=Path, default=Path("data/splits/heldout_post_cutoff.txt"))
    parser.add_argument(
        "--esm", type=Path, nargs="+", default=[Path("data/rcsb_esmc6b"), Path("data/afdb_esmc6b")]
    )
    parser.add_argument(
        "--training",
        type=Path,
        nargs="+",
        default=[Path("data/audit/rcsb-admitted.fasta"), Path("data/audit/afdb-v4-training.fasta")],
    )
    parser.add_argument("--per-bin", type=int, default=3)
    parser.add_argument("--max-entries", type=int, default=6000)
    parser.add_argument("--panel-seed", type=int, default=20261018)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "prepare" and args.checkpoint is None:
        parser.error("prepare requires --checkpoint")
    {"prepare": prepare, "condition": condition, "infer": infer, "summarize": summarize}[args.mode](
        args
    )


if __name__ == "__main__":
    main()
