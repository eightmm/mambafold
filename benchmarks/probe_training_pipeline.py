#!/usr/bin/env python
"""Bounded profiling of the production trainer, real corpus loader and DDP.

Run with torchrun. Training arguments follow ``--``. Checkpoint writes and W&B
are disabled for this diagnostic; all optimizer updates use the real trainer.
Each rank writes a report, then --summarize combines completed rank reports.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import importlib.util
import json
import os
import pstats
import resource
import statistics
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "mambafold").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def summarize(directory: Path) -> dict:
    reports = [json.loads(p.read_text()) for p in sorted(directory.glob("rank*.json"))]
    if not reports or len(reports) != reports[0]["world_size"]:
        raise RuntimeError("Missing rank reports")
    if any(not r["source_unchanged"] or not r["completed"] for r in reports):
        raise RuntimeError("Incomplete run or source changed during measurement")
    steps = [r["steps"] for r in reports]
    if len({len(s) for s in steps}) != 1:
        raise RuntimeError("Ranks completed different update counts")
    warmup = reports[0]["warmup_updates"]
    profiled = reports[0]["profile_update"]
    paired = []
    for rows in zip(*steps):
        index = rows[0]["step"]
        # The last row ends after main() returns, including final rendezvous
        # and process-group teardown. It is not a training-update duration.
        if index <= warmup or index in (profiled, profiled + 1) or index == len(steps[0]):
            continue
        if not all(row["updated"] for row in rows):
            raise RuntimeError("A measured optimizer update was skipped")
        paired.append({
            "step": index,
            "wall_ms_max": max(r["wall_ms"] for r in rows),
            "valid_residues": sum(r["valid_residues"] for r in rows),
            "padded_residues": sum(r["padded_residues"] for r in rows),
            "length_max": max(max(r["lengths"]) for r in rows),
            "loader_wait_ms_max": max(r.get("loader_wait_ms_local", 0) for r in rows),
            "stages_ms_max": {
                name: max(r["stages_ms"].get(name, 0) for r in rows)
                for name in {k for r in rows for k in r["stages_ms"]}
            },
        })
    if not paired:
        raise RuntimeError("No measured updates after warmup/profile exclusions")
    selected = {row["step"] for row in paired}
    duration_s = max(
        sum(row["wall_ms"] for row in rank_steps if row["step"] in selected)
        for rank_steps in steps
    ) / 1000
    summary = {
        "world_size": len(reports), "backend": reports[0]["backend"],
        "device": reports[0]["device"], "updates": len(paired),
        "median_ms": statistics.median(r["wall_ms_max"] for r in paired),
        "mean_ms": duration_s * 1000 / len(paired),
        "mean_step_max_ms": statistics.mean(r["wall_ms_max"] for r in paired),
        "updates_per_second": len(paired) / duration_s,
        "proteins_per_second": len(reports) * len(paired) / duration_s,
        "valid_residues_per_second": sum(r["valid_residues"] for r in paired) / duration_s,
        "padding_fraction": 1 - sum(r["valid_residues"] for r in paired)
        / sum(r["padded_residues"] for r in paired),
        "peak_allocated_gib": max(r["peak_allocated_gib"] for r in reports),
        "stages_median_ms": {
            name: statistics.median(r["stages_ms_max"].get(name, 0) for r in paired)
            for name in {k for r in paired for k in r["stages_ms_max"]}
        },
        "loader_wait_median_ms_max": statistics.median(r["loader_wait_ms_max"] for r in paired),
        "measured_steps": paired, "source_sha256": reports[0]["source_sha256"],
        "scope": "Real mixed-corpus loader, H2D, DDP, loss, optimizer, EMA and stdout; "
        "excludes startup, warmup/profile updates, final shutdown update, "
        "checkpoint writes and W&B.",
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "measured_steps"}))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--train-script", type=Path, default=Path("scripts/train.py"))
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--warmup-updates", type=int, default=6)
    parser.add_argument("--profile-update", type=int, default=8)
    parser.add_argument("--loader-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnose-stalls", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    if not args.report_dir or not torch.cuda.is_available():
        parser.error("a report directory and CUDA allocation are required")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_root.resolve()))
    # Load the chosen package before train.py inserts its checkout into sys.path.
    from mambafold.data.types import ProteinBatch
    from mambafold.train import engine
    from mambafold.train.ema import EMA

    spec = importlib.util.spec_from_file_location("production_train", args.train_script)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    original_digest = source_digest(args.source_root)
    rows = []
    current = None
    events = []
    profiler = None
    backend = os.environ.get("DIST_BACKEND", "nccl")
    active_calls = []
    gc_started = {}
    stop_monitor = threading.Event()
    main_thread = threading.get_ident()
    diagnostic_path = args.report_dir / f"stalls_rank{rank}.jsonl"

    def diagnostic(event):
        with diagnostic_path.open("a") as stream:
            stream.write(json.dumps(dict(unix_time=time.time(), **event)) + "\n")

    def monitor_stalls():
        last_sample = 0.0
        while not stop_monitor.wait(0.1):
            calls = list(active_calls)
            now = time.perf_counter()
            if calls and now - calls[0][1] > 0.5 and now - last_sample > 0.5:
                frame = sys._current_frames().get(main_thread)
                diagnostic(dict(
                    kind="slow_call", step=current["step"] if current else None,
                    stages=[name for name, _ in calls],
                    elapsed_ms=(now - calls[0][1]) * 1000,
                    stack=traceback.format_stack(frame, limit=18) if frame else [],
                ))
                last_sample = now

    def gc_event(phase, info):
        generation = info["generation"]
        if phase == "start":
            gc_started[generation] = time.perf_counter()
        elif generation in gc_started:
            elapsed = (time.perf_counter() - gc_started.pop(generation)) * 1000
            if current is not None:
                current["gc_ms"] = current.get("gc_ms", 0) + elapsed
                if elapsed > 10:
                    diagnostic(dict(kind="gc", step=current["step"], elapsed_ms=elapsed,
                                    generation=generation, collected=info["collected"],
                                    stages=[name for name, _ in active_calls],
                                    stack=traceback.format_stack(limit=18)))

    if args.diagnose_stalls:
        gc.callbacks.append(gc_event)
        threading.Thread(target=monitor_stalls, daemon=True).start()
        if hasattr(trainer, "freeze_training_setup"):
            original_freeze = trainer.freeze_training_setup

            def full_collection_ms():
                started = time.perf_counter()
                gc.collect(2)
                return (time.perf_counter() - started) * 1000

            @contextmanager
            def calibrated_freeze():
                # Startup-only counterfactual: compare a clean full-heap scan
                # with the same persistent roots in the permanent generation.
                cold = full_collection_ms()
                clean = full_collection_ms()
                previous = gc.get_freeze_count()
                with original_freeze() as count:
                    if count <= previous:
                        raise RuntimeError("GC correction did not freeze the training setup")
                    frozen = full_collection_ms()
                    (args.report_dir / f"gc_calibration_rank{rank}.json").write_text(json.dumps(
                        dict(cold_ms=cold, clean_ms=clean, frozen_ms=frozen,
                             frozen_objects=count), indent=2,
                    ) + "\n")
                    yield count

            trainer.freeze_training_setup = calibrated_freeze
    prewarmed_lengths = []
    original_prewarm = trainer.prewarm_sequence_kernels

    @wraps(original_prewarm)
    def prewarm(*a, **kw):
        prewarmed_lengths.extend(original_prewarm(*a, **kw))
        return prewarmed_lengths

    trainer.prewarm_sequence_kernels = prewarm
    original_loader = trainer.build_dataloaders
    loader_startup = None

    def build_dataloaders(*a, **kw):
        nonlocal loader_startup
        started = time.perf_counter()
        if args.loader_profile:
            with cProfile.Profile() as profile:
                loader, sampler, dataset = original_loader(*a, **kw)
            with (args.report_dir / f"loader_profile_rank{rank}.txt").open("w") as stream:
                pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(30)
        else:
            loader, sampler, dataset = original_loader(*a, **kw)
        loader_startup = time.perf_counter() - started

        class TimedLoader:
            def __len__(self):
                return len(loader)

            def __iter__(self):
                iterator = iter(loader)
                while True:
                    started = time.perf_counter()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        return
                    if current is not None:
                        current["loader_wait_ms_local"] = (
                            current.get("loader_wait_ms_local", 0)
                            + (time.perf_counter() - started) * 1000
                        )
                    yield batch

        return TimedLoader(), sampler, dataset

    trainer.build_dataloaders = build_dataloaders

    def measured(name, fn):
        @wraps(fn)
        def call(*a, **kw):
            if current is None:
                return fn(*a, **kw)
            started = time.perf_counter()
            cpu_started = time.thread_time()
            active_calls.append((name, started))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                with torch.profiler.record_function("pipeline/" + name):
                    result = fn(*a, **kw)
                end.record()
                events.append((name, start, end))
            finally:
                active_calls.pop()
                for key, elapsed in (
                    ("host_wall_ms", (time.perf_counter() - started) * 1000),
                    ("host_cpu_ms", (time.thread_time() - cpu_started) * 1000),
                ):
                    stages = current.setdefault(key, {})
                    stages[name] = stages.get(name, 0) + elapsed
            return result
        return call

    def finish():
        nonlocal current, profiler
        if current is None:
            return
        torch.cuda.synchronize()
        current["wall_ms"] = (time.perf_counter() - current.pop("started")) * 1000
        usage = resource.getrusage(resource.RUSAGE_SELF)
        current["resource_delta"] = {
            key: getattr(usage, key) - value
            for key, value in current.pop("resource_start").items()
        }
        memory = torch.cuda.memory_stats()
        current["allocator_delta"] = {
            key: memory.get(key, 0) - value
            for key, value in current.pop("allocator_start").items()
        }
        current["stages_ms"] = {}
        for name, start, end in events:
            current["stages_ms"][name] = (
                current["stages_ms"].get(name, 0) + start.elapsed_time(end)
            )
        rows.append(current)
        current = None
        events.clear()
        if profiler is not None:
            profiler.stop()
            averages = profiler.key_averages()
            (args.report_dir / f"profile_rank{rank}.txt").write_text(
                averages.table(sort_by="self_cuda_time_total", row_limit=60)
            )
            (args.report_dir / f"profile_rank{rank}.json").write_text(json.dumps([
                dict(key=e.key, calls=e.count, cpu_us=e.cpu_time_total,
                     self_cpu_us=e.self_cpu_time_total, device_us=e.device_time_total,
                     self_device_us=e.self_device_time_total)
                for e in averages
            ], indent=2))
            profiler = None

    original_optimizer = trainer.build_optimizer

    def build_optimizer(model, cfg):
        optimizer = original_optimizer(model, cfg)
        zero = optimizer.zero_grad

        @wraps(zero)
        def zero_grad(*a, **kw):
            nonlocal current, profiler
            finish()
            step = len(rows) + 1
            if step == args.warmup_updates + 1:
                torch.cuda.reset_peak_memory_stats()
            if step == args.profile_update:
                profiler = torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
                ])
                profiler.start()
            current = dict(step=step, started=time.perf_counter(), updated=False,
                           valid_residues=0, padded_residues=0, lengths=[], data_sha256=[])
            usage = resource.getrusage(resource.RUSAGE_SELF)
            current["resource_start"] = {key: getattr(usage, key) for key in (
                "ru_utime", "ru_stime", "ru_minflt", "ru_majflt", "ru_nvcsw", "ru_nivcsw",
            )}
            memory = torch.cuda.memory_stats()
            current["allocator_start"] = {key: memory.get(key, 0) for key in (
                "num_alloc_retries", "num_ooms", "num_device_alloc", "num_sync_all_streams",
            )}
            return zero(*a, **kw)

        optimizer.zero_grad = zero_grad
        for kind, child in getattr(optimizer, "optimizers", {"optimizer": optimizer}).items():
            child.step = measured(kind, child.step)
        return optimizer

    trainer.build_optimizer = build_optimizer
    original_model = trainer.build_model

    def build_model(*a, **kw):
        model = original_model(*a, **kw)
        model.forward = measured("model_forward", model.forward)
        return model

    trainer.build_model = build_model
    engine.allatom_loss_surface = measured("loss_surface", engine.allatom_loss_surface)
    engine.weighted_rigid_align = measured("alignment", engine.weighted_rigid_align)
    engine.soft_lddt_all_atom_loss = measured("lddt", engine.soft_lddt_all_atom_loss)
    trainer.allatom_forward_and_loss = measured("forward_loss", trainer.allatom_forward_and_loss)
    torch.Tensor.backward = measured("backward", torch.Tensor.backward)
    torch.nn.utils.clip_grad_norm_ = measured("clip", torch.nn.utils.clip_grad_norm_)
    torch.distributed.all_reduce = measured("control_reduce", torch.distributed.all_reduce)
    original_to = ProteinBatch.to

    @wraps(original_to)
    def batch_to(batch, *a, **kw):
        if current is not None and batch.device.type == "cpu":
            current["valid_residues"] += int(batch.res_mask.sum())
            current["padded_residues"] += batch.res_mask.numel()
            current["lengths"].append(batch.max_len)
            digest = hashlib.sha256(batch.res_type.numpy().tobytes())
            digest.update(batch.res_seq_nums.numpy().tobytes())
            current["data_sha256"].append(digest.hexdigest())
        return measured("transfer", original_to)(batch, *a, **kw)

    ProteinBatch.to = batch_to
    original_update = EMA.update

    @wraps(original_update)
    def update(ema, model):
        result = measured("ema", original_update)(ema, model)
        current["updated"] = True
        return result

    EMA.update = update
    original_log = trainer.log_metrics

    def log_metrics(step, total, averages, *a, **kw):
        if current is not None:
            current["loader_wait_ms"] = averages["perf_data_wait_ms_max"]
        return original_log(step, total, averages, *a, **kw)

    trainer.log_metrics = log_metrics
    trainer.save_checkpoint = lambda *a, **kw: None
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    sys.argv = [str(args.train_script), *train_args, "--no_wandb"]
    completed = False
    try:
        trainer.main()
        completed = True
    finally:
        finish()
        stop_monitor.set()
        if args.diagnose_stalls:
            gc.callbacks.remove(gc_event)
        report = dict(rank=rank, world_size=int(os.environ.get("WORLD_SIZE", 1)),
                      backend=backend, device=torch.cuda.get_device_name(local_rank),
                      warmup_updates=args.warmup_updates, profile_update=args.profile_update,
                      source_sha256=original_digest,
                      source_unchanged=source_digest(args.source_root) == original_digest,
                      completed=completed, steps=rows,
                      loader_startup_seconds=loader_startup,
                      loader_profile=args.loader_profile,
                      diagnose_stalls=args.diagnose_stalls,
                      prewarmed_lengths=prewarmed_lengths,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
        (args.report_dir / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
