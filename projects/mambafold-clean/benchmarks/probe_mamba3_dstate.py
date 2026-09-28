#!/usr/bin/env python
"""Probe which (d_state, mimo_rank) combinations a GPU can actually run.

The Mamba-3 SSD kernels stage their working set in shared memory, and the
backward pass needs the most of it. Past a point the kernel asks for more
dynamic shared memory than the architecture allows per block and the launch
fails outright — a hard architectural limit, not an out-of-memory condition
that a smaller batch would fix.

`mimo_rank` enters the same budget indirectly: the model derives
`chunk_size = base // mimo_rank`, and a longer chunk means more of the sequence
resident in shared memory at once. Rank and state are therefore not independent
knobs, which is why this sweeps the grid rather than one axis.

Every combination is exercised through the block the model actually uses, with a
real backward pass, because forward alone would clear cases that training cannot.

Usage:
    python benchmarks/probe_mamba3_dstate.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mambafold.model.bimamba3 import BiMamba3Block, _default_chunk_size  # noqa: E402


def device_report() -> dict:
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    report = {
        "name": props.name,
        "capability": f"{capability[0]}.{capability[1]}",
        "total_memory_gib": round(props.total_memory / 1024**3, 2),
        "multi_processor_count": props.multi_processor_count,
    }
    # The opt-in limit is what a kernel may request dynamically, and it is the
    # number these launches actually run into.
    for attr in ("shared_memory_per_block", "shared_memory_per_block_optin",
                 "shared_memory_per_multiprocessor"):
        if hasattr(props, attr):
            report[attr] = getattr(props, attr)
    return report


def classify(error: str) -> str:
    lowered = error.lower()
    if "shared memory" in lowered or "out of resource" in lowered:
        return "shared_memory_limit"
    if "chunk_size must be at least" in lowered:
        # mimo_rank drives chunk_size = base // rank; below the kernel minimum
        # the configuration cannot be expressed on this architecture at all.
        return "chunk_below_kernel_minimum"
    if "out of memory" in lowered:
        return "device_out_of_memory"
    if "no kernel image" in lowered or "not supported" in lowered:
        return "unsupported_arch"
    return "other"


def try_case(d_model: int, d_state: int, mimo_rank: int, headdim: int,
             batch: int, length: int, dtype: torch.dtype) -> dict:
    result = {
        "d_state": d_state,
        "mimo_rank": mimo_rank,
        "chunk_size": _default_chunk_size(mimo_rank),
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        # Keep the module in fp32 and reach bf16 through autocast, the way
        # training does. Casting the module wholesale also casts the bias
        # tensors the Mamba-3 kernels require in fp32, which fails for a reason
        # that has nothing to do with the limit being probed.
        block = BiMamba3Block(
            d_model=d_model, d_state=d_state, mimo_rank=mimo_rank,
            expand=2, headdim=headdim,
        ).cuda()
        x = torch.randn(batch, length, d_model, device="cuda", dtype=torch.float32,
                        requires_grad=True)
        mask = torch.ones(batch, length, device="cuda", dtype=torch.bool)

        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            out = block(x, mask)
        result["forward"] = "ok"
        out.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        result["backward"] = "ok"
        result["status"] = "ok"
        result["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        result.setdefault("forward", "fail")
        result["status"] = "fail"
        result["failure"] = classify(message)
        result["error"] = message.replace("\n", " ")[:400]
    finally:
        for name in ("block", "x", "out"):
            if name in dir():
                pass
        torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--headdim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--d_states", default="64,128,256")
    ap.add_argument("--mimo_ranks", default="1,2,4,8")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device visible", file=sys.stderr)
        return 1

    dtype = getattr(torch, args.dtype)
    device = device_report()
    print(json.dumps(device, indent=2), flush=True)

    d_states = [int(v) for v in args.d_states.split(",")]
    ranks = [int(v) for v in args.mimo_ranks.split(",")]

    cases = []
    for d_state in d_states:
        for rank in ranks:
            case = try_case(
                args.d_model, d_state, rank, args.headdim,
                args.batch, args.length, dtype,
            )
            flag = "ok  " if case["status"] == "ok" else "FAIL"
            extra = case.get("failure", case.get("peak_alloc_gib", ""))
            print(
                f"  {flag} d_state={d_state:<4} mimo_rank={rank:<2} "
                f"chunk={case['chunk_size']:<3} {extra}",
                flush=True,
            )
            if case["status"] == "fail":
                print(f"       {case['error'][:200]}", flush=True)
            cases.append(case)

    payload = {
        "device": device,
        "settings": {
            "d_model": args.d_model, "headdim": args.headdim,
            "batch": args.batch, "length": args.length, "dtype": args.dtype,
        },
        "cases": cases,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)
