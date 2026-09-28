#!/usr/bin/env python
"""Smallest possible distributed init, to separate the cluster from our code.

`scripts/train.py` segfaults inside NCCL communicator creation before it reaches
any of this project's own logic. Nothing here imports the model, the dataset or
mamba_ssm: if this crashes, the fault is the environment, and if it passes the
fault is ours. Run under torchrun.
"""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def main() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = os.environ.get("DIST_BACKEND", "nccl")
    eager = os.environ.get("PROBE_DEVICE_ID", "1") == "1"
    tag = f"[rank{local_rank}] backend={backend} device_id={'yes' if eager else 'no'}"

    print(f"{tag} torch={torch.__version__} "
          f"cuda={torch.version.cuda} devices={torch.cuda.device_count()} "
          f"visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(local_rank)
        print(f"{tag} device{local_rank}={props.name} cap={props.major}.{props.minor}", flush=True)
        torch.cuda.set_device(local_rank)

    kwargs = {"timeout": datetime.timedelta(minutes=2)}
    if eager and backend == "nccl" and torch.cuda.is_available():
        kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
    print(f"{tag} init_process_group ...", flush=True)
    dist.init_process_group(backend, **kwargs)
    print(f"{tag} init ok rank={dist.get_rank()}/{dist.get_world_size()}", flush=True)

    print(f"{tag} barrier ...", flush=True)
    dist.barrier()
    print(f"{tag} barrier ok", flush=True)

    dev = f"cuda:{local_rank}" if backend == "nccl" else "cpu"
    t = torch.ones(4, device=dev) * (dist.get_rank() + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(1, dist.get_world_size() + 1))
    ok = bool((t == expected).all())
    print(f"{tag} all_reduce {'ok' if ok else 'WRONG'} got {t[0].item()} want {expected}",
          flush=True)

    dist.destroy_process_group()
    print(f"{tag} DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
