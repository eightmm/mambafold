"""DDP utilities: process group setup, all_reduce, GPU monitor."""

import os
import re
import subprocess
import threading
from datetime import timedelta

import torch
import torch.distributed as dist


def _leading_int(value: str | None) -> int | None:
    """Parse Slurm counts such as ``32`` or ``32(x2)``."""
    if not value:
        return None
    match = re.match(r"\s*(\d+)", value)
    return int(match.group(1)) if match else None


def resolve_dataloader_workers(
    requested: int,
    world_size: int,
    *,
    reserve_per_rank: int = 1,
) -> tuple[int, int, str]:
    """Cap per-rank DataLoader workers to the CPUs already allocated.

    ``num_workers`` is per rank, while Slurm CPU allocations usually cover the
    whole torchrun task.  Keep one CPU per rank for the training process and
    divide only the remainder among DataLoader workers.  This function never
    requests or changes cluster resources.
    """
    if requested < 0:
        raise ValueError(f"num_workers must be >= 0, got {requested}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    slurm_cpus = _leading_int(os.environ.get("SLURM_CPUS_ON_NODE"))
    if slurm_cpus is not None:
        available = slurm_cpus
        source = "SLURM_CPUS_ON_NODE"
    else:
        try:
            available = len(os.sched_getaffinity(0))
            source = "sched_getaffinity"
        except (AttributeError, OSError):
            available = os.cpu_count() or world_size
            source = "os.cpu_count"

    per_rank = max(1, available // world_size)
    cap = max(0, per_rank - max(0, reserve_per_rank))
    return min(requested, cap), available, source


def setup_dist():
    """torchrun이 설정한 env vars로 process group 초기화.

    Returns:
        (is_dist, rank, world_size, device)
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        return False, 0, 1, "cuda" if torch.cuda.is_available() else "cpu"

    torch.cuda.set_device(local_rank)
    backend = os.environ.get("DIST_BACKEND", "nccl")
    dist.init_process_group(backend, timeout=timedelta(minutes=30))
    return True, dist.get_rank(), dist.get_world_size(), f"cuda:{local_rank}"


def _redirect_tmpdir_if_noexec():
    """If /tmp is mounted noexec, TileLang/ninja JIT .so loads fail. Redirect
    TMPDIR to a project-local .cache/tmp before any JIT kernel compiles."""
    import tempfile
    if os.environ.get("TMPDIR"):
        return  # respect user override
    tmp = tempfile.gettempdir()
    probe = os.path.join(tmp, ".mf_exec_probe.sh")
    try:
        with open(probe, "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(probe, 0o755)
        exec_ok = os.access(probe, os.X_OK) and subprocess.call(
            [probe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ) == 0
    except Exception:
        exec_ok = False
    finally:
        try:
            os.unlink(probe)
        except OSError:
            pass
    if exec_ok:
        return
    # Fallback: write next to the repo's working tree.
    fallback = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "tmp")
    )
    os.makedirs(fallback, exist_ok=True)
    os.environ["TMPDIR"] = fallback
    tempfile.tempdir = fallback


def enable_cuda_perf_flags():
    """Enable CUDA/cuDNN fast paths that are safe with bf16 autocast training.

    - TF32 for leftover fp32 matmuls (loss/optimizer math).
    - cuDNN benchmark for static-shape conv kernels.
    - bf16 reduced-precision reduction (stays within bf16 dynamic range).
    - Redirect TMPDIR if /tmp is noexec (required for TileLang JIT on B200).
    No-op for CUDA flags if CUDA isn't available; TMPDIR redirect always runs.
    """
    _redirect_tmpdir_if_noexec()
    if not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # bf16 has the same dynamic range as fp32, so reduced-precision reduction
    # in bf16 matmuls is safe and noticeably faster on Hopper/Blackwell.
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


def all_reduce_mean(tensor: torch.Tensor) -> float:
    """모든 rank의 텐서를 sum한 뒤 world_size로 나눈 평균 반환."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


def any_rank_true(value: bool, device: torch.device | str) -> bool:
    """Return whether ``value`` is true on any rank (all ranks must call)."""
    flag = torch.tensor([int(value)], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def distributed_max_int(value: int, device: torch.device | str) -> int:
    """Return the maximum integer value across ranks (all ranks must call)."""
    result = torch.tensor([value], device=device, dtype=torch.int64)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return int(result.item())


class GPUMonitor:
    """백그라운드 스레드에서 주기적으로 GPU 상태를 출력/로깅."""

    def __init__(self, interval: int = 60):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        try:
            import wandb as _wandb
        except ImportError:
            _wandb = None

        while not self._stop.wait(self.interval):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    text=True,
                ).strip()
                for line in out.splitlines():
                    idx, name, util, used, total = [x.strip() for x in line.split(",")]
                    print(f"  [GPU:{idx}] {name} | util={util}% | vram={used}/{total} MiB",
                          flush=True)
                    if _wandb is not None and _wandb.run is not None:
                        _wandb.log({"gpu/util_pct": int(util),
                                    "gpu/vram_used_mib": int(used)})
            except Exception:
                pass
