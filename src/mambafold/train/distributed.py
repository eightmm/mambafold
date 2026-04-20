"""DDP utilities: process group setup, all_reduce, GPU monitor."""

import os
import subprocess
import threading
from datetime import timedelta

import torch
import torch.distributed as dist


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


def all_reduce_mean(tensor: torch.Tensor) -> float:
    """모든 rank의 텐서를 sum한 뒤 world_size로 나눈 평균 반환."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


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
