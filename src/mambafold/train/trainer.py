"""MambaFold EqM trainer with wandb logging, EMA, and checkpoint support."""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import wandb

from mambafold.data.collate import ProteinCollator
from mambafold.data.dataset import AFDBDataset
from mambafold.model.mambafold import MambaFoldEqM
from mambafold.train.ema import EMA
from mambafold.train.engine import eval_step, train_step


def build_model(cfg: dict) -> MambaFoldEqM:
    return MambaFoldEqM(
        d_atom=cfg.get("d_atom", 256),
        d_res=cfg.get("d_res", 768),
        d_plm=cfg.get("d_plm", 1024),
        n_atom_enc=cfg.get("n_atom_enc", 4),
        n_trunk=cfg.get("n_trunk", 24),
        n_atom_dec=cfg.get("n_atom_dec", 4),
        use_plm=cfg.get("use_plm", False),
        atom_d_state=cfg.get("atom_d_state", 32),
        atom_mimo_rank=cfg.get("atom_mimo_rank", 2),
        atom_headdim=cfg.get("atom_headdim", 64),
        d_state=cfg.get("d_state", 64),
        mimo_rank=cfg.get("mimo_rank", 4),
        headdim=cfg.get("headdim", 64),
    )


def cosine_warmup_lr(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: Path, model, ema, optimizer, scheduler, step: int, cfg: dict):
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": cfg,
    }, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path: Path, model, ema, optimizer, scheduler, device) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    step = ckpt["step"]
    print(f"Resumed from step {step}: {path}")
    return step


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=cfg.get("wandb_project", "mambafold"),
        name=cfg.get("run_name", None),
        config=cfg,
        resume="allow",
    )

    # Model + EMA
    model = build_model(cfg).to(device)
    ema = EMA(model, decay=cfg.get("ema_decay", 0.999))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.2f}M parameters | device: {device}")

    # Optimizer + LR scheduler
    total_steps = cfg.get("total_steps", 500_000)
    warmup_steps = cfg.get("warmup_steps", 2000)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-2),
        betas=(0.9, 0.999),
    )
    scheduler = cosine_warmup_lr(optimizer, warmup_steps, total_steps)

    # Resume from checkpoint
    start_step = 0
    resume_path = cfg.get("resume", None)
    if resume_path and Path(resume_path).exists():
        start_step = load_checkpoint(
            Path(resume_path), model, ema, optimizer, scheduler, device
        )

    # Dataset + DataLoader
    dataset = AFDBDataset(
        data_dir=cfg["data_dir"],
        max_length=cfg.get("max_length", 256),
    )
    collator = ProteinCollator(
        augment=cfg.get("augment", True),
        copies_per_protein=cfg.get("copies_per_protein", 2),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.get("num_workers", 4) > 0,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} proteins | batch_size={cfg.get('batch_size', 4)}")

    log_interval = cfg.get("log_interval", 50)
    eval_interval = cfg.get("eval_interval", 500)
    ckpt_interval = cfg.get("ckpt_interval", 2000)
    finetune_start = cfg.get("finetune_start", total_steps // 2)

    step = start_step
    stage = "pretrain" if step < finetune_start else "finetune"

    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break

            if step >= finetune_start and stage == "pretrain":
                stage = "finetune"
                print(f"Step {step}: switching to finetune stage (alpha ramp enabled)")

            batch = batch.to(device)
            alpha_mode = "ramp" if stage == "finetune" else "const"

            metrics = train_step(
                model, batch, optimizer,
                grad_clip=cfg.get("grad_clip", 1.0),
                alpha_mode=alpha_mode,
                use_amp=cfg.get("use_amp", True),
            )
            scheduler.step()
            ema.update(model)
            step += 1

            if step % log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/loss": metrics["loss"],
                    "train/eqm": metrics["eqm"],
                    "train/lddt": metrics["lddt"],
                    "train/alpha": metrics["alpha"],
                    "train/lr": lr,
                    "stage": 0 if stage == "pretrain" else 1,
                }, step=step)
                print(
                    f"[{step:>7d}/{total_steps}] "
                    f"loss={metrics['loss']:.4f} "
                    f"eqm={metrics['eqm']:.4f} "
                    f"lddt={metrics['lddt']:.4f} "
                    f"lr={lr:.2e} "
                    f"({stage})"
                )

            if step % eval_interval == 0:
                eval_metrics = eval_step(
                    model, batch, use_amp=cfg.get("use_amp", True)
                )
                wandb.log({
                    "eval/eqm": eval_metrics["eqm"],
                    "eval/lddt": eval_metrics["lddt"],
                    "eval/grad_rms": eval_metrics["grad_rms"],
                }, step=step)

            if step % ckpt_interval == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{step:07d}.pt",
                    model, ema, optimizer, scheduler, step, cfg,
                )

    save_checkpoint(
        ckpt_dir / "final.pt", model, ema, optimizer, scheduler, step, cfg
    )
    wandb.finish()
    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="MambaFold EqM Training")

    # Data
    parser.add_argument("--data_dir", required=True, help="Directory with .pt protein files")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--copies_per_protein", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=True)

    # Model
    parser.add_argument("--d_atom", type=int, default=256)
    parser.add_argument("--d_res", type=int, default=768)
    parser.add_argument("--d_plm", type=int, default=1024)
    parser.add_argument("--n_atom_enc", type=int, default=4)
    parser.add_argument("--n_trunk", type=int, default=24)
    parser.add_argument("--n_atom_dec", type=int, default=4)
    parser.add_argument("--use_plm", action="store_true")
    parser.add_argument("--atom_d_state", type=int, default=32)
    parser.add_argument("--atom_mimo_rank", type=int, default=2)
    parser.add_argument("--atom_headdim", type=int, default=64)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--mimo_rank", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=64)

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--finetune_start", type=int, default=250_000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--use_amp", action="store_true", default=True)

    # Logging / checkpointing
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_project", default="mambafold")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--ckpt_interval", type=int, default=2000)

    args = parser.parse_args()
    train(vars(args))


if __name__ == "__main__":
    main()
