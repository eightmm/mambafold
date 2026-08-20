"""Run MambaFold inference on a single-chain benchmark id list.

Reads a text file of PDB IDs (one per line), samples one prediction per id with
the direct all-atom sampler, and writes paired PDBs:

    <out_dir>/<pdb_id>_pred.pdb   # model prediction (Kabsch-aligned to GT)
    <out_dir>/<pdb_id>_gt.pdb     # ground truth

Multichain examples are skipped by the dataset loader.

Usage:
    PYTHONPATH=src .venv/bin/python benchmarks/run_inference.py \
        --ckpt outputs/train/<phase>/ckpt_latest.pt \
        --ids  benchmarks/sets/t1_quick.txt \
        --out  benchmarks/results/<phase>_t1 \
        [--max_length 1024] [--n_steps 50] [--sampler ode|sde] [--no_ema]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE
from mambafold.data.dataset import RCSBDataset
from mambafold.data.transforms import center_and_scale
from mambafold.data.types import ProteinBatch
from mambafold.sampling import GeometryGuidanceConfig, sample
from mambafold.structure_io import write_mmcif, write_pdb
from mambafold.train.distributed import enable_cuda_perf_flags
from mambafold.train.trainer import load_from_checkpoint
from mambafold.utils.geometry import kabsch_align


def save_pdb(coords_aa, res_type_ids, atom_mask, b_factors, chain_id, out_path):
    """Backward-compatible wrapper used by benchmark helpers."""
    write_pdb(coords_aa, res_type_ids, atom_mask, b_factors, chain_id, out_path)


def save_cif(coords_aa, res_type_ids, atom_mask, b_factors, chain_id, out_path, entry_id):
    """Write the same atom slots as PDBx/mmCIF."""
    write_mmcif(
        coords_aa,
        res_type_ids,
        atom_mask,
        b_factors,
        chain_id,
        out_path,
        entry_id=entry_id,
    )


def prepare_static_batch(ex, device):
    """Move target features to ``device`` once for the full sampling trajectory."""
    L = ex.seq_len
    coords = ex.coords.unsqueeze(0)
    batch = ProteinBatch(
        res_type=ex.res_type.unsqueeze(0),
        res_seq_nums=ex.res_seq_nums.unsqueeze(0),
        atom_type=ex.atom_type.unsqueeze(0),
        pair_type=ex.pair_type.unsqueeze(0),
        res_mask=torch.ones(1, L, dtype=torch.bool),
        atom_mask=ex.atom_mask.unsqueeze(0),
        valid_mask=(ex.atom_mask & ex.observed_mask).unsqueeze(0),
        ca_mask=(ex.atom_mask[:, CA_ATOM_ID] & ex.observed_mask[:, CA_ATOM_ID]).unsqueeze(0),
        chain_id=ex.chain_id.unsqueeze(0),
        entity_id=ex.entity_id.unsqueeze(0),
        sym_id=ex.sym_id.unsqueeze(0),
        is_nterm=ex.is_nterm.unsqueeze(0),
        is_cterm=ex.is_cterm.unsqueeze(0),
        x_clean=coords,
        x_t=coords,
        eps=torch.zeros_like(coords),
        t=torch.zeros(1, 1, 1, 1),
        esm=ex.esm.unsqueeze(0) if ex.esm is not None else None,
    )
    return batch.to(torch.device(device))


def make_sampling_batch(static_batch, x, t_cur):
    """Reuse static GPU features while replacing only coordinates and time."""
    return replace(
        static_batch,
        x_t=x.unsqueeze(0),
        t=static_batch.t.new_full(static_batch.t.shape, float(t_cur)),
    )


def kabsch_align_to_gt(pred_aa, true_aa, mask_la):
    flat_p = pred_aa.reshape(-1, 3)
    flat_t = true_aa.reshape(-1, 3)
    flat_m = mask_la.reshape(-1)
    if int(flat_m.sum()) < 3:
        return pred_aa
    _, R, _ = kabsch_align(flat_p[flat_m], flat_t[flat_m])
    cp = flat_p[flat_m].mean(0)
    ct = flat_t[flat_m].mean(0)
    aligned = ((flat_p - cp) @ R.T) + ct
    return aligned.reshape(pred_aa.shape)


def main():
    enable_cuda_perf_flags()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--ckpt_provenance",
        default=None,
        help="Stable checkpoint path recorded in the manifest when --ckpt is a "
        "node-local staged copy",
    )
    p.add_argument("--ids", required=True, help="text file of PDB IDs, one per line")
    p.add_argument("--out", required=True)
    p.add_argument("--data_dir", default="data/rcsb")
    p.add_argument(
        "--esm_dir",
        default=None,
        help="ESM cache directory. For PLM-conditioned checkpoints this must be "
        "provided here or recorded explicitly in the checkpoint config.",
    )
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--n_steps", type=int, default=50, help="Sampler integration steps")
    p.add_argument(
        "--sampler",
        choices=["ode", "sde"],
        default="ode",
        help="ode = Euler flow path; sde = SimpleFold-style Euler-Maruyama",
    )
    p.add_argument(
        "--sde_tau",
        type=float,
        default=0.01,
        help="SimpleFold SDE stochasticity scale when --sampler=sde",
    )
    p.add_argument(
        "--sde_eps",
        type=float,
        default=0.01,
        help="SimpleFold SDE diffusion eps in w(t)=(1-t)/(t+eps)",
    )
    p.add_argument(
        "--sde_w_cutoff",
        type=float,
        default=0.99,
        help="Set diffusion coefficient to zero for t >= cutoff",
    )
    p.add_argument(
        "--sde_log_timesteps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SimpleFold log timesteps for SDE sampling",
    )
    p.add_argument(
        "--geometry_guidance_scale",
        type=float,
        default=0.0,
        help="GT-free geometry guidance strength; 0 keeps legacy sampling exactly",
    )
    p.add_argument(
        "--geometry_guidance_preset",
        choices=("bond_cleanup", "stereochemical", "self_avoidance"),
        default="bond_cleanup",
        help="bond_cleanup preserves the original three-term guidance; "
        "stereochemical enables topology, planarity, chirality, Ramachandran, "
        "and atom-radius clash barriers; self_avoidance separates an earlier "
        "residue-coherent nonlocal steric channel from late local cleanup",
    )
    p.add_argument(
        "--geometry_guidance_start",
        type=float,
        default=None,
        help="Apply local geometry guidance only after this flow time "
        "(default: 0.5, or 0.65 for self_avoidance)",
    )
    p.add_argument(
        "--geometry_guidance_every",
        type=int,
        default=None,
        help="Evaluate local geometry guidance every N solver steps "
        "(default: 1, or 2 for self_avoidance)",
    )
    p.add_argument(
        "--steric_guidance_scale",
        type=float,
        default=0.0,
        help="Independent nonlocal C-alpha self-avoidance strength",
    )
    p.add_argument(
        "--steric_guidance_start",
        type=float,
        default=0.35,
        help="Flow time at which the self-avoidance channel starts",
    )
    p.add_argument(
        "--steric_guidance_ramp_end",
        type=float,
        default=0.55,
        help="Flow time at which self-avoidance reaches full strength",
    )
    p.add_argument(
        "--steric_guidance_every",
        type=int,
        default=1,
        help="Evaluate self-avoidance every N solver steps",
    )
    p.add_argument(
        "--steric_ca_min_dist_A",
        type=float,
        default=3.6,
        help="Nonlocal C-alpha excluded-volume floor in Angstrom",
    )
    p.add_argument(
        "--steric_ca_seq_sep",
        type=int,
        default=12,
        help="Ignore same-chain C-alpha pairs at or below this sequence separation",
    )
    p.add_argument(
        "--steric_smoothing_radius",
        type=int,
        default=4,
        help="Gaussian residue-force smoothing radius (0 disables smoothing)",
    )
    p.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="Number of independent samples per target. Each is written as "
        "`<pid>_pred_seed<i>.pdb`; the seed-0 file is also linked at "
        "`<pid>_pred.pdb` so existing scoring code keeps working.",
    )
    p.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Sampling seeds = [seed_offset, seed_offset+1, ...]",
    )
    p.add_argument(
        "--output_format",
        choices=("pdb", "cif", "both"),
        default="pdb",
        help="Structure output format (default preserves the legacy PDB-only runner)",
    )
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    args = p.parse_args()

    local_start = args.geometry_guidance_start
    if local_start is None:
        local_start = 0.65 if args.geometry_guidance_preset == "self_avoidance" else 0.5
    local_every = args.geometry_guidance_every
    if local_every is None:
        local_every = 2 if args.geometry_guidance_preset == "self_avoidance" else 1

    if args.geometry_guidance_preset == "self_avoidance":
        geometry_guidance = GeometryGuidanceConfig.self_avoidance(
            local_scale=args.geometry_guidance_scale,
            steric_scale=args.steric_guidance_scale,
            local_start=local_start,
            local_every_n_steps=local_every,
            steric_start=args.steric_guidance_start,
            steric_ramp_end=args.steric_guidance_ramp_end,
            steric_every_n_steps=args.steric_guidance_every,
            steric_smoothing_radius=args.steric_smoothing_radius,
        )
        geometry_guidance = replace(
            geometry_guidance,
            steric_ca_min_dist_A=args.steric_ca_min_dist_A,
            steric_ca_seq_sep=args.steric_ca_seq_sep,
        )
    elif args.geometry_guidance_preset == "stereochemical":
        if args.steric_guidance_scale != 0.0:
            p.error("--steric_guidance_scale requires --geometry_guidance_preset self_avoidance")
        geometry_guidance = GeometryGuidanceConfig.stereochemical(
            scale=args.geometry_guidance_scale,
            start=local_start,
            every_n_steps=local_every,
        )
    else:
        if args.steric_guidance_scale != 0.0:
            p.error("--steric_guidance_scale requires --geometry_guidance_preset self_avoidance")
        geometry_guidance = GeometryGuidanceConfig(
            scale=args.geometry_guidance_scale,
            start=local_start,
            every_n_steps=local_every,
        )
    try:
        geometry_guidance.validate()
    except ValueError as exc:
        p.error(str(exc))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] ckpt={args.ckpt} ema={args.use_ema} device={device}")
    model = load_from_checkpoint(args.ckpt, device, use_ema=args.use_ema)
    model.eval()

    print(
        f"[infer] direct all-atom sampler={args.sampler} "
        f"(n_steps={args.n_steps}, tau={args.sde_tau}, eps={args.sde_eps}, "
        f"log_timesteps={args.sde_log_timesteps}, "
        f"geometry_guidance={geometry_guidance.scale}, "
        f"steric_guidance={geometry_guidance.steric_scale})"
    )

    # Auto-fill --esm_dir only from an explicit checkpoint value. Never guess a
    # legacy cache path: ESM3 and ESMC caches are not interchangeable.
    if args.esm_dir is None:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ck_args = ck.get("args", {})
        if not isinstance(ck_args, dict):
            ck_args = vars(ck_args)
        if ck_args.get("use_plm", False):
            args.esm_dir = ck_args.get("esm_dir")
            if not args.esm_dir:
                p.error(
                    "checkpoint uses PLM conditioning but does not record esm_dir; "
                    "pass --esm_dir explicitly"
                )
            print(f"[esm] auto-detected from ckpt: esm_dir={args.esm_dir}")
        del ck

    ids = [s.strip() for s in Path(args.ids).read_text().split() if s.strip()]
    print(f"[load] eval ids={len(ids)} from {args.ids}")

    # One dataset that we'll point at individual files.
    ds = RCSBDataset(
        args.data_dir,
        max_length=args.max_length,
        min_length=10,
        min_obs_ratio=0.0,
        esm_dir=args.esm_dir,
        single_chain_only=True,
    )

    summary_rows: list[dict] = []
    t0 = time.time()
    for i, pid in enumerate(ids):
        npz_path = REPO / args.data_dir / pid[1:3] / f"{pid}.npz"
        if not npz_path.exists():
            print(f"[skip] {pid}: missing npz")
            continue
        ds.files = [npz_path]
        try:
            ex = ds[0]
        except Exception:
            print(f"[skip] {pid}: not single-chain or invalid")
            continue
        if ex is None:
            print(f"[skip] {pid}: not single-chain or invalid")
            continue

        ex_c = center_and_scale(ex)
        L = ex.seq_len
        n_chains = int(ex.chain_id.max().item()) + 1 if ex.chain_id is not None else 1

        aa_mask = (ex_c.atom_mask & ex_c.observed_mask).numpy().astype(bool)
        true_aa = ex_c.coords.numpy() * COORD_SCALE
        res_type = ex_c.res_type.numpy()
        atom_mask_np = ex_c.atom_mask.numpy().astype(bool)
        chain_id_np = (
            ex_c.chain_id.numpy() if ex.chain_id is not None else np.zeros(L, dtype=np.int64)
        )

        # B-factor: zero (no per-atom pLDDT calc here — scoring step computes lDDT)
        b_zero = np.zeros_like(aa_mask, dtype=np.float32)

        # GT written once in the requested representation(s).
        if args.output_format in ("pdb", "both"):
            save_pdb(true_aa, res_type, aa_mask, b_zero, chain_id_np, out_dir / f"{pid}_gt.pdb")
        if args.output_format in ("cif", "both"):
            save_cif(
                true_aa,
                res_type,
                aa_mask,
                b_zero,
                chain_id_np,
                out_dir / f"{pid}_gt.cif",
                f"{pid}_gt",
            )

        static_batch = prepare_static_batch(ex_c, device)
        n_ok = 0
        seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))
        for si, sd in enumerate(seeds):
            try:
                with torch.no_grad():
                    _, pred_aa, _, _, conf = sample(
                        model,
                        ex,
                        lambda x, ti: make_sampling_batch(static_batch, x, ti),
                        n_steps=args.n_steps,
                        seed=sd,
                        device=device,
                        sampler=args.sampler,
                        sde_tau=args.sde_tau,
                        sde_eps=args.sde_eps,
                        sde_w_cutoff=args.sde_w_cutoff,
                        sde_log_timesteps=args.sde_log_timesteps,
                        record_trajectory=False,
                        geometry_guidance=geometry_guidance,
                    )
            except torch.cuda.OutOfMemoryError:
                print(f"[oom] {pid}: L={L} chains={n_chains} seed={sd}")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"[err] {pid} seed={sd}: {type(e).__name__}: {e}")
                continue

            # Predicted pLDDT (per-residue) → per-atom B-factor column (0-100 scale).
            b_pred = (
                np.asarray(conf, dtype=np.float32)[:, None]
                * 100.0
                * atom_mask_np.astype(np.float32)
            )
            pred_aa_aligned = kabsch_align_to_gt(pred_aa, true_aa, aa_mask)
            if args.output_format in ("pdb", "both"):
                seed_path = out_dir / f"{pid}_pred_seed{si}.pdb"
                save_pdb(pred_aa_aligned, res_type, atom_mask_np, b_pred, chain_id_np, seed_path)
            if args.output_format in ("cif", "both"):
                save_cif(
                    pred_aa_aligned,
                    res_type,
                    atom_mask_np,
                    b_pred,
                    chain_id_np,
                    out_dir / f"{pid}_pred_seed{si}.cif",
                    f"{pid}_pred_seed{si}",
                )
            # First successful seed also written as the canonical "<pid>_pred.pdb"
            if n_ok == 0:
                if args.output_format in ("pdb", "both"):
                    save_pdb(
                        pred_aa_aligned,
                        res_type,
                        atom_mask_np,
                        b_pred,
                        chain_id_np,
                        out_dir / f"{pid}_pred.pdb",
                    )
                if args.output_format in ("cif", "both"):
                    save_cif(
                        pred_aa_aligned,
                        res_type,
                        atom_mask_np,
                        b_pred,
                        chain_id_np,
                        out_dir / f"{pid}_pred.cif",
                        f"{pid}_pred",
                    )
            n_ok += 1

        if n_ok == 0:
            # All seeds failed → drop the GT too so the scorer doesn't see a half-pair
            for suffix in ("pdb", "cif"):
                (out_dir / f"{pid}_gt.{suffix}").unlink(missing_ok=True)
            continue

        summary_rows.append({"pdb_id": pid, "L": int(L), "n_chains": n_chains, "n_seeds_ok": n_ok})
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(ids) - i - 1)
        print(
            f"[{i + 1:>4}/{len(ids)}] {pid}  L={L:>5}  chains={n_chains:>2}  "
            f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s"
        )

    manifest = {
        "ckpt": str(args.ckpt_provenance or args.ckpt),
        "checkpoint_staged": args.ckpt_provenance is not None,
        "ids_file": str(args.ids),
        "n_steps": args.n_steps,
        "sampler": args.sampler,
        "sde_tau": args.sde_tau,
        "sde_eps": args.sde_eps,
        "sde_w_cutoff": args.sde_w_cutoff,
        "sde_log_timesteps": args.sde_log_timesteps,
        "geometry_guidance_preset": args.geometry_guidance_preset,
        "geometry_guidance": asdict(geometry_guidance),
        "output_format": args.output_format,
        "max_length": args.max_length,
        "use_ema": args.use_ema,
        "single_chain_only": True,
        "n_predicted": len(summary_rows),
        "rows": summary_rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] predictions written: {len(summary_rows)} → {out_dir}")


if __name__ == "__main__":
    main()
