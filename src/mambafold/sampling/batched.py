"""Guidance-free batched ODE/SDE inference.

This module keeps all static sequence/chemistry/PLM tensors on one device for
the full trajectory and replaces only coordinates, time, and self-conditioning
at each solver step.  Randomness is owned by one persistent ``torch.Generator``
per batch row, so a sample's random stream does not depend on batch order or on
the padded length of neighboring proteins.

The public result container uses padded CPU tensors:

* ``final_ca``: ``float32 [B, L, 3]`` in Angstrom
* ``final_aa``: ``float32 [B, L, A, 3]`` in Angstrom
* ``confidence``: ``float32 [B, L]`` in the model's native [0, 1] scale
* ``trunk_latent``: optional ``float32 [B, L, D]``
* ``residue_mask``: ``bool [B, L]`` identifying unpadded residues
* ``schedule``: ``float32 [n_steps + 1]``

Masked atoms/residues are zero in every returned value.  Geometry guidance is
intentionally not supported here; use the legacy single-sample sampler for
guided inference until a batch-safe guidance implementation is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE, PAIR_PAD_ID
from mambafold.data.types import ProteinBatch, ProteinExample
from mambafold.sampling.samplers import _inference_autocast_dtype
from mambafold.utils.geometry import masked_centroid

_T_END = 0.99


@dataclass(frozen=True)
class BatchedSample:
    """Padded, detached CPU outputs from one batched sampling call.

    Coordinates are converted back to Angstrom.  All floating-point fields are
    returned as ``float32`` for a stable public/cache interface, irrespective
    of the model's autocast dtype.  ``trunk_latent`` is ``None`` when
    ``return_trunk_latent=False`` was requested.
    """

    final_ca: Tensor
    final_aa: Tensor
    confidence: Tensor
    trunk_latent: Tensor | None
    residue_mask: Tensor
    schedule: Tensor


def _validate_example(example: ProteinExample, *, atom_slots: int) -> None:
    length = example.seq_len
    if length < 1:
        raise ValueError("inference examples must contain at least one residue")
    residue_fields = (
        "res_type",
        "res_seq_nums",
        "chain_id",
        "entity_id",
        "sym_id",
        "is_nterm",
        "is_cterm",
    )
    for name in residue_fields:
        value = getattr(example, name)
        if value.shape[0] < length:
            raise ValueError(f"{name} is shorter than seq_len={length}")
    atom_fields = ("atom_type", "pair_type", "coords", "atom_mask", "observed_mask")
    for name in atom_fields:
        value = getattr(example, name)
        if value.shape[0] < length or value.shape[1] != atom_slots:
            raise ValueError(
                f"{name} must have residue/atom axes compatible with [{length}, {atom_slots}]"
            )
    if example.coords.shape[-1] != 3:
        raise ValueError("coords must have shape [L, A, 3]")
    if example.esm is not None and example.esm.shape[0] < length:
        raise ValueError(f"ESM embedding is shorter than seq_len={length}")


def prepare_inference_batch(
    examples: list[ProteinExample],
    device: str | torch.device,
    length_bin: int = 0,
) -> ProteinBatch:
    """Pad already centered/scaled examples into one static inference batch.

    Unlike :class:`~mambafold.data.collate.ProteinCollator`, this function does
    not center, scale, rotate, corrupt, or copy examples.  Callers must supply
    coordinates in the model's normalized units.  ``length_bin > 0`` rounds
    the longest sequence up to the next multiple, which bounds compiled model
    shapes for length-bucketed inference.

    All ESM embeddings must either be absent or share one feature dimension and
    dtype.  Their original dtype (notably ESMC ``float16``) is preserved through
    padding and the single host-to-device transfer.
    """
    if not examples:
        raise ValueError("examples must be non-empty")
    if not isinstance(length_bin, int) or length_bin < 0:
        raise ValueError("length_bin must be a non-negative integer")

    atom_slots = examples[0].atom_mask.shape[1]
    for example in examples:
        _validate_example(example, atom_slots=atom_slots)

    max_length = max(example.seq_len for example in examples)
    if length_bin > 0:
        max_length = math.ceil(max_length / length_bin) * length_bin
    batch_size = len(examples)

    coord_dtype = examples[0].coords.dtype
    if not coord_dtype.is_floating_point:
        raise ValueError("coords must use a floating-point dtype")
    if any(example.coords.dtype != coord_dtype for example in examples[1:]):
        raise ValueError("all examples must use the same coordinate dtype")

    esm_presence = [example.esm is not None for example in examples]
    if any(esm_presence) and not all(esm_presence):
        raise ValueError("ESM embeddings must be present for every example or for none")
    esm: Tensor | None = None
    if all(esm_presence):
        first_esm = examples[0].esm
        assert first_esm is not None
        esm_dim = first_esm.shape[-1]
        esm_dtype = first_esm.dtype
        for example in examples[1:]:
            assert example.esm is not None
            if example.esm.shape[-1] != esm_dim:
                raise ValueError("all ESM embeddings must share one feature dimension")
            if example.esm.dtype != esm_dtype:
                raise ValueError("all ESM embeddings must share one dtype")
        esm = torch.zeros(batch_size, max_length, esm_dim, dtype=esm_dtype)

    res_type = torch.zeros(batch_size, max_length, dtype=torch.long)
    res_seq_nums = torch.zeros(batch_size, max_length, dtype=torch.long)
    atom_type = torch.zeros(batch_size, max_length, atom_slots, dtype=torch.long)
    pair_type = torch.full(
        (batch_size, max_length, atom_slots),
        PAIR_PAD_ID,
        dtype=torch.long,
    )
    res_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    atom_mask = torch.zeros(batch_size, max_length, atom_slots, dtype=torch.bool)
    valid_mask = torch.zeros_like(atom_mask)
    ca_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    chain_id = torch.zeros(batch_size, max_length, dtype=torch.long)
    entity_id = torch.zeros(batch_size, max_length, dtype=torch.long)
    sym_id = torch.zeros(batch_size, max_length, dtype=torch.long)
    is_nterm = torch.zeros(batch_size, max_length, dtype=torch.bool)
    is_cterm = torch.zeros(batch_size, max_length, dtype=torch.bool)
    x_clean = torch.zeros(
        batch_size,
        max_length,
        atom_slots,
        3,
        dtype=coord_dtype,
    )

    for row, example in enumerate(examples):
        length = example.seq_len
        observed = example.observed_mask[:length].bool().cpu()
        example_atom_mask = example.atom_mask[:length].bool().cpu()
        res_type[row, :length] = example.res_type[:length].long().cpu()
        res_seq_nums[row, :length] = example.res_seq_nums[:length].long().cpu()
        atom_type[row, :length] = example.atom_type[:length].long().cpu()
        example_pair_type = example.pair_type[:length].long().cpu()
        pair_type[row, :length] = torch.where(
            example_atom_mask,
            example_pair_type,
            torch.full_like(example_pair_type, PAIR_PAD_ID),
        )
        res_mask[row, :length] = True
        atom_mask[row, :length] = example_atom_mask
        valid_mask[row, :length] = example_atom_mask & observed
        ca_mask[row, :length] = example_atom_mask[:, CA_ATOM_ID] & observed[:, CA_ATOM_ID]
        chain_id[row, :length] = example.chain_id[:length].long().cpu()
        entity_id[row, :length] = example.entity_id[:length].long().cpu()
        sym_id[row, :length] = example.sym_id[:length].long().cpu()
        is_nterm[row, :length] = example.is_nterm[:length].bool().cpu()
        is_cterm[row, :length] = example.is_cterm[:length].bool().cpu()
        x_clean[row, :length] = example.coords[:length].to(device="cpu", dtype=coord_dtype)
        if esm is not None:
            assert example.esm is not None
            esm[row, :length] = example.esm[:length].to(device="cpu", dtype=esm.dtype)

    static_batch = ProteinBatch(
        res_type=res_type,
        res_seq_nums=res_seq_nums,
        atom_type=atom_type,
        pair_type=pair_type,
        res_mask=res_mask,
        atom_mask=atom_mask,
        valid_mask=valid_mask,
        ca_mask=ca_mask,
        chain_id=chain_id,
        entity_id=entity_id,
        sym_id=sym_id,
        is_nterm=is_nterm,
        is_cterm=is_cterm,
        x_clean=x_clean,
        x_t=x_clean.clone(),
        eps=torch.zeros_like(x_clean),
        t=torch.zeros(batch_size, 1, 1, 1, dtype=coord_dtype),
        esm=esm,
    )
    return static_batch.to(torch.device(device))


def _center_per_example(coords: Tensor, atom_mask: Tensor) -> Tensor:
    """Center each row over valid atoms and force masked coordinates to zero."""
    mask_f = atom_mask.unsqueeze(-1).to(coords.dtype)
    flat_coords = coords.reshape(coords.shape[0], -1, 3)
    flat_mask = atom_mask.reshape(atom_mask.shape[0], -1)
    center = masked_centroid(flat_coords, flat_mask).unsqueeze(1)
    return (coords - center) * mask_f


def _right_padded_lengths(res_mask: Tensor) -> list[int]:
    lengths = res_mask.sum(dim=1).tolist()
    for row, length in enumerate(lengths):
        expected = torch.arange(res_mask.shape[1], device=res_mask.device) < length
        if not torch.equal(res_mask[row], expected):
            raise ValueError("sample_batch requires right-padded contiguous residue masks")
        if length < 1:
            raise ValueError("every batch row must contain at least one residue")
    return [int(length) for length in lengths]


def _make_generators(device: torch.device, seeds: list[int]) -> list[torch.Generator]:
    generators = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        generators.append(generator)
    return generators


def _rowwise_noise(
    shape: torch.Size,
    lengths: list[int],
    generators: list[torch.Generator],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Draw unpadded rows independently, preserving each seed's RNG stream."""
    batch_size, max_length, atom_slots, xyz = shape
    noise = torch.zeros(batch_size, max_length, atom_slots, xyz, device=device, dtype=dtype)
    for row, (length, generator) in enumerate(zip(lengths, generators, strict=True)):
        noise[row, :length] = torch.randn(
            length,
            atom_slots,
            xyz,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    return noise


def _schedule(
    *,
    n_steps: int,
    sampler: str,
    sde_log_timesteps: bool,
    device: torch.device,
) -> Tensor:
    if sampler == "sde" and sde_log_timesteps:
        schedule = 1.0 - torch.logspace(-2, 0, n_steps + 1, device=device).flip(0)
        schedule = schedule - schedule.min()
        return (schedule / schedule.max()).clamp(min=1e-4, max=1.0)
    if sampler == "sde":
        return torch.linspace(1e-4, 1.0, n_steps + 1, device=device)
    return torch.linspace(0.0, _T_END, n_steps + 1, device=device)


@torch.no_grad()
def sample_batch(
    model,
    static_batch: ProteinBatch,
    seeds: list[int],
    *,
    n_steps: int = 50,
    sampler: str = "ode",
    sde_tau: float = 0.01,
    sde_eps: float = 0.01,
    sde_w_cutoff: float = 0.99,
    sde_log_timesteps: bool = True,
    return_trunk_latent: bool = True,
) -> BatchedSample:
    """Sample one independent guidance-free trajectory per static batch row.

    ``len(seeds)`` must equal ``static_batch.batch_size``.  For multiple seeds
    of one sequence, repeat that :class:`ProteinExample` when constructing the
    static batch.  A generator remains attached to each row for initial and SDE
    noise, making its draws independent of batch ordering and neighboring
    padding.  The function does not mutate ``static_batch`` or global RNG state.
    """
    if sampler not in {"ode", "sde"}:
        raise ValueError(f"unknown sampler: {sampler}")
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if len(seeds) != static_batch.batch_size:
        raise ValueError(
            f"expected one seed per batch row ({static_batch.batch_size}), got {len(seeds)}"
        )
    if sde_tau < 0.0:
        raise ValueError("sde_tau must be non-negative")
    if sde_eps <= 0.0:
        raise ValueError("sde_eps must be positive")

    model.eval()
    device = static_batch.device
    if static_batch.x_t.device != device or static_batch.atom_mask.device != device:
        raise ValueError("all static batch tensors must be on one device")
    lengths = _right_padded_lengths(static_batch.res_mask.bool())
    generators = _make_generators(device, seeds)
    atom_mask = static_batch.atom_mask.bool()
    atom_mask_f = atom_mask.unsqueeze(-1).to(static_batch.x_t.dtype)

    x = _rowwise_noise(
        static_batch.x_t.shape,
        lengths,
        generators,
        device=device,
        dtype=static_batch.x_t.dtype,
    )
    schedule = _schedule(
        n_steps=n_steps,
        sampler=sampler,
        sde_log_timesteps=sde_log_timesteps,
        device=device,
    )
    schedule_cpu = schedule.detach().float().cpu()
    amp_enabled = device.type == "cuda"
    amp_dtype = _inference_autocast_dtype(str(device))
    x_self_cond: Tensor | None = None

    for step in range(n_steps):
        time = float(schedule_cpu[step].clamp(min=1e-4))
        delta_t = float(schedule_cpu[step + 1] - schedule_cpu[step])
        x = _center_per_example(x, atom_mask)
        batch = replace(
            static_batch,
            x_t=x,
            t=static_batch.t.new_full(static_batch.t.shape, time),
            x_self_cond=x_self_cond,
        )
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            output = model(batch, return_aux=False)
            velocity = output["v_atom"]
        velocity = velocity * atom_mask_f
        clean_estimate = (x + (1.0 - time) * velocity).detach() * atom_mask_f
        x_self_cond = clean_estimate

        if sampler == "sde":
            weight = 0.0 if time >= sde_w_cutoff else (1.0 - time) / (time + sde_eps)
            score = ((time * velocity) - x) / max(1.0 - time, 1e-6)
            x = (x + delta_t * (velocity + weight * score)) * atom_mask_f
            noise_scale = math.sqrt(max(2.0 * delta_t * weight * sde_tau, 0.0))
            if noise_scale > 0.0 and step < n_steps - 1:
                noise = _rowwise_noise(
                    x.shape,
                    lengths,
                    generators,
                    device=device,
                    dtype=x.dtype,
                )
                noise = _center_per_example(noise, atom_mask)
                x = (x + noise_scale * noise) * atom_mask_f
        else:
            x = (x + delta_t * velocity) * atom_mask_f
        x = _center_per_example(x, atom_mask)

    final_time = float(schedule_cpu[-1])
    final_batch = replace(
        static_batch,
        x_t=x,
        t=static_batch.t.new_full(static_batch.t.shape, final_time),
        x_self_cond=x_self_cond,
    )
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        final_output = model(final_batch, return_aux=False)
        final_velocity = final_output["v_atom"] * atom_mask_f
        confidence = final_output["conf"]

    if sampler == "sde":
        clean = x * atom_mask_f
    else:
        clean = (x + (1.0 - final_time) * final_velocity) * atom_mask_f
    clean = _center_per_example(clean, atom_mask)

    residue_mask = static_batch.res_mask.bool()
    confidence = confidence * residue_mask.to(confidence.dtype)
    trunk_latent: Tensor | None = None
    if return_trunk_latent:
        if "trunk_latent" not in final_output:
            raise KeyError("model output does not contain required 'trunk_latent'")
        trunk_latent = final_output["trunk_latent"]
        trunk_latent = trunk_latent * residue_mask.unsqueeze(-1).to(trunk_latent.dtype)
        trunk_latent = trunk_latent.detach().float().cpu()

    final_aa = clean.detach().float().cpu() * COORD_SCALE
    # Scaling to Angstrom introduces one final float32 rounding.  Re-center in
    # the public output units so the documented zero-CoM invariant also holds
    # after serialization, rather than only in normalized solver coordinates.
    final_aa = _center_per_example(final_aa, atom_mask.detach().cpu())
    return BatchedSample(
        final_ca=final_aa[:, :, CA_ATOM_ID, :],
        final_aa=final_aa,
        confidence=confidence.detach().float().cpu(),
        trunk_latent=trunk_latent,
        residue_mask=residue_mask.detach().cpu(),
        schedule=schedule_cpu,
    )
