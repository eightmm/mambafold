"""Reference-free, bounded stereochemical correction for late sampling steps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from mambafold.data.constants import COORD_SCALE, ID_TO_AA, RESIDUE_ATOM_TO_SLOT
from mambafold.data.types import ProteinBatch
from mambafold.losses.geometry import stereochemical_losses


@dataclass(frozen=True)
class LateGeometryGuidance:
    """Correct a clean-coordinate estimate after selected late SDE steps.

    ``max_step_A=0`` disables guidance exactly. The parameter file is the
    OpenStructure Engh-Huber bond/angle table, not a benchmark reference.
    """

    props_path: Path | None = None
    start_t: float = 0.90
    every_n_steps: int = 10
    max_step_A: float = 0.02
    bond_weight: float = 1.0
    angle_weight: float = 1.0
    clash_weight: float = 1.0
    backbone_scale: float = 0.1

    def validate(self) -> None:
        if not 0.0 <= self.start_t < 1.0:
            raise ValueError("guidance start_t must be in [0, 1)")
        if self.every_n_steps < 1:
            raise ValueError("guidance every_n_steps must be positive")
        if self.max_step_A < 0.0:
            raise ValueError("guidance max_step_A must be non-negative")
        if not 0.0 <= self.backbone_scale <= 1.0:
            raise ValueError("guidance backbone_scale must be in [0, 1]")
        if min(self.bond_weight, self.angle_weight, self.clash_weight) < 0.0:
            raise ValueError("guidance energy weights must be non-negative")
        if self.max_step_A > 0.0:
            if self.props_path is None or not self.props_path.is_file():
                raise ValueError("enabled guidance requires an OpenStructure stereo parameter file")
            if self.bond_weight + self.angle_weight + self.clash_weight == 0.0:
                raise ValueError("enabled guidance requires a nonzero energy weight")


@lru_cache(maxsize=8)
def _reference_tables(path: str) -> dict[str, Tensor]:
    """Load only standard-residue bond and angle means from OpenStructure."""
    sections: dict[str, dict[str, list[tuple[tuple[str, ...], float]]]] = {"bond": {}, "angle": {}}
    section = ""
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if line.startswith("Bond"):
            section = "bond"
        elif line.startswith("Angle"):
            section = "angle"
        elif line.startswith("Non-bonded"):
            break
        elif line == "-":
            section = ""
        elif section and line:
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"invalid stereo parameter line: {raw}")
            names = tuple(fields[0].split("-"))
            expected = 2 if section == "bond" else 3
            if len(names) != expected:
                raise ValueError(f"invalid atom names in stereo parameter line: {raw}")
            sections[section].setdefault(fields[1], []).append((names, float(fields[2])))

    if not all(sections[name] for name in ("bond", "angle")):
        raise ValueError("stereo parameter file lacks bond or angle targets")
    n_types = max(ID_TO_AA) + 1
    result: dict[str, Tensor] = {}
    for section, width in (("bond", 2), ("angle", 3)):
        max_count = max(
            len(sections[section].get(ID_TO_AA.get(type_id, "UNK"), []))
            for type_id in range(n_types)
        )
        slots = torch.zeros(n_types, max_count, width, dtype=torch.long)
        targets = torch.zeros(n_types, max_count)
        present = torch.zeros(n_types, max_count, dtype=torch.bool)
        for type_id, residue in ID_TO_AA.items():
            lookup = RESIDUE_ATOM_TO_SLOT[residue]
            for item, (names, mean) in enumerate(sections[section].get(residue, [])):
                if any(atom not in lookup for atom in names):
                    continue
                slots[type_id, item] = torch.tensor([lookup[atom] for atom in names])
                targets[type_id, item] = mean if section == "bond" else math.cos(math.radians(mean))
                present[type_id, item] = True
        result[f"{section}_slots"] = slots
        result[f"{section}_targets"] = targets
        result[f"{section}_present"] = present
    return result


def _gather(coords: Tensor, slots: Tensor) -> Tensor:
    return torch.gather(coords, 2, slots[..., None].expand(*slots.shape, 3))


def guidance_energy(
    coords: Tensor,
    batch: ProteinBatch,
    config: LateGeometryGuidance,
    tables: dict[str, Tensor],
) -> Tensor:
    """A per-example geometry energy with no reference-coordinate input."""
    coords_A = coords.float() * COORD_SCALE
    atom_mask = batch.atom_mask.bool()
    res_type = batch.res_type.long()
    terms = []
    if config.bond_weight:
        slots = tables["bond_slots"][res_type]
        valid = tables["bond_present"][res_type]
        valid = valid & torch.gather(atom_mask, 2, slots[..., 0])
        valid = valid & torch.gather(atom_mask, 2, slots[..., 1])
        left, right = (_gather(coords_A, slots[..., index]) for index in range(2))
        distance = torch.linalg.vector_norm(left - right, dim=-1)
        target = tables["bond_targets"][res_type]
        loss = F.smooth_l1_loss(distance, target, beta=0.1, reduction="none")
        terms.append(
            config.bond_weight * (loss * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)
        )
    if config.angle_weight:
        slots = tables["angle_slots"][res_type]
        valid = tables["angle_present"][res_type]
        for index in range(3):
            valid = valid & torch.gather(atom_mask, 2, slots[..., index])
        left, center, right = (_gather(coords_A, slots[..., index]) for index in range(3))
        first = F.normalize(left - center, dim=-1, eps=1e-6)
        second = F.normalize(right - center, dim=-1, eps=1e-6)
        cosine = (first * second).sum(dim=-1)
        target = tables["angle_targets"][res_type]
        loss = (cosine - target).square()
        terms.append(
            config.angle_weight
            * (loss * valid).sum(dim=(1, 2))
            / valid.sum(dim=(1, 2)).clamp(min=1)
        )
    if config.clash_weight:
        # Existing training clash term already uses canonical emitted atoms and
        # excludes covalent neighbours. Bond/angle outputs are ignored because
        # their training targets depend on experimental reference coordinates.
        clash = stereochemical_losses(
            coords,
            coords.detach(),
            atom_mask,
            atom_mask,
            batch.res_mask,
            res_type,
            batch.atom_type,
            batch.res_seq_nums,
            batch.chain_id,
        )["clash"]
        terms.append(config.clash_weight * clash)
    return torch.stack(terms).sum(dim=0)


def guide_clean_estimate(
    clean: Tensor,
    batch: ProteinBatch,
    config: LateGeometryGuidance,
    *,
    time: float,
) -> Tensor:
    """Return a bounded coordinate correction in normalized model units."""
    if config.max_step_A == 0.0 or time < config.start_t:
        return torch.zeros_like(clean)
    assert config.props_path is not None
    tables = {
        name: value.to(clean.device)
        for name, value in _reference_tables(str(config.props_path.resolve())).items()
    }
    with torch.enable_grad():
        point = clean.detach().float().requires_grad_(True)
        energy = guidance_energy(point, batch, config, tables)
        gradient = torch.autograd.grad(energy.sum(), point)[0]
    mask = batch.atom_mask.bool()
    gradient = torch.nan_to_num(gradient) * mask[..., None]
    gradient[..., :4, :] *= config.backbone_scale
    # Normalize each example separately so mixed-length batches cannot change
    # guidance strength. One atom moves at most max_step_A per correction.
    rms = (gradient.square().sum(dim=(1, 2, 3)) / (3 * mask.sum(dim=(1, 2)).clamp(min=1))).sqrt()
    direction = gradient / rms[:, None, None, None].clamp(min=1e-8)
    atom_norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    direction = direction / atom_norm.clamp(min=1.0)
    progress = (time - config.start_t) / (1.0 - config.start_t)
    correction = -direction * (config.max_step_A * progress / COORD_SCALE)
    return correction.to(clean.dtype) * mask[..., None]
