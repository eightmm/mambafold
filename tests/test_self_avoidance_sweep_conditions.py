"""Configuration-contract tests for the self-avoidance benchmark runner."""

from dataclasses import asdict

import pytest

from benchmarks.run_self_avoidance_sweep import (
    _conditions,
    _parse_vdw_channel_grid,
)
from benchmarks.summarize_physics_guidance import INTENDED_GUIDANCE_FIELDS


def test_segment_ablation_changes_only_the_declared_weight() -> None:
    conditions = dict(
        _conditions(
            [1.0],
            physics_ablation=True,
            segment_weight=0.5,
            segment_every_n_steps=2,
        )
    )
    control = asdict(conditions["steric_1"])
    treatment = asdict(conditions["steric_1_segment"])

    assert control["steric_segment_every_n_steps"] == 2
    assert treatment["steric_segment_every_n_steps"] == 2
    assert control["steric_segment_weight"] == 0.0
    assert treatment["steric_segment_weight"] == 0.5
    for field in INTENDED_GUIDANCE_FIELDS:
        control.pop(field)
        treatment.pop(field)
    assert treatment == control


def test_independent_vdw_grid_changes_only_scale_and_interval() -> None:
    conditions = dict(
        _conditions(
            [1.0],
            vdw_channel_grid=[(0.03, 8), (0.1, 2)],
        )
    )
    control = asdict(conditions["steric_1"])
    assert control["vdw_scale"] == 0.0
    assert control["vdw_every_n_steps"] == 8
    assert control["vdw_overlap_tolerance_A"] == 1.5
    assert control["vdw_max_step_A"] == 0.01
    assert control["all_atom_clash_weight"] == 0.0

    for name, scale, interval in (
        ("steric_1_vdw_sep_s0p03_e8", 0.03, 8),
        ("steric_1_vdw_sep_s0p10_e2", 0.1, 2),
    ):
        treatment = asdict(conditions[name])
        assert treatment["vdw_scale"] == scale
        assert treatment["vdw_every_n_steps"] == interval
        treatment.pop("vdw_scale")
        treatment.pop("vdw_every_n_steps")
        expected = dict(control)
        expected.pop("vdw_scale")
        expected.pop("vdw_every_n_steps")
        assert treatment == expected


def test_vdw_grid_parser_fails_closed() -> None:
    assert _parse_vdw_channel_grid(["0.03:8", "0.1:2"]) == [
        (0.03, 8),
        (0.1, 2),
    ]
    with pytest.raises(ValueError, match="expected SCALE:INTERVAL"):
        _parse_vdw_channel_grid(["0.1"])
    with pytest.raises(ValueError, match="duplicates"):
        _parse_vdw_channel_grid(["0.1:2", "0.1:2"])
