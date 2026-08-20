"""Focused tests for paired geometry-guidance aggregation."""

import numpy as np

from benchmarks.summarize_geometry_guidance_validity import bootstrap_delta


def test_bootstrap_delta_is_exact_for_constant_paired_effect():
    low, high = bootstrap_delta(np.full(7, -0.25, dtype=np.float64))
    assert low == -0.25
    assert high == -0.25
