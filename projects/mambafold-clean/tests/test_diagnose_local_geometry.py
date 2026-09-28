from __future__ import annotations

import math

import numpy as np

from benchmarks.diagnose_local_geometry import angular_distance, dihedral, signed_volume


def test_named_atom_chirality_changes_sign_on_inversion() -> None:
    atoms = {
        "N": np.array([1.0, 0.0, 0.0]),
        "CA": np.zeros(3),
        "C": np.array([0.0, 1.0, 0.0]),
        "CB": np.array([0.0, 0.0, 1.0]),
    }
    names = ("N", "CA", "C", "CB")
    assert signed_volume(atoms, names) == 1.0
    atoms["CB"] = -atoms["CB"]
    assert signed_volume(atoms, names) == -1.0


def test_dihedral_and_wrapped_state_distance() -> None:
    a = np.array([0.0, 1.0, 0.0])
    b = np.zeros(3)
    c = np.array([1.0, 0.0, 0.0])
    angle = math.radians(-60)
    d = np.array([1.0, math.cos(angle), math.sin(angle)])
    assert math.isclose(dihedral(a, b, c, d), -60.0, abs_tol=1e-12)
    assert angular_distance(-179.0, 180.0) == 1.0
