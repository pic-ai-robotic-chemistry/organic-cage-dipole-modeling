from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from ase import Atoms

from .utils import random_unit_vector, rotation_matrix_from_axis_angle


def translate_fragment(atoms: Atoms, indices: Sequence[int], shift: np.ndarray) -> None:
    atoms.positions[np.asarray(indices, dtype=int)] += np.asarray(shift, dtype=float)


def rotate_fragment(atoms: Atoms, indices: Sequence[int], R: np.ndarray, center: np.ndarray | None = None) -> None:
    idx = np.asarray(indices, dtype=int)
    if center is None:
        center = atoms.positions[idx].mean(axis=0)
    rel = atoms.positions[idx] - center
    atoms.positions[idx] = center + rel @ R.T


def random_translate(atoms: Atoms, indices: Sequence[int], sigma: float, rng: np.random.Generator) -> None:
    translate_fragment(atoms, indices, rng.normal(scale=sigma, size=3))


def random_rotate(atoms: Atoms, indices: Sequence[int], max_angle_deg: float, rng: np.random.Generator) -> None:
    axis = random_unit_vector(rng)
    angle = rng.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg))
    R = rotation_matrix_from_axis_angle(axis, angle)
    rotate_fragment(atoms, indices, R)


def radial_move_fragment(
    atoms: Atoms,
    indices: Sequence[int],
    center: np.ndarray,
    radial_sigma: float,
    tangential_sigma: float,
    rng: np.random.Generator,
) -> None:
    idx = np.asarray(indices, dtype=int)
    frag_center = atoms.positions[idx].mean(axis=0)
    r = frag_center - center
    rnorm = np.linalg.norm(r)
    if rnorm < 1e-8:
        n = random_unit_vector(rng)
    else:
        n = r / rnorm
    radial = rng.normal(scale=radial_sigma) * n
    tangent = rng.normal(scale=tangential_sigma, size=3)
    tangent -= np.dot(tangent, n) * n
    translate_fragment(atoms, idx, radial + tangent)
