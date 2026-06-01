from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write

KB_EV_PER_K = 8.617333262145e-5


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_atoms(path: str | Path, index: int | str = 0) -> Atoms:
    return read(str(path), index=index)


def write_atoms(path: str | Path, atoms: Atoms, append: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write(str(path), atoms, format="extxyz", append=append)


def center_of_geometry(atoms: Atoms, indices=None) -> np.ndarray:
    pos = atoms.positions if indices is None else atoms.positions[np.asarray(indices, dtype=int)]
    return np.mean(pos, axis=0)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rotation_matrix_align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return R such that R @ a ~= b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-15)
    b = b / (np.linalg.norm(b) + 1e-15)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180-degree rotation around any perpendicular axis.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return rotation_matrix_from_axis_angle(axis, math.pi)
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
