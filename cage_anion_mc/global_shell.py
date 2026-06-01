from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from ase import Atoms
from ase.io import write

from .fragments import (
    detect_fragments,
    fragment_center,
    anion_head_center,
    anion_tail_center,
)
from .utils import (
    center_of_geometry,
    random_unit_vector,
    rotation_matrix_align_vectors,
    rotation_matrix_from_axis_angle,
)

OrientationMode = Literal["tail_out", "tail_in", "random"]
DirectionMode = Literal["fibonacci", "random"]


@dataclass
class ShellBuildReport:
    n_anions: int
    attempts: int
    restart: int
    origin: list[float]
    radii: list[float]
    min_atom_distance: float
    min_head_distance: float | None


def pair_min_distance(a: Atoms, b: Atoms) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    diff = a.positions[:, None, :] - b.positions[None, :, :]
    return float(np.linalg.norm(diff, axis=-1).min())


def min_distance_to_positions(frag: Atoms, symbols: list[str], positions: list[np.ndarray]) -> float:
    if not positions:
        return float("inf")
    placed = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))
    return pair_min_distance(frag, placed)


def fibonacci_directions(n: int, rng: np.random.Generator, jitter: float = 0.15) -> list[np.ndarray]:
    """Approximately uniform directions on a sphere with optional small random jitter."""
    dirs: list[np.ndarray] = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    offset = rng.random() * 2.0 * np.pi
    for k in range(n):
        z = 1.0 - 2.0 * (k + 0.5) / n
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = k * golden + offset
        v = np.array([r * np.cos(phi), r * np.sin(phi), z], dtype=float)
        if jitter > 0.0:
            v = v + jitter * rng.normal(size=3)
            v = v / (np.linalg.norm(v) + 1e-15)
        dirs.append(v)
    rng.shuffle(dirs)
    return dirs


def random_directions(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [random_unit_vector(rng) for _ in range(n)]


def template_head_tail_vector(template: Atoms, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return head_center, tail_center, head->tail vector for a template.

    For Br, head == tail == geometric center and the vector is z.
    """
    try:
        info = detect_fragments(template, mode=mode)
        if info.anion_type.lower() != "br" and len(info.anion_fragments) >= 1:
            head = anion_head_center(template, info, 0)
            tail = anion_tail_center(template, info, 0)
            vec = tail - head
            if np.linalg.norm(vec) > 1e-8:
                return head, tail, vec
        center = fragment_center(template, range(len(template)))
        return center, center, np.array([0.0, 0.0, 1.0])
    except Exception:
        center = fragment_center(template, range(len(template)))
        return center, center, np.array([0.0, 0.0, 1.0])


def orient_template(
    template: Atoms,
    template_head: np.ndarray,
    template_vec: np.ndarray,
    normal: np.ndarray,
    rng: np.random.Generator,
    orientation_mode: OrientationMode = "tail_out",
    roll: bool = True,
) -> Atoms:
    frag = template.copy()
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-15)

    if len(frag) > 1:
        if orientation_mode == "tail_out":
            target_vec = normal
        elif orientation_mode == "tail_in":
            target_vec = -normal
        elif orientation_mode == "random":
            target_vec = random_unit_vector(rng)
        else:
            raise ValueError(f"Unknown orientation_mode: {orientation_mode}")
        R = rotation_matrix_align_vectors(template_vec, target_vec)
        frag.positions[:] = template_head + (frag.positions - template_head) @ R.T
        if roll:
            roll_R = rotation_matrix_from_axis_angle(target_vec, rng.uniform(0.0, 2.0 * np.pi))
            frag.positions[:] = template_head + (frag.positions - template_head) @ roll_R.T
    return frag


def build_random_shell(
    cage: Atoms,
    template: Atoms,
    n_anions: int,
    mode: str = "auto",
    radius_min: float = 5.0,
    radius_max: float = 11.0,
    min_atom_distance: float = 1.20,
    min_head_distance: float | None = None,
    seed: int = 1234,
    direction_mode: DirectionMode = "fibonacci",
    orientation_mode: OrientationMode = "tail_out",
    direction_jitter: float = 0.20,
    attempts_per_anion: int = 2000,
    max_restarts: int = 50,
) -> tuple[Atoms, ShellBuildReport]:
    """Build cage + N randomly placed anions.

    For OTf, the template SO3 head is placed at a shell radius and the CF3 tail is
    oriented outward by default. For Br, the Br atom is placed at the shell radius.
    """
    if radius_min <= 0 or radius_max <= radius_min:
        raise ValueError("radius_min/radius_max must be positive and radius_max > radius_min")
    rng_master = np.random.default_rng(seed)
    origin = center_of_geometry(cage)
    t_head, _t_tail, t_vec = template_head_tail_vector(template, mode)

    for restart in range(max_restarts):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        symbols = list(cage.get_chemical_symbols())
        positions = [p.copy() for p in cage.positions]
        placed_heads: list[np.ndarray] = []
        radii: list[float] = []
        attempts = 0

        base_dirs = fibonacci_directions(n_anions, rng, jitter=direction_jitter) if direction_mode == "fibonacci" else random_directions(n_anions, rng)
        ok = True
        for i in range(n_anions):
            placed_this = False
            # Use a quasi-uniform direction first; if it fails, keep trying fresh random directions.
            candidate_dirs = [base_dirs[i]]
            candidate_dirs.extend(random_directions(attempts_per_anion, rng))
            for normal in candidate_dirs:
                attempts += 1
                r = float(rng.uniform(radius_min, radius_max))
                target_head = origin + r * normal
                frag = orient_template(
                    template,
                    t_head,
                    t_vec,
                    normal,
                    rng,
                    orientation_mode=orientation_mode,
                    roll=True,
                )
                frag.positions[:] += target_head - t_head
                if min_head_distance is not None and placed_heads:
                    dheads = [np.linalg.norm(target_head - h) for h in placed_heads]
                    if min(dheads) < min_head_distance:
                        continue
                if min_distance_to_positions(frag, symbols, positions) < min_atom_distance:
                    continue
                symbols.extend(frag.get_chemical_symbols())
                positions.extend([p.copy() for p in frag.positions])
                placed_heads.append(target_head.copy())
                radii.append(r)
                placed_this = True
                break
            if not placed_this:
                ok = False
                break
        if ok:
            atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))
            report = ShellBuildReport(
                n_anions=n_anions,
                attempts=attempts,
                restart=restart,
                origin=[float(x) for x in origin],
                radii=[float(x) for x in radii],
                min_atom_distance=float(min_atom_distance),
                min_head_distance=None if min_head_distance is None else float(min_head_distance),
            )
            return atoms, report

    raise RuntimeError(
        f"Failed to place {n_anions} anions after {max_restarts} restarts. "
        f"Try larger radius range, smaller min distances, or direction_mode=random."
    )


def write_random_shell(path: str | Path, *args, **kwargs) -> ShellBuildReport:
    atoms, report = build_random_shell(*args, **kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write(str(path), atoms, format="extxyz")
    return report
