from __future__ import annotations

from typing import Sequence

import numpy as np
from ase import Atoms

from .fragments import FragmentInfo, anion_head_center, fragment_center


def hard_min_distance_ok(
    atoms: Atoms,
    cage_indices: Sequence[int],
    anion_fragments: Sequence[Sequence[int]],
    min_heavy_heavy: float = 1.05,
    min_h_any: float = 0.65,
) -> bool:
    symbols = atoms.get_chemical_symbols()
    pos = atoms.positions
    all_anion = [i for frag in anion_fragments for i in frag]
    mobile = set(all_anion)
    # Check all distinct pairs involving at least one anion atom.
    n = len(atoms)
    for a in all_anion:
        for b in range(n):
            if a == b:
                continue
            # Avoid double-checking anion-internal bonded atoms within same fragment.
            same_fragment = any(a in frag and b in frag for frag in anion_fragments)
            if same_fragment:
                continue
            d = np.linalg.norm(pos[a] - pos[b])
            threshold = min_h_any if (symbols[a] == "H" or symbols[b] == "H") else min_heavy_heavy
            if d < threshold:
                return False
    return True


def shell_constraint_ok(
    atoms: Atoms,
    frag_info: FragmentInfo,
    center: np.ndarray,
    min_radius: float | None,
    max_radius: float | None,
    center_mode: str = "head",
) -> bool:
    for k, frag in enumerate(frag_info.anion_fragments):
        if center_mode == "head":
            c = anion_head_center(atoms, frag_info, k)
        else:
            c = fragment_center(atoms, frag)
        r = np.linalg.norm(c - center)
        if min_radius is not None and r < min_radius:
            return False
        if max_radius is not None and r > max_radius:
            return False
    return True


def shell_harmonic_restraint(
    atoms: Atoms,
    frag_info: FragmentInfo,
    center: np.ndarray,
    min_radius: float | None,
    max_radius: float | None,
    k_eva2: float = 0.0,
    center_mode: str = "head",
) -> float:
    if k_eva2 <= 0:
        return 0.0
    e = 0.0
    for k, frag in enumerate(frag_info.anion_fragments):
        c = anion_head_center(atoms, frag_info, k) if center_mode == "head" else fragment_center(atoms, frag)
        r = np.linalg.norm(c - center)
        if min_radius is not None and r < min_radius:
            e += 0.5 * k_eva2 * (r - min_radius) ** 2
        if max_radius is not None and r > max_radius:
            e += 0.5 * k_eva2 * (r - max_radius) ** 2
    return float(e)
