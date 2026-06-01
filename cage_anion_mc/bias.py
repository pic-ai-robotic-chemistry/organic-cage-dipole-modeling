from __future__ import annotations

from typing import Any, Dict

import numpy as np
from ase import Atoms

from .constraints import shell_harmonic_restraint
from .fragments import FragmentInfo, anion_head_center, fragment_center


def formal_coarse_dipole(
    atoms: Atoms,
    frag_info: FragmentInfo,
    origin: np.ndarray,
    q_anion: float = -1.0,
    center_mode: str = "head",
) -> np.ndarray:
    """Formal anion position term, sum_i q_i r_i, about a chosen origin.

    This mirrors the coarse model used in the discussion. For OTf, `head` uses
    the SO3 center; for Br it is identical to COM.
    """
    mu = np.zeros(3)
    for k, frag in enumerate(frag_info.anion_fragments):
        c = anion_head_center(atoms, frag_info, k) if center_mode == "head" else fragment_center(atoms, frag)
        mu += q_anion * (c - origin)
    return mu


def shell_restraint_energy(atoms: Atoms, frag_info: FragmentInfo, origin: np.ndarray, cfg: Dict[str, Any]) -> float:
    shell = cfg.get("shell", {}) or cfg.get("mc", {}).get("shell", {})
    if not shell or not shell.get("enabled", False):
        return 0.0
    return shell_harmonic_restraint(
        atoms,
        frag_info,
        origin,
        min_radius=shell.get("min_radius"),
        max_radius=shell.get("max_radius"),
        k_eva2=float(shell.get("k_eva2", 0.0)),
        center_mode=shell.get("center_mode", "head"),
    )


def coarse_overlap_penalty(
    atoms: Atoms,
    frag_info: FragmentInfo,
    cfg: Dict[str, Any],
) -> float:
    """Optional soft overlap penalty. Default zero to avoid double-counting.

    Use as a stabilizing restraint only; PolarMACE already supplies the real
    short-range physics in the final Metropolis energy.
    """
    params = cfg.get("soft_overlap", {}) or {}
    k = float(params.get("k_eva2", 0.0))
    if k <= 0:
        return 0.0
    r0_heavy = float(params.get("r0_heavy", 1.2))
    r0_h = float(params.get("r0_h", 0.75))
    symbols = atoms.get_chemical_symbols()
    pos = atoms.positions
    mobile = [i for frag in frag_info.anion_fragments for i in frag]
    e = 0.0
    for a in mobile:
        for b in range(len(atoms)):
            if a == b:
                continue
            if any(a in frag and b in frag for frag in frag_info.anion_fragments):
                continue
            d = np.linalg.norm(pos[a] - pos[b])
            r0 = r0_h if (symbols[a] == "H" or symbols[b] == "H") else r0_heavy
            if d < r0:
                e += 0.5 * k * (d - r0) ** 2
    return float(e)


def bias_energy(atoms: Atoms, frag_info: FragmentInfo, origin: np.ndarray, cfg: Dict[str, Any]) -> float:
    if cfg.get("bias", {}).get("enabled", False) is False:
        # Keep only explicit shell restraint if enabled.
        return shell_restraint_energy(atoms, frag_info, origin, cfg.get("bias", {}))
    bcfg = cfg.get("bias", {})
    return shell_restraint_energy(atoms, frag_info, origin, bcfg) + coarse_overlap_penalty(atoms, frag_info, bcfg)
