from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from ase import Atoms


@dataclass
class FragmentInfo:
    mode: str
    cage_indices: List[int]
    anion_fragments: List[List[int]]
    anion_type: str
    head_indices: List[List[int]]
    tail_indices: List[List[int]]
    notes: str = "indices are zero-based"

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "FragmentInfo":
        return FragmentInfo(**d)


def _symbols(atoms: Atoms) -> List[str]:
    return list(atoms.get_chemical_symbols())


def _distance_matrix(atoms: Atoms) -> np.ndarray:
    pos = atoms.positions
    diff = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def detect_br_fragments(atoms: Atoms) -> FragmentInfo:
    symbols = _symbols(atoms)
    anions = [[i] for i, s in enumerate(symbols) if s == "Br"]
    if not anions:
        raise ValueError("No Br atoms found for Br fragment detection.")
    anion_set = {i for frag in anions for i in frag}
    cage = [i for i in range(len(atoms)) if i not in anion_set]
    return FragmentInfo(
        mode="br",
        cage_indices=cage,
        anion_fragments=anions,
        anion_type="Br",
        head_indices=[frag[:] for frag in anions],
        tail_indices=[[] for _ in anions],
    )


def detect_otf_fragments(
    atoms: Atoms,
    so_cutoff: float = 1.95,
    sc_cutoff: float = 2.25,
    cf_cutoff: float = 1.65,
) -> FragmentInfo:
    """Detect CF3SO3-/OTf- fragments from geometry.

    For each S atom, find nearest 3 O atoms, nearest bonded C atom, and nearest
    3 F atoms around that carbon. This works for the cage examples where OTf
    atoms are not necessarily contiguous in the XYZ file.
    """
    symbols = _symbols(atoms)
    dist = _distance_matrix(atoms)
    s_idx = [i for i, s in enumerate(symbols) if s == "S"]
    if not s_idx:
        raise ValueError("No sulfur atoms found for OTf detection.")

    used_o: set[int] = set()
    used_c: set[int] = set()
    used_f: set[int] = set()
    fragments: List[List[int]] = []
    heads: List[List[int]] = []
    tails: List[List[int]] = []

    for si in s_idx:
        o_candidates = [i for i, s in enumerate(symbols) if s == "O" and dist[si, i] < so_cutoff]
        if len(o_candidates) < 3:
            # Fall back to nearest three oxygens; cage O atoms are usually farther.
            o_candidates = sorted([i for i, s in enumerate(symbols) if s == "O"], key=lambda j: dist[si, j])[:3]
        else:
            o_candidates = sorted(o_candidates, key=lambda j: dist[si, j])[:3]

        c_candidates = [i for i, s in enumerate(symbols) if s == "C" and dist[si, i] < sc_cutoff]
        if len(c_candidates) < 1:
            c_candidates = sorted([i for i, s in enumerate(symbols) if s == "C"], key=lambda j: dist[si, j])[:1]
        ci = sorted(c_candidates, key=lambda j: dist[si, j])[0]

        f_candidates = [i for i, s in enumerate(symbols) if s == "F" and dist[ci, i] < cf_cutoff]
        if len(f_candidates) < 3:
            f_candidates = sorted([i for i, s in enumerate(symbols) if s == "F"], key=lambda j: dist[ci, j])[:3]
        else:
            f_candidates = sorted(f_candidates, key=lambda j: dist[ci, j])[:3]

        frag = [si] + o_candidates + [ci] + f_candidates
        if len(set(frag)) != 8:
            raise ValueError(f"OTf fragment around S index {si} is not 8 unique atoms: {frag}")
        fragments.append(frag)
        heads.append([si] + o_candidates)       # SO3 head / charge-rich group.
        tails.append([ci] + f_candidates)       # CF3 tail.
        used_o.update(o_candidates)
        used_c.add(ci)
        used_f.update(f_candidates)

    anion_set = {i for frag in fragments for i in frag}
    cage = [i for i in range(len(atoms)) if i not in anion_set]
    return FragmentInfo(
        mode="otf",
        cage_indices=cage,
        anion_fragments=fragments,
        anion_type="OTf",
        head_indices=heads,
        tail_indices=tails,
    )


def detect_so4_fragments(atoms: Atoms, so_cutoff: float = 1.90) -> FragmentInfo:
    """Detect SO4(2-) fragments from S-O connectivity.

    A sulfate is treated as a rigid multi-atom anion. It has no tail group, so
    the whole SO4 fragment is used as both the anion fragment and the head
    center for shell placement/constraints.
    """
    symbols = _symbols(atoms)
    dist = _distance_matrix(atoms)
    s_idx = [i for i, s in enumerate(symbols) if s == "S"]
    if not s_idx:
        raise ValueError("No sulfur atoms found for SO4 detection.")

    used_o: set[int] = set()
    fragments: List[List[int]] = []
    heads: List[List[int]] = []
    tails: List[List[int]] = []

    for si in s_idx:
        o_candidates = [
            i
            for i, s in enumerate(symbols)
            if s == "O" and i not in used_o and dist[si, i] < so_cutoff
        ]
        if len(o_candidates) < 4:
            o_candidates = [
                i
                for i, s in sorted(
                    [(i, s) for i, s in enumerate(symbols) if s == "O" and i not in used_o],
                    key=lambda item: dist[si, item[0]],
                )
            ][:4]
        else:
            o_candidates = sorted(o_candidates, key=lambda j: dist[si, j])[:4]

        frag = [si] + o_candidates
        if len(set(frag)) != 5:
            raise ValueError(f"SO4 fragment around S index {si} is not 5 unique atoms: {frag}")
        fragments.append(frag)
        heads.append(frag[:])
        tails.append([])
        used_o.update(o_candidates)

    anion_set = {i for frag in fragments for i in frag}
    cage = [i for i in range(len(atoms)) if i not in anion_set]
    return FragmentInfo(
        mode="so4",
        cage_indices=cage,
        anion_fragments=fragments,
        anion_type="SO4",
        head_indices=heads,
        tail_indices=tails,
    )


def detect_fragments(atoms: Atoms, mode: str = "auto") -> FragmentInfo:
    mode = mode.lower()
    symbols = _symbols(atoms)
    if mode == "auto":
        if any(s == "Br" for s in symbols):
            return detect_br_fragments(atoms)
        if any(s == "S" for s in symbols) and any(s == "F" for s in symbols):
            return detect_otf_fragments(atoms)
        if any(s == "S" for s in symbols) and any(s == "O" for s in symbols):
            return detect_so4_fragments(atoms)
        raise ValueError("Could not auto-detect anions. Use mode='br', mode='otf', mode='so4', or provide fragments.json.")
    if mode == "br":
        return detect_br_fragments(atoms)
    if mode in {"otf", "triflate", "cf3so3"}:
        return detect_otf_fragments(atoms)
    if mode in {"so4", "sulfate", "sulfate2"}:
        return detect_so4_fragments(atoms)
    raise ValueError(f"Unknown fragment detection mode: {mode}")


def fragment_center(atoms: Atoms, indices: Sequence[int]) -> np.ndarray:
    return np.mean(atoms.positions[np.asarray(indices, dtype=int)], axis=0)


def anion_head_center(atoms: Atoms, frag_info: FragmentInfo, k: int) -> np.ndarray:
    head = frag_info.head_indices[k]
    if head:
        return fragment_center(atoms, head)
    return fragment_center(atoms, frag_info.anion_fragments[k])


def anion_tail_center(atoms: Atoms, frag_info: FragmentInfo, k: int) -> np.ndarray:
    tail = frag_info.tail_indices[k]
    if tail:
        return fragment_center(atoms, tail)
    return fragment_center(atoms, frag_info.anion_fragments[k])


def anion_orientation_vector(atoms: Atoms, frag_info: FragmentInfo, k: int) -> np.ndarray:
    """SO3 head -> CF3 tail for OTf. For Br, returns zero vector."""
    if frag_info.anion_type.lower() == "br":
        return np.zeros(3)
    u = anion_tail_center(atoms, frag_info, k) - anion_head_center(atoms, frag_info, k)
    n = np.linalg.norm(u)
    return u / (n + 1e-15)


def subset_atoms(atoms: Atoms, indices: Sequence[int]) -> Atoms:
    return atoms[list(indices)]
