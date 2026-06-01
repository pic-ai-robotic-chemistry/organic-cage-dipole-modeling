#!/usr/bin/env python
from __future__ import annotations

import argparse
import yaml
import numpy as np
from ase.io import read

from cage_anion_mc.fragments import detect_fragments, anion_head_center, fragment_center
from cage_anion_mc.constraints import shell_constraint_ok, hard_min_distance_ok
from cage_anion_mc.utils import center_of_geometry


def main():
    ap = argparse.ArgumentParser(description="Check fragment detection and initial MC constraints.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    atoms = read(cfg["input_xyz"])
    frag = detect_fragments(atoms, cfg.get("fragment_mode", "auto"))
    origin = center_of_geometry(atoms, frag.cage_indices)

    print(f"input_xyz: {cfg['input_xyz']}")
    print(f"fragment_mode: {frag.mode}")
    print(f"anion_type: {frag.anion_type}")
    print(f"n_anions: {len(frag.anion_fragments)}")
    print(f"n_cage_atoms: {len(frag.cage_indices)}")
    print(f"cage_center: {origin}")

    mode = cfg.get("mc", {}).get("constraints", {}).get("shell", {}).get("center_mode", "head")
    radii = []
    for k, af in enumerate(frag.anion_fragments):
        c = anion_head_center(atoms, frag, k) if mode == "head" else fragment_center(atoms, af)
        r = float(np.linalg.norm(c - origin))
        radii.append(r)
        print(f"anion {k+1:02d}: shell_radius_{mode} = {r:.3f} Å")

    shell = cfg.get("mc", {}).get("constraints", {}).get("shell", {})
    minr = shell.get("min_radius")
    maxr = shell.get("max_radius")
    print(f"shell min/max from config: {minr}, {maxr}")
    print(f"initial shell radii min/max: {min(radii):.3f}, {max(radii):.3f}")

    shell_ok = shell_constraint_ok(
        atoms, frag, origin,
        min_radius=minr, max_radius=maxr,
        center_mode=mode,
    ) if shell.get("enabled", False) else True

    md = cfg.get("mc", {}).get("constraints", {}).get("min_distance", {})
    dist_ok = hard_min_distance_ok(
        atoms, frag.cage_indices, frag.anion_fragments,
        min_heavy_heavy=float(md.get("heavy_heavy", 1.05)),
        min_h_any=float(md.get("h_any", 0.65)),
    ) if md.get("enabled", True) else True

    print(f"initial shell_constraint_ok: {shell_ok}")
    print(f"initial hard_min_distance_ok: {dist_ok}")
    if not shell_ok:
        print("WARNING: initial structure violates shell constraint. Lower min_radius or increase max_radius.")
    if not dist_ok:
        print("WARNING: initial structure violates hard min-distance constraint.")


if __name__ == "__main__":
    main()
