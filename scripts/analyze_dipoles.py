#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cage_anion_mc.bias import formal_coarse_dipole
from cage_anion_mc.fragments import detect_fragments, FragmentInfo, anion_head_center, fragment_center
from cage_anion_mc.polarmace_backend import build_polarmace_calculator, set_polar_metadata, get_density_outputs
from cage_anion_mc.utils import load_json, load_yaml, center_of_geometry, ensure_dir


def fragment_dipole_decomp(atoms, frag_info, origin, charges, atomic_dipoles=None, center_mode="head"):
    q = np.asarray(charges, dtype=float)
    atom_dip = np.zeros((len(atoms), 3)) if atomic_dipoles is None else np.asarray(atomic_dipoles, dtype=float)
    cage = np.asarray(frag_info.cage_indices, dtype=int)
    cage_mu = (q[cage, None] * (atoms.positions[cage] - origin)).sum(axis=0) + atom_dip[cage].sum(axis=0)
    an_pos = np.zeros(3)
    an_int = np.zeros(3)
    q_frags = []
    for k, frag in enumerate(frag_info.anion_fragments):
        idx = np.asarray(frag, dtype=int)
        if center_mode == "head":
            fc = anion_head_center(atoms, frag_info, k)
        else:
            fc = fragment_center(atoms, frag)
        qfrag = float(q[idx].sum())
        q_frags.append(qfrag)
        an_pos += qfrag * (fc - origin)
        an_int += (q[idx, None] * (atoms.positions[idx] - fc)).sum(axis=0) + atom_dip[idx].sum(axis=0)
    total_from_parts = cage_mu + an_pos + an_int
    return cage_mu, an_pos, an_int, total_from_parts, q_frags


def main():
    ap = argparse.ArgumentParser(description="Analyze PolarMACE dipoles and cage/anion coarse decomposition.")
    ap.add_argument("--xyz", required=True, help="Trajectory XYZ/EXTXYZ or single XYZ")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--center-mode", default="head", choices=["head", "geom"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = ensure_dir(args.out or str(Path(cfg.get("output_dir", "runs/fixedN_mc")) / "analysis"))
    frames = read(args.xyz, index=f"::{args.stride}")
    if not isinstance(frames, list):
        frames = [frames]
    frag_json = cfg.get("fragments_json") or str(Path(cfg.get("output_dir", ".")) / "fragments.json")
    if Path(frag_json).exists():
        frag_info = FragmentInfo.from_dict(load_json(frag_json))
    else:
        frag_info = detect_fragments(frames[0], mode=cfg.get("fragment_mode", "auto"))

    calc = build_polarmace_calculator(cfg)
    charge = float(cfg.get("charge", 0.0))
    spin = float(cfg.get("spin", 1.0))
    formal_q_anion = float(cfg.get("formal_q_anion", -1.0))
    rows = []
    for iframe, atoms in enumerate(frames):
        origin = center_of_geometry(atoms, frag_info.cage_indices)
        set_polar_metadata(atoms, charge=charge, spin=spin, external_field=cfg.get("external_field", [0.0, 0.0, 0.0]))
        atoms.calc = calc
        e = float(atoms.get_potential_energy())
        den = get_density_outputs(calc)
        dip = np.asarray(den.get("dipole", [np.nan, np.nan, np.nan]), dtype=float)
        formal = formal_coarse_dipole(atoms, frag_info, origin, q_anion=formal_q_anion, center_mode=args.center_mode)
        row = {
            "frame": iframe,
            "energy_eV": e,
            "dipole_x": dip[0], "dipole_y": dip[1], "dipole_z": dip[2],
            "formal_anion_pos_x": formal[0], "formal_anion_pos_y": formal[1], "formal_anion_pos_z": formal[2],
        }
        if "charges" in den:
            cage_mu, an_pos, an_int, total_parts, q_frags = fragment_dipole_decomp(
                atoms,
                frag_info,
                origin,
                den["charges"],
                den.get("atomic_dipoles"),
                center_mode=args.center_mode,
            )
            for prefix, vec in [
                ("part_cage", cage_mu),
                ("part_anion_position", an_pos),
                ("part_anion_internal", an_int),
                ("part_total", total_parts),
            ]:
                row[f"{prefix}_x"] = vec[0]
                row[f"{prefix}_y"] = vec[1]
                row[f"{prefix}_z"] = vec[2]
            row["mean_fragment_charge"] = float(np.mean(q_frags))
            row["std_fragment_charge"] = float(np.std(q_frags))
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = outdir / "dipole_decomposition.csv"
    df.to_csv(csv_path, index=False)
    summary = {"n_frames": len(df), "columns": list(df.columns)}
    for col in df.columns:
        if col.endswith("_x") or col.endswith("_y") or col.endswith("_z") or col == "energy_eV":
            summary[col] = {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=0))}
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {csv_path}")
    print(f"Wrote {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
