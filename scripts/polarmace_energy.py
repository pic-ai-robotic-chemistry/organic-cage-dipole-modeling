#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ase.io import read

from cage_anion_mc.polarmace_backend import build_polarmace_calculator, energy_forces_dipole


def main():
    ap = argparse.ArgumentParser(description="Single-point PolarMACE energy/dipole check.")
    ap.add_argument("xyz")
    ap.add_argument("--model", default="polar-1-m")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--charge", type=float, default=0.0)
    ap.add_argument("--spin", type=float, default=1.0)
    args = ap.parse_args()

    atoms = read(args.xyz)
    cfg = {"polarmace": {"model": args.model, "device": args.device, "default_dtype": args.dtype}}
    calc = build_polarmace_calculator(cfg)
    e, f, dip = energy_forces_dipole(atoms, calc, charge=args.charge, spin=args.spin)
    print(f"Energy_eV: {e:.12f}")
    print(f"Max_force_eV_A: {abs(f).max():.6f}")
    if dip is not None:
        print("Dipole:", " ".join(f"{x:.8f}" for x in dip))
    if hasattr(calc, "results") and "density_coefficients" in calc.results:
        print("density_coefficients shape:", calc.results["density_coefficients"].shape)


if __name__ == "__main__":
    main()
