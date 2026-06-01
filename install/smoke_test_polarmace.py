#!/usr/bin/env python
from __future__ import annotations

import argparse
from ase.build import molecule

from mace.calculators import mace_polar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64")
    args = ap.parse_args()

    atoms = molecule("H2O")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.info["external_field"] = [0.0, 0.0, 0.0]
    atoms.calc = mace_polar(model="polar-1-m", device=args.device, default_dtype=args.dtype)
    e = atoms.get_potential_energy()
    print("PolarMACE smoke-test energy_eV:", e)
    print("Dipole:", atoms.calc.results.get("dipole", None))
    print("density_coefficients shape:", atoms.calc.results.get("density_coefficients", []).shape)


if __name__ == "__main__":
    main()
