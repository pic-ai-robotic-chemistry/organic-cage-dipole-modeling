#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ase.io import read

from cage_anion_mc.fragments import detect_fragments, FragmentInfo
from cage_anion_mc.mc import FixedNMC
from cage_anion_mc.polarmace_backend import build_polarmace_calculator
from cage_anion_mc.utils import load_json, load_yaml, dump_json, ensure_dir


def main():
    ap = argparse.ArgumentParser(description="Run fixed-N cage-anion rigid-body MC with PolarMACE energy.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    input_xyz = cfg["input_xyz"]
    atoms = read(input_xyz)

    frag_json = cfg.get("fragments_json")
    if frag_json:
        frag_info = FragmentInfo.from_dict(load_json(frag_json))
    else:
        frag_info = detect_fragments(atoms, mode=cfg.get("fragment_mode", "auto"))

    out = ensure_dir(cfg.get("output_dir", "runs/fixedN_mc"))
    dump_json(frag_info.to_dict(), out / "fragments.json")

    calc = build_polarmace_calculator(cfg)
    driver = FixedNMC(atoms, frag_info, calc, cfg)
    driver.run()
    print(f"Done. Acceptance={driver.stats.acceptance:.3f}")
    print(f"Log: {driver.log_path}")
    print(f"Trajectory: {driver.traj_path}")
    print(f"Best: {driver.best_path}")


if __name__ == "__main__":
    main()
