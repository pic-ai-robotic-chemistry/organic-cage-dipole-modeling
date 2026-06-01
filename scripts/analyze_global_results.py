#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def latest_stage_dir(seed_dir: Path) -> Path:
    stages = sorted(seed_dir.glob("stage_*_*"))
    if not stages:
        raise FileNotFoundError(f"No stage directories found in {seed_dir}")
    return stages[-1]


def main():
    ap = argparse.ArgumentParser(description="Analyze final best structures from a global search run and collect dipoles.")
    ap.add_argument("--global-dir", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--analyze", choices=["best", "trajectory"], default="best")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = Path(args.global_dir)
    rows = []
    script = Path(__file__).resolve().parent / "analyze_dipoles.py"
    for seed_dir in sorted(root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        stage_dir = latest_stage_dir(seed_dir)
        cfg = stage_dir / "stage_config.yaml"
        xyz = seed_dir / "final_best.xyz" if args.analyze == "best" else stage_dir / "trajectory.xyz"
        out = seed_dir / ("analysis_final_best" if args.analyze == "best" else "analysis_final_trajectory")
        summary_path = out / "summary.json"
        if not summary_path.exists() or not args.skip_existing:
            cmd = [sys.executable, str(script), "--xyz", str(xyz), "--config", str(cfg), "--out", str(out), "--stride", str(args.stride)]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        row = {"seed_dir": seed_dir.name, "analysis": args.analyze, "n_frames": s.get("n_frames", 0)}
        # Flatten mean/std for common quantities.
        for key, val in s.items():
            if isinstance(val, dict) and "mean" in val:
                row[f"{key}_mean"] = val.get("mean")
                row[f"{key}_std"] = val.get("std")
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = root / f"global_dipole_summary_{args.analyze}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    if len(df):
        cols = [c for c in ["seed_dir", "energy_eV_mean", "dipole_x_mean", "dipole_y_mean", "dipole_z_mean", "formal_anion_pos_z_mean"] if c in df.columns]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
