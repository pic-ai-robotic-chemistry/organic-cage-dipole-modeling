#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cage_anion_mc.global_shell import build_random_shell


def main():
    ap = argparse.ArgumentParser(description="Build a random fixed-N cage-anion shell from cage.xyz and anion_template.xyz.")
    ap.add_argument("--cage", required=True)
    ap.add_argument("--anion", required=True)
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("--radius-min", type=float, default=5.0)
    ap.add_argument("--radius-max", type=float, default=11.0)
    ap.add_argument("--min-atom-distance", "--min-distance", dest="min_atom_distance", type=float, default=1.2)
    ap.add_argument("--min-head-distance", type=float, default=None)
    ap.add_argument("--direction-mode", default="fibonacci", choices=["fibonacci", "random"])
    ap.add_argument("--direction-jitter", type=float, default=0.20)
    ap.add_argument("--orientation", default="tail_out", choices=["tail_out", "tail_in", "random"])
    ap.add_argument("--attempts-per-anion", type=int, default=2000)
    ap.add_argument("--max-restarts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="start_shell.xyz")
    ap.add_argument("--mode", default="auto", choices=["auto", "br", "otf"])
    args = ap.parse_args()

    cage = read(args.cage)
    tmpl = read(args.anion)
    atoms, report = build_random_shell(
        cage=cage,
        template=tmpl,
        n_anions=args.N,
        mode=args.mode,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        min_atom_distance=args.min_atom_distance,
        min_head_distance=args.min_head_distance,
        seed=args.seed,
        direction_mode=args.direction_mode,
        orientation_mode=args.orientation,
        direction_jitter=args.direction_jitter,
        attempts_per_anion=args.attempts_per_anion,
        max_restarts=args.max_restarts,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write(args.out, atoms, format="extxyz")
    report_path = Path(args.out).with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, indent=2)
    print(f"Wrote {args.out} with N={args.N}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
