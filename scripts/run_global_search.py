#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cage_anion_mc.fragments import detect_fragments, subset_atoms, FragmentInfo
from cage_anion_mc.global_shell import build_random_shell
from cage_anion_mc.mc import FixedNMC
from cage_anion_mc.polarmace_backend import build_polarmace_calculator
from cage_anion_mc.utils import ensure_dir, dump_json, load_yaml


def write_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def prepare_cage_template(cfg: Dict[str, Any], root: Path):
    mode = cfg.get("fragment_mode", "auto")
    prep = ensure_dir(root / "prepared")
    if cfg.get("cage_xyz") and cfg.get("anion_template_xyz"):
        cage = read(cfg["cage_xyz"])
        template = read(cfg["anion_template_xyz"])
        write(prep / "cage.xyz", cage, format="extxyz")
        write(prep / "anion_template.xyz", template, format="extxyz")
        return cage, template

    source_xyz = cfg.get("source_xyz") or cfg.get("input_xyz")
    if not source_xyz:
        raise ValueError("Global config needs source_xyz/input_xyz, or cage_xyz + anion_template_xyz.")
    atoms = read(source_xyz)
    info = detect_fragments(atoms, mode=mode)
    dump_json(info.to_dict(), prep / "source_fragments.json")
    cage = subset_atoms(atoms, info.cage_indices)
    template = subset_atoms(atoms, info.anion_fragments[0])
    write(prep / "cage.xyz", cage, format="extxyz")
    write(prep / "anion_template.xyz", template, format="extxyz")
    print(f"Prepared cage/template from {source_xyz}: cage_atoms={len(cage)}, template_atoms={len(template)}, n_source_anions={len(info.anion_fragments)}")
    return cage, template


def base_stage_config(global_cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["fragment_mode", "charge", "spin", "external_field", "polarmace", "bias"]
    out = {k: copy.deepcopy(global_cfg[k]) for k in keys if k in global_cfg}
    return out


def merge_stage(base: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    # stage may put MC params under mc or top-level convenience keys
    for k, v in stage.items():
        if k in {"name"}:
            continue
        cfg[k] = copy.deepcopy(v)
    if "mc" not in cfg:
        cfg["mc"] = {}
    return cfg


def run_one_stage(input_xyz: Path, outdir: Path, cfg: Dict[str, Any], calc, seed: int) -> Dict[str, Any]:
    atoms = read(str(input_xyz))
    frag_info = detect_fragments(atoms, mode=cfg.get("fragment_mode", "auto"))
    cfg = copy.deepcopy(cfg)
    cfg["input_xyz"] = str(input_xyz)
    cfg["output_dir"] = str(outdir)
    cfg["seed"] = int(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    write_yaml(cfg, outdir / "stage_config.yaml")
    dump_json(frag_info.to_dict(), outdir / "fragments.json")
    driver = FixedNMC(atoms, frag_info, calc, cfg)
    driver.run()
    return {
        "stage_dir": str(outdir),
        "best_xyz": str(outdir / "best.xyz"),
        "trajectory_xyz": str(outdir / "trajectory.xyz"),
        "mc_log": str(outdir / "mc_log.csv"),
        "best_energy_eV": float(driver.best_energy),
        "acceptance": float(driver.stats.acceptance),
        "attempted": int(driver.stats.attempted),
        "accepted": int(driver.stats.accepted),
        "rejected_constraint": int(driver.stats.rejected_constraint),
        "rejected_energy": int(driver.stats.rejected_energy),
        "rejected_exception": int(driver.stats.rejected_exception),
    }


def main():
    ap = argparse.ArgumentParser(description="Global fixed-N cage-anion search: random shells + staged PolarMACE MC annealing.")
    ap.add_argument("--config", required=True, help="Global search YAML config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    name = cfg.get("name", Path(args.config).stem)
    root = ensure_dir(cfg.get("output_dir", f"global_runs/{name}"))
    write_yaml(cfg, root / "global_config.yaml")

    cage, template = prepare_cage_template(cfg, root)
    calc = build_polarmace_calculator(cfg)

    search = cfg.get("global_search", {})
    n_anions = int(search.get("n_anions", cfg.get("n_anions", 12)))
    seeds = search.get("seeds", [1001, 1002, 1003, 1004, 1005])
    start_cfg = search.get("start", {})
    stages = search.get("stages", [])
    if not stages:
        raise ValueError("global_search.stages is empty. Provide at least one annealing/MC stage.")

    rows = []
    for iseed, seed in enumerate(seeds):
        seed_dir = ensure_dir(root / f"seed_{seed}")
        print(f"\n=== Global seed {seed} ({iseed+1}/{len(seeds)}) ===")
        start_xyz = seed_dir / "start_random.xyz"
        atoms_start, report = build_random_shell(
            cage=cage,
            template=template,
            n_anions=n_anions,
            mode=cfg.get("fragment_mode", "auto"),
            radius_min=float(start_cfg.get("radius_min", 5.0)),
            radius_max=float(start_cfg.get("radius_max", 11.0)),
            min_atom_distance=float(start_cfg.get("min_atom_distance", start_cfg.get("min_distance", 1.20))),
            min_head_distance=start_cfg.get("min_head_distance", None),
            seed=int(seed),
            direction_mode=start_cfg.get("direction_mode", "fibonacci"),
            orientation_mode=start_cfg.get("orientation", "tail_out"),
            direction_jitter=float(start_cfg.get("direction_jitter", 0.20)),
            attempts_per_anion=int(start_cfg.get("attempts_per_anion", 2000)),
            max_restarts=int(start_cfg.get("max_restarts", 50)),
        )
        write(str(start_xyz), atoms_start, format="extxyz")
        with open(seed_dir / "start_report.json", "w", encoding="utf-8") as f:
            json.dump(report.__dict__, f, indent=2)

        previous_xyz = start_xyz
        base = base_stage_config(cfg)
        stage_results = []
        for istage, stage in enumerate(stages, start=1):
            sname = stage.get("name", f"stage{istage}")
            outdir = seed_dir / f"stage_{istage:02d}_{sname}"
            stage_cfg = merge_stage(base, stage)
            # Default fragment mode/charge if omitted.
            stage_cfg.setdefault("fragment_mode", cfg.get("fragment_mode", "auto"))
            stage_cfg.setdefault("charge", cfg.get("charge", 0))
            stage_cfg.setdefault("spin", cfg.get("spin", 1))
            stage_cfg.setdefault("external_field", cfg.get("external_field", [0.0, 0.0, 0.0]))
            stage_cfg.setdefault("polarmace", cfg.get("polarmace", {}))
            print(f"--- seed {seed}: {sname}, input={previous_xyz} ---")
            res = run_one_stage(previous_xyz, outdir, stage_cfg, calc, seed=int(seed) + 10000 * istage)
            res.update({"seed": seed, "stage": istage, "stage_name": sname})
            rows.append(res)
            stage_results.append(res)
            previous_xyz = Path(res["best_xyz"])

        final_best = seed_dir / "final_best.xyz"
        shutil.copyfile(previous_xyz, final_best)
        # A compact per-seed result file.
        with open(seed_dir / "seed_summary.json", "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "final_best_xyz": str(final_best), "stages": stage_results}, f, indent=2)

    df = pd.DataFrame(rows)
    df.to_csv(root / "global_stage_results.csv", index=False)
    # Final stage only, sorted by best energy.
    final_stage = max(int(r["stage"]) for r in rows)
    finals = df[df["stage"] == final_stage].copy().sort_values("best_energy_eV")
    finals.to_csv(root / "global_final_ranking.csv", index=False)
    print(f"\nWrote {root / 'global_stage_results.csv'}")
    print(f"Wrote {root / 'global_final_ranking.csv'}")
    if len(finals):
        print("\nBest seeds:")
        print(finals[["seed", "best_energy_eV", "acceptance", "best_xyz"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
