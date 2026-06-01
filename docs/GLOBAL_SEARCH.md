# Global Fixed-N Cage-Anion Search

The global-search workflow samples a fixed number of anions around a fixed cage.

For each seed, the code:

1. reads a cage structure and one anion template from the YAML config
2. generates a random anion shell
3. runs staged Metropolis MC using PolarMACE energies
4. stores the best structure found for that seed
5. ranks all seeds by final-stage best energy

## Latest Example Configs

The current repository snapshot keeps only the latest self-contained example configs:

```text
configs/latest_new_conformer_20260511/
```

Available configs:

```text
cc3_3f_br12_global.yaml
cc3_3f_otf12_global.yaml
cc3_3f_so4_6_global.yaml
cc19_3f_br12_global.yaml
cc19_3f_otf12_global.yaml
cc19_3f_so4_6_global.yaml
```

Run one example:

```bash
python scripts/run_global_search.py \
  --config configs/latest_new_conformer_20260511/cc3_3f_otf12_global.yaml
```

## Stage Structure

The latest configs use three stages:

- coarse search: higher temperature, larger moves, and hop moves
- annealing: intermediate temperature and smaller moves
- refinement: lower temperature and small local moves

Br systems use translation, radial/tangential moves, hopping, and breathing. OTf and SO4 systems also use rigid-body rotations.

## Output Layout

A run writes one directory per system:

```text
global_runs_sjr_conformers_20260511_clean/<system>/
├── seed_1001/
│   ├── start_random.xyz
│   ├── stage_01_.../
│   ├── stage_02_.../
│   ├── stage_03_.../
│   ├── final_best.xyz
│   └── seed_summary.json
├── seed_1002/
│   └── ...
├── global_stage_results.csv
└── global_final_ranking.csv
```

`final_best.xyz` is the historical lowest-energy structure found by MC for that seed.

## Analyze Final-Best Structures

```bash
python scripts/analyze_global_results.py \
  --global-dir global_runs_sjr_conformers_20260511_clean/cc3_3f_otf12_global \
  --analyze best
```

This runs `scripts/analyze_dipoles.py` on each seed's `final_best.xyz`.

## Interpreting Dipole Decomposition

`scripts/analyze_dipoles.py` reports:

- PolarMACE total dipole
- cage contribution
- anion-position contribution
- anion-internal contribution
- closure of the MACE-derived decomposition

The decomposition is derived from PolarMACE atom-centered charges and local atomic dipoles. It should be interpreted as a fragment grouping of the PolarMACE total dipole.

