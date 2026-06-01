# PolarMACE Cage-Anion Shell Monte Carlo

This repository contains a fixed-number cage-anion shell Monte Carlo workflow using PolarMACE as the energy and dipole model.

The code is intended for molecular cage salts where a cationic cage is surrounded by a fixed number of anions. The workflow samples anion positions and orientations around a fixed cage, ranks low-energy shell structures, and analyzes total dipoles using a PolarMACE-derived fragment decomposition.

## MLIP Used

The machine-learning interatomic potential used by the example configs is **MACE-Polar / PolarMACE** through the ASE calculator interface.

In the YAML configs this is specified as:

```yaml
polarmace:
  model: polar-1-m
  device: cuda
  default_dtype: float64
```

`polar-1-m` is the model identifier used by the local PolarMACE/MACE-Polar installation. The repository does not include the model weights or cache; users need a working PolarMACE installation before running the examples.

## Scope

Supported systems include:

- monatomic anions such as `Br-`
- rigid molecular anions such as `OTf-` / `CF3SO3-`
- sulfate-like `SO4(2-)` fragments

The workflow is fixed-N: the number of anions is specified in the input configuration and is not sampled.

## Repository Layout

- `cage_anion_mc/`
  Core Python package for fragment detection, shell generation, rigid-body moves, constraints, Metropolis MC, PolarMACE backend, and utility functions.

- `scripts/`
  Command-line tools for preparing fragments, generating initial shells, running MC, running global searches, and analyzing dipole decompositions.

- `configs/latest_new_conformer_20260511/`
  Self-contained example configs for six cage-anion systems built from two cage conformers and three anion types.

- `example_inputs/`
  Small example XYZ inputs used by the latest example configs: cage conformers and anion templates.

- `docs/`
  Method notes for the MC model and global search workflow.

- `install/`
  Environment setup and smoke-test helpers.

## Installation

Create the environment with your preferred conda/mamba workflow. A reference environment file is provided:

```bash
conda env create -f environment.yml
conda activate polarmace-mc
```

PolarMACE model availability depends on your local installation and cache. See `INSTALL.md` and the scripts in `install/` for setup notes.

## Example: Run a Global Search

```bash
python scripts/run_global_search.py \
  --config configs/latest_new_conformer_20260511/cc3_3f_otf12_global.yaml
```

Each global search:

1. reads a fixed cage structure and one anion template
2. generates random initial anion shells for multiple seeds
3. runs staged MC search and annealing
4. writes per-seed trajectories, logs, and final-best structures
5. writes final ranking tables

## Example: Analyze Dipole Decomposition

```bash
python scripts/analyze_dipoles.py \
  --xyz path/to/structure.xyz \
  --config configs/latest_new_conformer_20260511/cc3_3f_otf12_global.yaml \
  --out analysis/cc3_3f_otf12 \
  --center-mode head
```

The analysis reports:

- PolarMACE total energy
- PolarMACE total dipole vector
- cage contribution
- anion-position contribution
- anion-internal contribution
- closure of the fragment decomposition

The decomposition is PolarMACE-derived: atom-centered charges and local atomic dipoles are regrouped by fragment. It is an interpretation of the PolarMACE total dipole, not an independent dipole predictor.

## Example: Analyze All Final-Best Structures

```bash
python scripts/analyze_global_results.py \
  --global-dir global_runs_sjr_conformers_20260511_clean/cc3_3f_otf12_global \
  --analyze best
```

## Latest Example Configs

The latest example configs are in:

```text
configs/latest_new_conformer_20260511/
```

They use:

- `CC3_3F_cage.xyz`
- `CC19_3F_cage.xyz`
- `br_template.xyz`
- `otf_template.xyz`
- `sjr_so4_template.xyz`

from:

```text
example_inputs/
```

These configs are self-contained relative to the repository root.

## Notes

- Cage atoms are fixed during MC.
- Anions are sampled as rigid bodies during MC.
- Br moves include translation, radial/tangential motion, hopping, and shell breathing.
- OTf and SO4 moves also include rigid-body rotation.
- Geometry constraints reject nonphysical structures before PolarMACE energy evaluation.
- Optional bias terms are supported but disabled in the provided latest example configs.
