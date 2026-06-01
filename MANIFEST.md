# Manifest

## Core Package

- `cage_anion_mc/global_shell.py`: random shell initialization.
- `cage_anion_mc/fragments.py`: fragment detection for cage, Br, OTf, and SO4 systems.
- `cage_anion_mc/rigid.py`: rigid-body transformation utilities.
- `cage_anion_mc/constraints.py`: geometry and shell constraints.
- `cage_anion_mc/mc.py`: Metropolis Monte Carlo engine.
- `cage_anion_mc/polarmace_backend.py`: PolarMACE calculator wrapper.
- `cage_anion_mc/bias.py`: optional bias and coarse helper functions.
- `cage_anion_mc/utils.py`: shared I/O and geometry helpers.

## Command-Line Scripts

- `scripts/run_global_search.py`: multi-seed, multi-stage global MC.
- `scripts/run_fixedN_mc.py`: fixed-N MC driver.
- `scripts/analyze_dipoles.py`: total dipole and fragment decomposition analysis.
- `scripts/analyze_global_results.py`: batch analysis for global-search outputs.
- `scripts/prepare_fragments.py`: extract cage and anion templates.
- `scripts/make_initial_shell.py`: generate random initial shells.
- `scripts/polarmace_energy.py`: single-structure PolarMACE energy helper.
- `scripts/debug_constraints.py`: constraint diagnostics.

## Example Inputs

- `example_inputs/CC3_3F_cage.xyz`
- `example_inputs/CC19_3F_cage.xyz`
- `example_inputs/br_template.xyz`
- `example_inputs/otf_template.xyz`
- `example_inputs/sjr_so4_template.xyz`

## Example Configs

- `configs/latest_new_conformer_20260511/cc3_3f_br12_global.yaml`
- `configs/latest_new_conformer_20260511/cc3_3f_otf12_global.yaml`
- `configs/latest_new_conformer_20260511/cc3_3f_so4_6_global.yaml`
- `configs/latest_new_conformer_20260511/cc19_3f_br12_global.yaml`
- `configs/latest_new_conformer_20260511/cc19_3f_otf12_global.yaml`
- `configs/latest_new_conformer_20260511/cc19_3f_so4_6_global.yaml`

All six latest configs use the PolarMACE/MACE-Polar model identifier `polar-1-m`.

## Intentionally Excluded

- production trajectories
- MC logs
- result CSVs and figures
- PPT files
- manuscript text
- generated structure packages
- local absolute paths
- remote SSH details
- model cache files
