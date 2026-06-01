# MC model and relation to the coarse dipole formula

## Ensemble

For your stated use case, the anion number is fixed: N = 8, 12, etc. This is a fixed-N adsorption-shell problem, not a chemical-potential GCMC problem.

The sampled state is

```text
X = {R_i, Omega_i}_{i=1..N}
```

where `R_i` is anion position and `Omega_i` is its rigid-body orientation. Br- has only position; OTf- has position and orientation.

## Final Metropolis energy

The scripts use

```text
E_total(X) = E_PolarMACE(X) + E_restraint(X)
```

`E_restraint` is optional and should be weak. It is meant to keep the sampling in the adsorption shell, not to replace physics.

## Coarse dipole formula

The analysis compares PolarMACE dipoles with

```text
mu_total ~= mu_cage + sum_i q_i r_i + sum_i mu_anion_internal
```

For Br-:

```text
r_i = Br position relative to cage center
mu_anion_internal = 0
```

For OTf-:

```text
r_i = SO3 head center relative to cage center
mu_anion_internal is associated with SO3 -> CF3 orientation and internal charge anisotropy
```

## Charged systems

For net charged systems, e.g. +12 cage with 8 monovalent anions = +4 total charge, the absolute dipole depends on origin. The scripts consistently use the cage center as the analysis origin for coarse decomposition.

## Practical strategy

1. Start from an existing 12-anion structure or build a random shell.
2. Use short fixed-N MC to tune move sizes and shell bounds.
3. Run several independent MC trajectories at high temperature.
4. Quench/refine best structures by MC at lower temperature or by local optimization.
5. Analyze `sum q_i r_i`, OTf orientation fields, and PolarMACE total dipoles.
