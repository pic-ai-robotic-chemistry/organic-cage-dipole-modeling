from __future__ import annotations

from typing import Any, Dict

from ase import Atoms


def build_polarmace_calculator(cfg: Dict[str, Any]):
    """Build a PolarMACE calculator from config.

    Requires MACE main-branch installation plus graph_electrostatics.
    """
    try:
        from mace.calculators import mace_polar
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import mace.calculators.mace_polar. Install MACE from source "
            "on the main branch and install graph_electrostatics; see INSTALL.md."
        ) from exc

    pm = cfg.get("polarmace", cfg)
    return mace_polar(
        model=pm.get("model", "polar-1-m"),
        device=pm.get("device", "cpu"),
        default_dtype=pm.get("default_dtype", pm.get("dtype", "float64")),
    )


def set_polar_metadata(atoms: Atoms, charge: float = 0.0, spin: float = 1.0, external_field=None) -> None:
    atoms.info["charge"] = float(charge)
    atoms.info["spin"] = float(spin)
    atoms.info["external_field"] = [0.0, 0.0, 0.0] if external_field is None else list(external_field)


def energy_forces_dipole(atoms: Atoms, calc, charge: float, spin: float, external_field=None):
    set_polar_metadata(atoms, charge=charge, spin=spin, external_field=external_field)
    atoms.calc = calc
    e = float(atoms.get_potential_energy())
    f = atoms.get_forces()
    dip = None
    if hasattr(calc, "results"):
        dip = calc.results.get("dipole", None)
    return e, f, dip


def get_density_outputs(calc):
    """Return a dict with optional PolarMACE density outputs."""
    results = getattr(calc, "results", {})
    out = {}
    if "dipole" in results:
        out["dipole"] = results["dipole"]
    if "density_coefficients" in results:
        p = results["density_coefficients"]
        out["density_coefficients"] = p
        out["charges"] = p[:, 0]
        # MACE docs: cartesian atomic dipoles are p[:, [3, 1, 2]].
        if p.shape[1] >= 4:
            out["atomic_dipoles"] = p[:, [3, 1, 2]]
    if "charges" in results and "charges" not in out:
        out["charges"] = results["charges"]
    return out
