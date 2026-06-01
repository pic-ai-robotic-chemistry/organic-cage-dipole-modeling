from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from ase import Atoms
from ase.io import write
from tqdm import trange

from .bias import bias_energy
from .constraints import hard_min_distance_ok, shell_constraint_ok
from .fragments import FragmentInfo, anion_head_center, fragment_center, anion_orientation_vector
from .polarmace_backend import set_polar_metadata
from .rigid import random_rotate, random_translate, radial_move_fragment, translate_fragment, rotate_fragment
from .utils import KB_EV_PER_K, center_of_geometry, ensure_dir, random_unit_vector, rotation_matrix_align_vectors, rotation_matrix_from_axis_angle


@dataclass
class MCStats:
    attempted: int = 0
    accepted: int = 0
    rejected_energy: int = 0
    rejected_constraint: int = 0
    rejected_exception: int = 0

    @property
    def acceptance(self) -> float:
        return self.accepted / max(self.attempted, 1)


class FixedNMC:
    """Fixed-N rigid-body MC for cage-anion adsorption shells.

    The cage is kept fixed. Each anion fragment is moved as a rigid body.
    Final Metropolis decisions use PolarMACE energy plus optional weak restraints.
    """

    def __init__(self, atoms: Atoms, frag_info: FragmentInfo, calc, cfg: Dict[str, Any]):
        self.atoms = atoms
        self.frag_info = frag_info
        self.calc = calc
        self.cfg = cfg
        self.mc_cfg = cfg.get("mc", {})
        self.rng = np.random.default_rng(int(cfg.get("seed", 12345)))
        self.temperature = float(cfg.get("temperature", self.mc_cfg.get("temperature", 300.0)))
        self.beta = 1.0 / (KB_EV_PER_K * self.temperature)
        self.charge = float(cfg.get("charge", 0.0))
        self.spin = float(cfg.get("spin", 1.0))
        self.external_field = cfg.get("external_field", [0.0, 0.0, 0.0])
        self.output_dir = ensure_dir(cfg.get("output_dir", "runs/fixedN_mc"))
        self.log_path = self.output_dir / "mc_log.csv"
        self.traj_path = self.output_dir / "trajectory.xyz"
        self.best_path = self.output_dir / "best.xyz"
        self.stats = MCStats()
        self.origin = center_of_geometry(self.atoms, self.frag_info.cage_indices)
        self.best_energy = None
        self.best_positions = None
        self.debug_raise = bool(self.mc_cfg.get("raise_exceptions", False))

    def _ml_energy(self) -> float:
        set_polar_metadata(self.atoms, charge=self.charge, spin=self.spin, external_field=self.external_field)
        self.atoms.calc = self.calc
        return float(self.atoms.get_potential_energy())

    def total_energy(self) -> Tuple[float, float, float]:
        e_ml = self._ml_energy()
        e_bias = bias_energy(self.atoms, self.frag_info, self.origin, self.cfg)
        return e_ml + e_bias, e_ml, e_bias

    def constraints_ok(self) -> bool:
        c_cfg = self.mc_cfg.get("constraints", {})
        min_dist_cfg = c_cfg.get("min_distance", {})
        if min_dist_cfg.get("enabled", True):
            if not hard_min_distance_ok(
                self.atoms,
                self.frag_info.cage_indices,
                self.frag_info.anion_fragments,
                min_heavy_heavy=float(min_dist_cfg.get("heavy_heavy", 1.05)),
                min_h_any=float(min_dist_cfg.get("h_any", 0.65)),
            ):
                return False
        shell = c_cfg.get("shell", {})
        if shell.get("enabled", False):
            if not shell_constraint_ok(
                self.atoms,
                self.frag_info,
                self.origin,
                min_radius=shell.get("min_radius"),
                max_radius=shell.get("max_radius"),
                center_mode=shell.get("center_mode", "head"),
            ):
                return False
        return True

    def _shell_limits_for_hop(self) -> tuple[float, float]:
        shell = self.mc_cfg.get("constraints", {}).get("shell", {})
        rmin_raw = shell.get("min_radius", 1.0)
        rmax_raw = shell.get("max_radius", 12.0)
        # YAML null becomes None; handle it safely.
        rmin = 1.0 if rmin_raw is None else float(rmin_raw)
        rmax = 12.0 if rmax_raw is None else float(rmax_raw)
        if rmax <= rmin:
            rmax = rmin + 1.0
        return rmin, rmax

    def propose(self) -> Tuple[str, int, np.ndarray]:
        weights = self.mc_cfg.get("move_weights", {})
        names = ["translate", "rotate", "radial", "hop", "breathing"]
        w = np.array([
            float(weights.get("translate", 0.45)),
            float(weights.get("rotate", 0.30)),
            float(weights.get("radial", 0.15)),
            float(weights.get("hop", 0.05)),
            float(weights.get("breathing", 0.05)),
        ])
        if w.sum() <= 0:
            raise ValueError("All MC move weights are zero.")
        w = w / w.sum()
        move = str(self.rng.choice(names, p=w))
        old_positions = self.atoms.positions.copy()
        n_an = len(self.frag_info.anion_fragments)
        k = int(self.rng.integers(0, n_an))
        frag = self.frag_info.anion_fragments[k]

        if move == "translate":
            sigma = float(self.mc_cfg.get("translation_sigma", 0.20))
            random_translate(self.atoms, frag, sigma, self.rng)
        elif move == "rotate":
            if len(frag) > 1:
                max_ang = float(self.mc_cfg.get("rotation_max_deg", 10.0))
                random_rotate(self.atoms, frag, max_ang, self.rng)
        elif move == "radial":
            radial_sigma = float(self.mc_cfg.get("radial_sigma", 0.20))
            tangent_sigma = float(self.mc_cfg.get("tangential_sigma", 0.20))
            radial_move_fragment(self.atoms, frag, self.origin, radial_sigma, tangent_sigma, self.rng)
        elif move == "hop":
            # Large surface hop: move one anion head center to a random shell point.
            # For anisotropic OTf, optionally reorient SO3->CF3 tail outward after the hop.
            rmin, rmax = self._shell_limits_for_hop()
            r = self.rng.uniform(rmin, rmax)
            normal = random_unit_vector(self.rng)
            target = self.origin + r * normal
            c = anion_head_center(self.atoms, self.frag_info, k)
            translate_fragment(self.atoms, frag, target - c)
            if self.mc_cfg.get("hop_reorient", True) and self.frag_info.anion_type.lower() != "br":
                u = anion_orientation_vector(self.atoms, self.frag_info, k)
                if np.linalg.norm(u) > 1e-8:
                    R = rotation_matrix_align_vectors(u, normal)
                    rotate_fragment(self.atoms, frag, R, center=target)
                    if self.mc_cfg.get("hop_random_roll", True):
                        roll = rotation_matrix_from_axis_angle(normal, self.rng.uniform(0.0, 2.0 * np.pi))
                        rotate_fragment(self.atoms, frag, roll, center=target)
        elif move == "breathing":
            # Radially scale all anion centers; useful for shell relaxation.
            scale_sigma = float(self.mc_cfg.get("breathing_sigma", 0.03))
            scale = float(np.exp(self.rng.normal(scale=scale_sigma)))
            for frag2 in self.frag_info.anion_fragments:
                c = fragment_center(self.atoms, frag2)
                shift = (scale - 1.0) * (c - self.origin)
                translate_fragment(self.atoms, frag2, shift)
        return move, k, old_positions

    def _initial_shell_report(self) -> str:
        shell = self.mc_cfg.get("constraints", {}).get("shell", {})
        if not shell.get("enabled", False):
            return ""
        mode = shell.get("center_mode", "head")
        radii = []
        for k, frag in enumerate(self.frag_info.anion_fragments):
            c = anion_head_center(self.atoms, self.frag_info, k) if mode == "head" else fragment_center(self.atoms, frag)
            radii.append(float(np.linalg.norm(c - self.origin)))
        return f"initial_shell_min={min(radii):.3f}; initial_shell_max={max(radii):.3f}"

    def run(self) -> None:
        n_steps = int(self.cfg.get("n_steps", self.mc_cfg.get("n_steps", 1000)))
        write_every = int(self.cfg.get("write_every", self.mc_cfg.get("write_every", 100)))
        log_every = int(self.cfg.get("log_every", self.mc_cfg.get("log_every", 10)))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.traj_path.exists():
            self.traj_path.unlink()

        e_current, e_ml_current, e_bias_current = self.total_energy()
        self.best_energy = e_current
        self.best_positions = self.atoms.positions.copy()
        write(self.best_path, self.atoms, format="extxyz")

        initial_constraints = self.constraints_ok()
        shell_report = self._initial_shell_report()
        if not initial_constraints:
            print("WARNING: initial structure violates current hard constraints. "
                  "MC will reject most/all moves until constraints are relaxed. "
                  f"{shell_report}")

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(
                "step,E_total_eV,E_ML_eV,E_bias_eV,accepted,acceptance,"
                "rejected_constraint,rejected_energy,rejected_exception,last_move,last_reason,last_dE_eV\n"
            )
            f.write(
                f"0,{e_current:.12f},{e_ml_current:.12f},{e_bias_current:.12f},1,{self.stats.acceptance:.6f},"
                f"0,0,0,init,init,{0.0:.12f}\n"
            )

        last_move = "init"
        last_reason = "init"
        last_dE = 0.0

        for step in trange(1, n_steps + 1, desc="MC"):
            self.stats.attempted += 1
            accepted = False
            try:
                last_move, _, old_positions = self.propose()
            except Exception as exc:
                self.stats.rejected_exception += 1
                if self.debug_raise:
                    raise
                last_reason = "propose_exception:" + str(exc).replace(",", ";")
                last_dE = float("nan")
                continue

            if not self.constraints_ok():
                self.atoms.positions[:] = old_positions
                self.stats.rejected_constraint += 1
                last_reason = "constraint"
                last_dE = float("nan")
            else:
                try:
                    e_new, e_ml_new, e_bias_new = self.total_energy()
                    dE = e_new - e_current
                    last_dE = float(dE)
                    if dE <= 0.0 or self.rng.random() < np.exp(-self.beta * dE):
                        accepted = True
                        last_reason = "accepted"
                        e_current, e_ml_current, e_bias_current = e_new, e_ml_new, e_bias_new
                        self.stats.accepted += 1
                        if e_current < self.best_energy:
                            self.best_energy = e_current
                            self.best_positions = self.atoms.positions.copy()
                            write(self.best_path, self.atoms, format="extxyz")
                    else:
                        self.atoms.positions[:] = old_positions
                        self.stats.rejected_energy += 1
                        last_reason = "metropolis"
                except Exception as exc:
                    self.atoms.positions[:] = old_positions
                    self.stats.rejected_exception += 1
                    if self.debug_raise:
                        raise
                    last_reason = "energy_exception:" + str(exc).replace(",", ";")
                    last_dE = float("nan")

            if step % log_every == 0:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{step},{e_current:.12f},{e_ml_current:.12f},{e_bias_current:.12f},"
                        f"{int(accepted)},{self.stats.acceptance:.6f},"
                        f"{self.stats.rejected_constraint},{self.stats.rejected_energy},{self.stats.rejected_exception},"
                        f"{last_move},{last_reason},{last_dE:.12f}\n"
                    )
            if step % write_every == 0:
                self.atoms.info["E_total_eV"] = float(e_current)
                self.atoms.info["E_ML_eV"] = float(e_ml_current)
                self.atoms.info["E_bias_eV"] = float(e_bias_current)
                self.atoms.info["mc_step"] = int(step)
                write(self.traj_path, self.atoms, format="extxyz", append=True)

        if self.best_positions is not None:
            self.atoms.positions[:] = self.best_positions
            write(self.best_path, self.atoms, format="extxyz")
