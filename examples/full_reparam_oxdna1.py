"""Full reparameterization of oxDNA1."""

import argparse
import copy
import json
import logging
from collections.abc import Callable
from dataclasses import InitVar
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import chex
import jax
import jax.numpy as jnp
import jax_md
import mythos.energy.dna1 as dna1_energy
import mythos.input.topology as jdna_top
import mythos.observables.base as obs_base
import mythos.optimization.objective as jdna_objective
import mythos.optimization.optimization as jdna_optimization
import optax
import pandas as pd
from mythos.energy.base import EnergyFunction
from mythos.observables.diameter import Diameter
from mythos.observables.melting_temp import MeltingTemp
from mythos.observables.persistence_length import PersistenceLength
from mythos.observables.pitch import PitchAngle, compute_pitch
from mythos.observables.propeller import PropellerTwist
from mythos.observables.rise import Rise
from mythos.observables.stretch_torsion import ExtensionZ, TwistXY, stretch_torsion
from mythos.simulators.base import SimulatorOutput
from mythos.simulators.lammps.lammps_oxdna import LAMMPSoxDNASimulator
from mythos.simulators.oxdna import oxDNASimulator
from mythos.simulators.oxdna.oxdna import UmbrellaEnergyInfo, oxDNAUmbrellaSampler
from mythos.ui.loggers.console import ConsoleLogger
from mythos.ui.loggers.disk import FileLogger
from mythos.ui.loggers.multilogger import MultiLogger
from mythos.utils.types import Params

jax.config.update("jax_enable_x64", val=True)
logging.basicConfig(level=logging.INFO)

# Type alias for DiffTRe loss function return type
LossOutput = tuple[float, tuple[tuple[str, float], dict]]
NM_PER_OXDNA_LENGTH_UNIT = 0.8518  # 1 oxDNA length unit = 0.8518 nm

# Default configuration for all objectives and simulators
DEFAULT_CONFIG: dict[str, Any] = {
    "mechanical": {
        "system_dir": "data/full_reparam_oxdna1/mechanical/lammps/40bp_duplex",
        "temperature": "300K",
        "stretch_forces": [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40],  # pN
        "stretch_torque": 0,  # pN·nm
        "twist_torques": [5, 10, 15, 20, 25, 30],
        "twist_force": 2,  # pN
        "n_replicas": 1,
        "n_steps": 1_250_000,
        "targets": {
            "stretch_modulus": 1000.0,  # pN
            "torsional_modulus": 460.0,  # pN·nm^2
            "twist_stretch_coupling": -90.0,  # pN·nm
        },
    },
    "structural": {
        "system_dir": "data/full_reparam_oxdna1/structural/20bp_duplex",
        "temperature": "300K",
        "n_replicas": 1,
        "n_steps": 1_000_000,
        "snapshot_interval": 10_000,
        "sigma_backbone": 0.70,  # oxDNA1 default excluded volume sigma
        "targets": {
            "helix_diameter": 20.0,  # Angstroms
            "helical_pitch": 3.57,  # nm
            "propeller_twist": -12.6,  # degrees
            "rise": 0.34,  # nm
        },
    },
    "persistence": {
        "system_dir": "data/full_reparam_oxdna1/mechanical/60bp_duplex",
        "temperature": "300K",
        "n_replicas": 1,
        "n_steps": 1_000_000,
        "snapshot_interval": 10_000,
        "n_equilibration_steps": 20,
        "targets": {
            "persistence_length": 50.0,  # nm
        },
    },
    "thermo": {
        "1": {
            "system_dir": "data/full_reparam_oxdna1/thermodynamic/5bp_duplex",
            "temperature": "300K",
            "n_replicas": 10,
            "n_steps": 1_000_000,
            "snapshot_interval": 10_000,
            "temperature_range": [260, 340],  # K, for extrapolation
            "temperature_range_points": 20,
            "targets": {
                "melting_temperature": 294.2,  # K
            },
            "enable": True,
        },
        "2": {
            "system_dir": "data/full_reparam_oxdna1/thermodynamic/8bp_duplex",
            "temperature": "300K",
            "n_replicas": 10,
            "n_steps": 1_000_000,
            "snapshot_interval": 10_000,
            "temperature_range": [290, 360],  # K, for extrapolation
            "temperature_range_points": 20,
            "targets": {
                "melting_temperature": 324.6,  # K
            },
            "enable": True,
        }
    },
}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    def deep_merge(base: dict, overrides: dict) -> dict:
        for key, value in overrides.items():
            if isinstance(base.get(key), dict):
                deep_merge(base[key], value)
            elif key in base:
                base[key] = value
        return base

    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is not None:
        with config_path.open() as f:
            overrides = json.load(f)
        deep_merge(config, overrides)
    return config


@chex.dataclass(frozen=True, kw_only=True)
class LAMMPSStretchSimulator(LAMMPSoxDNASimulator):
    force: InitVar[float] = None
    torque: InitVar[float] = None

    def __post_init__(self, force: float, torque: float) -> None:
        LAMMPSoxDNASimulator.__post_init__(self)
        object.__setattr__(self, "variables", {**self.variables, "force": force, "torque": torque})

    def run_simulation(self, *args, opt_params: Params, **kwargs) -> SimulatorOutput:
        output = LAMMPSoxDNASimulator.run_simulation(self, *args, params=opt_params, **kwargs)
        tagged_traj = output.observables[0].with_state_metadata(self.variables.copy())
        return SimulatorOutput(observables=[tagged_traj], state=output.state)


def create_stretch_twist_simulators(
    input_dir: Path,
    energy_fn: EnergyFunction,
    mech_cfg: dict[str, Any],
    variables: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[LAMMPSStretchSimulator]:
    stretch_forces = mech_cfg["stretch_forces"]
    stretch_torque = mech_cfg["stretch_torque"]
    twist_torques = mech_cfg["twist_torques"]
    twist_force = mech_cfg["twist_force"]
    n_replicas = mech_cfg["n_replicas"]
    variables = variables or {}
    config_pairs = [(stretch_torque, f) for f in stretch_forces] + [(t, twist_force) for t in twist_torques]
    return [
        LAMMPSStretchSimulator(
            input_dir=str(input_dir),
            energy_fn=energy_fn,
            force=force,
            torque=torque,
            name=f"stretch_f{force}_t{torque}_r{r}",
            variables=variables.copy(),
            **kwargs,
        )
        for (torque, force) in config_pairs
        for r in range(n_replicas)
    ]


def read_box_size_lammps(data_path: Path) -> jnp.ndarray:
    box = [-1.0, -1.0, -1.0]
    with data_path.open("r") as f:
        for line in f:
            if line.strip().endswith(" xlo xhi"):
                box[0] = float(line.split()[1]) - float(line.split()[0])
            if line.strip().endswith(" ylo yhi"):
                box[1] = float(line.split()[1]) - float(line.split()[0])
            if line.strip().endswith(" zlo zhi"):
                box[2] = float(line.split()[1]) - float(line.split()[0])
            if all(dim > 0 for dim in box):
                break
    return jnp.array(box)


def read_box_size_oxdna(conf_path: Path) -> jnp.ndarray:
    """Box size is on line 2: b = x y z"""
    with conf_path.open("r") as f:
        # Skip first line (time)
        f.readline()
        # Read box size line
        box_line = f.readline().strip()
        # Parse "b = x y z" format
        parts = box_line.split("=")[1].strip().split()
        return jnp.array([float(x) for x in parts])


def tree_mean(trees: list) -> Any:
    if len(trees) == 1:
        return trees[0]
    return jax.tree.map(lambda *x: jnp.mean(jnp.stack(x)), *trees)


def setup_oxdna_system(system_dir: Path, kt: float) -> tuple[jdna_top.Topology, Callable, EnergyFunction]:
    topology = jdna_top.from_oxdna_file(system_dir / "sys.top")
    box = read_box_size_oxdna(system_dir / "sys.conf")
    displacement_fn = jax_md.space.periodic(box)[0]

    energy_fn = dna1_energy.create_default_energy_fn(
        topology=topology,
        displacement_fn=displacement_fn,
    ).with_params(kt=kt)

    return topology, displacement_fn, energy_fn


def setup_lammps_system(system_dir: Path, kt: float) -> tuple[jdna_top.Topology, Callable, EnergyFunction]:
    topology = jdna_top.from_oxdna_file(system_dir / "data.top")
    box = read_box_size_lammps(system_dir / "data")
    displacement_fn = jax_md.space.periodic(box)[0]

    energy_fn = dna1_energy.create_default_energy_fn(
        topology=topology,
        displacement_fn=displacement_fn,
    ).without_terms(
        "BondedExcludedVolume"  # LAMMPS doesn't implement this term
    ).with_params(kt=kt)

    return topology, displacement_fn, energy_fn


def create_stretch_torsion_objectives(
    simulators: list[LAMMPSStretchSimulator],
    energy_fn: EnergyFunction,
    top: jdna_top.Topology,
    displacement_fn: Callable,
    kt: float,
    mech_cfg: dict[str, Any],
) -> list[jdna_objective.DiffTReObjective]:
    stretch_torque = mech_cfg["stretch_torque"]
    twist_force = mech_cfg["twist_force"]
    stretch_forces_list = mech_cfg["stretch_forces"]
    twist_torques_list = mech_cfg["twist_torques"]
    target_stretch_modulus = mech_cfg["targets"]["stretch_modulus"]
    target_torsion_modulus = mech_cfg["targets"]["torsional_modulus"]
    target_twist_stretch_coupling = mech_cfg["targets"]["twist_stretch_coupling"]

    transform_fn = energy_fn.energy_fns[0].transform_fn
    n_nucs_per_strand = top.n_nucleotides // 2

    # Define end base pairs for extension measurement
    bp1 = jnp.array([0, 2 * n_nucs_per_strand - 1], dtype=jnp.int32)
    bp2 = jnp.array([n_nucs_per_strand - 1, n_nucs_per_strand], dtype=jnp.int32)

    # Get quartets for twist measurement
    quartets = obs_base.get_duplex_quartets(n_nucs_per_strand)

    # Create observables
    extension_obs = ExtensionZ(rigid_body_transform_fn=transform_fn, bp1=bp1, bp2=bp2, displacement_fn=displacement_fn)
    twist_obs = TwistXY(rigid_body_transform_fn=transform_fn, quartets=quartets, displacement_fn=displacement_fn)
    beta = jnp.array(1 / kt, dtype=jnp.float64)

    # Precompute force/torque values (constants for differentiation)
    stretch_forces = jnp.array(stretch_forces_list, dtype=jnp.float64)
    twist_torques = jnp.array(twist_torques_list, dtype=jnp.float64)
    n_force_segments = len(stretch_forces_list)
    n_torque_segments = len(twist_torques_list)

    def compute_moduli_from_traj(
        traj: Any, weights: jnp.ndarray
    ) -> tuple[float, float, float]:
        forces_arr = jnp.array([md["force"] for md in traj.metadata])
        torques_arr = jnp.array([md["torque"] for md in traj.metadata])
        force_segment_ids = jnp.searchsorted(stretch_forces, forces_arr).astype(jnp.int32)
        torque_segment_ids = jnp.searchsorted(twist_torques, torques_arr).astype(jnp.int32)
        stretch_mask = torques_arr == stretch_torque  # held torque for stretch experiments
        twist_mask = forces_arr == twist_force  # held force for twist experiments

        # Compute observables for all states once
        all_extensions = extension_obs(traj) * NM_PER_OXDNA_LENGTH_UNIT
        all_twists = twist_obs(traj)

        # For stretch experiments: weighted average extension per force level using segment_sum
        # Zero out weights for non-stretch states, then aggregate by force segment
        stretch_weights = jnp.where(stretch_mask, weights, 0.0)
        weighted_stretch_ext = stretch_weights * all_extensions
        force_ext_sum = jax.ops.segment_sum(weighted_stretch_ext, force_segment_ids, num_segments=n_force_segments)
        force_weight_sum = jax.ops.segment_sum(stretch_weights, force_segment_ids, num_segments=n_force_segments)
        force_extensions = force_ext_sum / (force_weight_sum + 1e-10)

        # For twist experiments: weighted average extension and twist per torque level
        twist_weights = jnp.where(twist_mask, weights, 0.0)
        weighted_twist_ext = twist_weights * all_extensions
        weighted_twist_tw = twist_weights * all_twists
        torque_ext_sum = jax.ops.segment_sum(weighted_twist_ext, torque_segment_ids, num_segments=n_torque_segments)
        torque_twist_sum = jax.ops.segment_sum(weighted_twist_tw, torque_segment_ids, num_segments=n_torque_segments)
        torque_weight_sum = jax.ops.segment_sum(twist_weights, torque_segment_ids, num_segments=n_torque_segments)
        torque_extensions = torque_ext_sum / (torque_weight_sum + 1e-10)
        torque_twists = torque_twist_sum / (torque_weight_sum + 1e-10)

        return stretch_torsion(
            stretch_forces, force_extensions, twist_torques, torque_extensions, torque_twists
        )

    def stretch_modulus_loss_fn(traj: Any, weights: jnp.ndarray, *_, **__) -> LossOutput:
        s_eff, c, g = compute_moduli_from_traj(traj, weights)
        loss = jnp.sqrt((s_eff - target_stretch_modulus) ** 2)
        return loss, (("stretch_modulus", s_eff), {"torsional_modulus": c, "twist_stretch_coupling": g})

    def torsional_modulus_loss_fn(traj: Any, weights: jnp.ndarray, *_, **__) -> LossOutput:
        s_eff, c, g = compute_moduli_from_traj(traj, weights)
        loss = jnp.sqrt((c - target_torsion_modulus) ** 2)
        return loss, (("torsional_modulus", c), {"stretch_modulus": s_eff, "twist_stretch_coupling": g})

    def twist_stretch_coupling_loss_fn(traj: Any, weights: jnp.ndarray, *_, **__) -> LossOutput:
        s_eff, c, g = compute_moduli_from_traj(traj, weights)
        loss = jnp.sqrt((g - target_twist_stretch_coupling) ** 2)
        return loss, (("twist_stretch_coupling", g), {"stretch_modulus": s_eff, "torsional_modulus": c})

    # Collect required observables from all simulators
    required_observables = [obs for sim in simulators for obs in sim.exposes()]

    # Common arguments for all objectives
    common_kwargs = {
        "required_observables": required_observables,
        "energy_fn": energy_fn,
        "beta": beta,
        "n_equilibration_steps": 20,
        "min_n_eff_factor": 0.95,
        "max_valid_opt_steps": 10,
    }

    # Create the three objectives
    stretch_objective = jdna_objective.DiffTReObjective(
        name="stretch_modulus",
        grad_or_loss_fn=stretch_modulus_loss_fn,
        **common_kwargs,
    )

    torsional_objective = jdna_objective.DiffTReObjective(
        name="torsional_modulus",
        grad_or_loss_fn=torsional_modulus_loss_fn,
        **common_kwargs,
    )

    coupling_objective = jdna_objective.DiffTReObjective(
        name="twist_stretch_coupling",
        grad_or_loss_fn=twist_stretch_coupling_loss_fn,
        **common_kwargs,
    )

    return [stretch_objective, torsional_objective, coupling_objective]


def create_structural_simulators(
    input_dir: Path,
    energy_fn: EnergyFunction,
    struct_cfg: dict[str, Any],
    **kwargs: Any,
) -> list[oxDNASimulator]:
    n_steps = struct_cfg["n_steps"]
    snapshot_interval = struct_cfg["snapshot_interval"]
    n_replicas = struct_cfg["n_replicas"]
    overrides = {"steps": n_steps, "print_conf_interval": snapshot_interval, "print_energy_interval": snapshot_interval}
    return [
        oxDNASimulator(
            input_dir=str(input_dir),
            energy_fn=energy_fn,
            name=f"structural_20bp_r{r}",
            input_overrides=overrides,
            **kwargs,
        )
        for r in range(n_replicas)
    ]


def get_h_bonded_base_pairs(n_nucs_per_strand: int) -> jnp.ndarray:
    s1_nucs = list(range(n_nucs_per_strand))
    s2_nucs = list(range(n_nucs_per_strand, n_nucs_per_strand * 2))
    s2_nucs.reverse()
    return jnp.array(list(zip(s1_nucs, s2_nucs, strict=True)), dtype=jnp.int32)


def create_structural_objectives(
    simulators: list[oxDNASimulator],
    energy_fn: EnergyFunction,
    top: jdna_top.Topology,
    displacement_fn: Callable,
    kt: float,
    struct_cfg: dict[str, Any],
) -> list[jdna_objective.DiffTReObjective]:
    target_helix_diameter = struct_cfg["targets"]["helix_diameter"]
    target_helical_pitch = struct_cfg["targets"]["helical_pitch"]
    target_propeller_twist = struct_cfg["targets"]["propeller_twist"]
    target_rise = struct_cfg["targets"]["rise"]
    sigma_backbone = struct_cfg["sigma_backbone"]

    transform_fn = energy_fn.energy_fns[0].transform_fn
    n_nucs_per_strand = top.n_nucleotides // 2

    # Get base pairs and quartets for observables
    h_bonded_bps = get_h_bonded_base_pairs(n_nucs_per_strand)
    quartets = obs_base.get_duplex_quartets(n_nucs_per_strand)

    # Create observables
    diameter_obs = Diameter(
        rigid_body_transform_fn=transform_fn,
        h_bonded_base_pairs=h_bonded_bps,
        displacement_fn=displacement_fn,
    )
    pitch_obs = PitchAngle(
        rigid_body_transform_fn=transform_fn,
        quartets=quartets,
        displacement_fn=displacement_fn,
    )
    propeller_obs = PropellerTwist(
        rigid_body_transform_fn=transform_fn,
        h_bonded_base_pairs=h_bonded_bps,
    )
    rise_obs = Rise(
        rigid_body_transform_fn=transform_fn,
        quartets=quartets,
        displacement_fn=displacement_fn,
    )

    beta = jnp.array(1 / kt, dtype=jnp.float64)

    # Conversion: Angstroms to nm
    angstrom_to_nm = 0.1

    def helix_diameter_loss_fn(traj: Any, weights: jnp.ndarray, *_, **__) -> LossOutput:
        diameters = diameter_obs(traj, sigma_backbone)  # Returns Angstroms
        expected_diameter = jnp.dot(weights, diameters)
        loss = jnp.sqrt((expected_diameter - target_helix_diameter) ** 2)
        return loss, (("helix_diameter", expected_diameter), {})

    def helical_pitch_loss_fn(traj: Any, weights: jnp.ndarray, *_, **__) -> LossOutput:
        pitch_angles = pitch_obs(traj)  # Returns radians
        expected_pitch_angle = jnp.dot(weights, pitch_angles)
        bp_per_turn = compute_pitch(expected_pitch_angle)
        # Is this how we should do it??
        # Helical pitch = bp/turn * rise (in nm);
        rises = rise_obs(traj) * angstrom_to_nm  # Convert Å to nm
        expected_rise = jnp.dot(weights, rises)
        helical_pitch_nm = bp_per_turn * expected_rise
        loss = jnp.sqrt((helical_pitch_nm - target_helical_pitch) ** 2)
        return loss, (("helical_pitch", helical_pitch_nm), {})

    def propeller_twist_loss_fn(traj: Any, weights: jnp.ndarray, *_, **__) -> LossOutput:
        obs = propeller_obs(traj)
        expected_prop_twist = jnp.dot(weights, obs)
        loss = jnp.sqrt((expected_prop_twist - target_propeller_twist) ** 2)
        return loss, (("prop_twist", expected_prop_twist), {})

    def rise_loss_fn(traj: Any, weights: jnp.ndarray, *_, **__) -> LossOutput:
        rises = rise_obs(traj) * angstrom_to_nm  # Convert Å to nm
        expected_rise = jnp.dot(weights, rises)
        loss = jnp.sqrt((expected_rise - target_rise) ** 2)
        return loss, (("rise", expected_rise), {})

    # Collect required observables from all simulators
    required_observables = [obs for sim in simulators for obs in sim.exposes()]

    # Common arguments for all objectives
    common_kwargs = {
        "required_observables": required_observables,
        "energy_fn": energy_fn,
        "beta": beta,
        "n_equilibration_steps": 20,
        "min_n_eff_factor": 0.95,
        "max_valid_opt_steps": 10,
    }

    # Create the four structural objectives
    diameter_objective = jdna_objective.DiffTReObjective(
        name="helix_diameter",
        grad_or_loss_fn=helix_diameter_loss_fn,
        **common_kwargs,
    )

    pitch_objective = jdna_objective.DiffTReObjective(
        name="helical_pitch",
        grad_or_loss_fn=helical_pitch_loss_fn,
        **common_kwargs,
    )

    propeller_objective = jdna_objective.DiffTReObjective(
        name="propeller_twist",
        grad_or_loss_fn=propeller_twist_loss_fn,
        **common_kwargs,
    )

    rise_objective = jdna_objective.DiffTReObjective(
        name="rise",
        grad_or_loss_fn=rise_loss_fn,
        **common_kwargs,
    )

    return [diameter_objective, pitch_objective, propeller_objective, rise_objective]


def create_persistence_simulators(
    input_dir: Path,
    energy_fn: EnergyFunction,
    persist_cfg: dict[str, Any],
    **kwargs: Any,
) -> list[oxDNASimulator]:
    n_steps = persist_cfg["n_steps"]
    snapshot_interval = persist_cfg["snapshot_interval"]
    n_replicas = persist_cfg["n_replicas"]
    overrides = {"steps": n_steps, "print_conf_interval": snapshot_interval, "print_energy_interval": snapshot_interval}
    return [
        oxDNASimulator(
            input_dir=str(input_dir),
            energy_fn=energy_fn,
            name=f"persistence_60bp_r{r}",
            input_overrides=overrides,
            **kwargs,
        )
        for r in range(n_replicas)
    ]


def create_persistence_objective(
    simulators: list[oxDNASimulator],
    energy_fn: EnergyFunction,
    top: jdna_top.Topology,
    displacement_fn: Callable,
    kt: float,
    persist_cfg: dict[str, Any],
) -> jdna_objective.DiffTReObjective:
    target_persistence_length = persist_cfg["targets"]["persistence_length"]
    n_equilibration_steps = persist_cfg["n_equilibration_steps"]

    transform_fn = energy_fn.energy_fns[0].transform_fn
    n_nucs_per_strand = top.n_nucleotides // 2
    quartets = obs_base.get_duplex_quartets(n_nucs_per_strand)

    persistence_obs = PersistenceLength(
        rigid_body_transform_fn=transform_fn,
        quartets=quartets,
        displacement_fn=displacement_fn,
    )

    beta = jnp.array(1 / kt, dtype=jnp.float64)

    def persistence_length_loss_fn(traj: Any, weights: jnp.ndarray, *_, **__) -> LossOutput:
        lp = persistence_obs(traj, weights) * NM_PER_OXDNA_LENGTH_UNIT
        loss = (lp - target_persistence_length) ** 2
        return loss, (("persistence_length", lp), {})

    return jdna_objective.DiffTReObjective(
        name="persistence_length",
        grad_or_loss_fn=persistence_length_loss_fn,
        required_observables=[obs for sim in simulators for obs in sim.exposes()],
        energy_fn=energy_fn,
        beta=beta,
        n_equilibration_steps=n_equilibration_steps,
        max_valid_opt_steps=10,
    )


def create_thermo_simulators(
    input_dir: Path,
    energy_fn: EnergyFunction,
    thermo_cfg: dict[str, Any],
    thermo_name: str = "default",
    **kwargs: Any,
) -> list[oxDNAUmbrellaSampler]:
    n_steps = thermo_cfg["n_steps"]
    n_replicas = thermo_cfg["n_replicas"]
    snapshot_interval = thermo_cfg["snapshot_interval"]
    overrides = {
        "steps": n_steps,
        "equilibration_steps": int(0.5 * n_steps),
        "print_energy_interval": snapshot_interval,
        "print_conf_interval": snapshot_interval,
    }
    return [
        oxDNAUmbrellaSampler(
            input_dir=str(input_dir),
            energy_fn=energy_fn,
            name=f"thermo_{thermo_name}_r{r}",
            input_overrides=overrides,
            **kwargs,
        )
        for r in range(n_replicas)
    ]


def create_thermo_objective(
    simulators: list[oxDNAUmbrellaSampler],
    energy_fn: EnergyFunction,
    kt: float,
    thermo_cfg: dict[str, Any],
    thermo_name: str = "default",
) -> jdna_objective.DiffTReObjective:
    from mythos.utils.units import get_kt

    target_melting_temp = thermo_cfg["targets"]["melting_temperature"]
    temp_range = thermo_cfg["temperature_range"]
    temp_range_points = thermo_cfg["temperature_range_points"]

    # Get kt values for temperature range (for extrapolation)
    kt_range = get_kt(jnp.linspace(temp_range[0], temp_range[1], temp_range_points))

    melting_temp_fn = MeltingTemp(
        rigid_body_transform_fn=1,  # not used
        sim_temperature=kt,
        temperature_range=kt_range,
        energy_fn=energy_fn,
    )
    # Target in sim units (kt)
    target_kt = get_kt(target_melting_temp)
    beta = jnp.array(1 / kt, dtype=jnp.float64)

    def melting_temp_loss_fn(
        traj: Any, weights: jnp.ndarray, _energy_model: Any, opt_params: Params, observables: list
    ) -> LossOutput:
        # Filter energy info from observables
        e_info = pd.concat([i for i in observables if isinstance(i, UmbrellaEnergyInfo)])
        melting_temp = melting_temp_fn(traj, e_info["bond"].to_numpy(), e_info["weight"].to_numpy(), opt_params)
        expected_melting_temp = jnp.dot(weights, melting_temp).sum()
        loss = jnp.sqrt((expected_melting_temp - target_kt) ** 2)
        return loss, (("melting_temperature", expected_melting_temp), {})

    return jdna_objective.DiffTReObjective(
        name=f"melting_temperature_{thermo_name}",
        grad_or_loss_fn=melting_temp_loss_fn,
        required_observables=[obs for sim in simulators for obs in sim.exposes()],
        energy_fn=energy_fn,
        beta=beta,
        n_equilibration_steps=0,  # Equilibration handled in simulation run
        min_n_eff_factor=0.95,
    )

def thermo_reweight_simulators(group: list[str], component_states: dict[str, Any]) -> None:
    all_weights = pd.concat([component_states[i]["weights"] for i in group])
    all_weights = all_weights.groupby(all_weights.index.names).mean()
    all_weights /= all_weights.min()
    for name in group:
        component_states[name]["weights"] = all_weights

# Valid objective names for command-line selection
MECHANICAL_OBJECTIVES = {"stretch_modulus", "torsional_modulus", "twist_stretch_coupling"}
PERSISTENCE_OBJECTIVES = {"persistence_length"}
STRUCTURAL_OBJECTIVES = {"helix_diameter", "helical_pitch", "propeller_twist", "rise"}
THERMO_OBJECTIVES = {"melting_temperature"}
VALID_OBJECTIVES = MECHANICAL_OBJECTIVES | PERSISTENCE_OBJECTIVES | STRUCTURAL_OBJECTIVES | THERMO_OBJECTIVES


def run_optimization(
    config: dict[str, Any],
    learning_rate: float = 5e-4,
    opt_steps: int = 100,
    selected_objectives: set[str] = VALID_OBJECTIVES,
    use_aim: bool = False,
    oxdna_source_path: Path | None = None,
    metrics_file: Path | None = None,
) -> Params:
    import ray
    from mythos.utils.units import get_kt_from_string
    from tqdm import tqdm

    # Get per-system kt values from config
    mech_kt = get_kt_from_string(config["mechanical"]["temperature"])
    struct_kt = get_kt_from_string(config["structural"]["temperature"])
    persist_kt = get_kt_from_string(config["persistence"]["temperature"])

    # Get system directories from config
    mechanical_input_dir = Path(config["mechanical"]["system_dir"])
    structural_input_dir = Path(config["structural"]["system_dir"])
    persistence_input_dir = Path(config["persistence"]["system_dir"])

    # Determine which objective categories are needed
    need_mechanical = bool(selected_objectives & MECHANICAL_OBJECTIVES)
    need_persistence = bool(selected_objectives & PERSISTENCE_OBJECTIVES)
    need_structural = bool(selected_objectives & STRUCTURAL_OBJECTIVES)
    need_thermo = any(obj.startswith("melting_temperature") for obj in selected_objectives)

    all_simulators = []
    all_objectives = []
    # create a default energy function to get opt_params for all cases. Since
    # the EF is throw away (but topo and disp fn are needed), we can use mocks
    # here (they do not affect the opt_params structure).
    opt_params = dna1_energy.create_default_energy_fn(topology=MagicMock(), displacement_fn=MagicMock()).opt_params()

    # Setup mechanical objectives (LAMMPS simulators, 40bp duplex)
    if need_mechanical:
        mech_top, mech_displacement_fn, mech_energy_fn = setup_lammps_system(mechanical_input_dir, mech_kt)

        stretch_twist_simulators = create_stretch_twist_simulators(
            input_dir=mechanical_input_dir,
            energy_fn=mech_energy_fn,
            mech_cfg=config["mechanical"],
            input_file_name="in",
            variables={"T": mech_kt, "nsteps": config["mechanical"]["n_steps"]},
        )

        mechanical_objectives = create_stretch_torsion_objectives(
            simulators=stretch_twist_simulators,
            energy_fn=mech_energy_fn,
            top=mech_top,
            displacement_fn=mech_displacement_fn,
            kt=mech_kt,
            mech_cfg=config["mechanical"],
        )

        all_simulators.extend(stretch_twist_simulators)
        all_objectives.extend(mechanical_objectives)

    # Setup structural objectives (oxDNA MD simulator, 20bp duplex)
    if need_structural:
        struct_top, struct_displacement_fn, struct_energy_fn = setup_oxdna_system(
            structural_input_dir, struct_kt
        )

        structural_simulators = create_structural_simulators(
            input_dir=structural_input_dir,
            energy_fn=struct_energy_fn,
            struct_cfg=config["structural"],
            source_path=oxdna_source_path,
        )

        structural_objectives = create_structural_objectives(
            simulators=structural_simulators,
            energy_fn=struct_energy_fn,
            top=struct_top,
            displacement_fn=struct_displacement_fn,
            kt=struct_kt,
            struct_cfg=config["structural"],
        )

        all_simulators.extend(structural_simulators)
        all_objectives.extend(structural_objectives)

    # Setup persistence length objective (oxDNA MD simulator, 60bp duplex)
    if need_persistence:
        persist_top, persist_displacement_fn, persist_energy_fn = setup_oxdna_system(
            persistence_input_dir, persist_kt
        )

        persistence_simulators = create_persistence_simulators(
            input_dir=persistence_input_dir,
            energy_fn=persist_energy_fn,
            persist_cfg=config["persistence"],
            source_path=oxdna_source_path,
        )

        persistence_objective = create_persistence_objective(
            simulators=persistence_simulators,
            energy_fn=persist_energy_fn,
            top=persist_top,
            displacement_fn=persist_displacement_fn,
            kt=persist_kt,
            persist_cfg=config["persistence"],
        )

        all_simulators.extend(persistence_simulators)
        all_objectives.append(persistence_objective)

    # Setup thermo objectives (oxDNA umbrella sampling)
    # Iterate over enabled thermo sub-configs
    thermo_simulator_groups = []  # Groups of simulator names for reweighting
    if need_thermo:
        for thermo_name, thermo_cfg in config["thermo"].items():
            if not thermo_cfg.get("enable", False):
                continue

            thermo_input_dir = Path(thermo_cfg["system_dir"])
            thermo_kt = get_kt_from_string(thermo_cfg["temperature"])

            thermo_top, thermo_displacement_fn, thermo_energy_fn = setup_oxdna_system(
                thermo_input_dir, thermo_kt
            )

            thermo_simulators = create_thermo_simulators(
                input_dir=thermo_input_dir,
                energy_fn=thermo_energy_fn,
                thermo_cfg=thermo_cfg,
                thermo_name=thermo_name,
                source_path=oxdna_source_path,
            )

            thermo_objective = create_thermo_objective(
                simulators=thermo_simulators,
                energy_fn=thermo_energy_fn,
                kt=thermo_kt,
                thermo_cfg=thermo_cfg,
                thermo_name=thermo_name,
            )

            all_simulators.extend(thermo_simulators)
            all_objectives.append(thermo_objective)
            thermo_simulator_groups.append([s.name for s in thermo_simulators])

    logging.info("Using objectives: %s", [obj.name for obj in all_objectives])

    # Setup Ray optimizer
    optimizer = jdna_optimization.RayOptimizer(
        objectives=all_objectives,
        simulators=all_simulators,
        aggregate_grad_fn=tree_mean,
        optimizer=optax.adam(learning_rate=learning_rate),
    )

    # Run optimization loop
    logging.info("Starting oxDNA1 reparameterization with %d objectives", len(all_objectives))
    if need_mechanical:
        mech_targets = config["mechanical"]["targets"]
        logging.info(
            "  Mechanical targets: S_eff=%.1f pN, C=%.1f pN·nm², g=%.1f pN·nm",
            mech_targets["stretch_modulus"],
            mech_targets["torsional_modulus"],
            mech_targets["twist_stretch_coupling"],
        )
    if need_structural:
        struct_targets = config["structural"]["targets"]
        logging.info(
            "  Structural targets: diameter=%.1f Å, pitch=%.2f nm, propeller=%.1f°, rise=%.2f nm",
            struct_targets["helix_diameter"],
            struct_targets["helical_pitch"],
            struct_targets["propeller_twist"],
            struct_targets["rise"],
        )
    if need_persistence:
        persist_targets = config["persistence"]["targets"]
        logging.info("  Persistence length target: Lp=%.1f nm", persist_targets["persistence_length"])
    if need_thermo:
        for thermo_name, thermo_cfg in config["thermo"].items():
            if thermo_cfg.get("enable", False):
                logging.info(
                    "  Melting temperature target (%s): Tm=%.1f K",
                    thermo_name,
                    thermo_cfg["targets"]["melting_temperature"],
                )

    loggers = [ConsoleLogger()]
    if use_aim:
        from mythos.ui.loggers.aim import AimLogger
        name = f"full_reparam_oxdna1-{len(all_objectives)}obj"
        if len(all_objectives) == 1:
            name = f"oxdna1-{all_objectives[0].name}"
        aim_logger = AimLogger(experiment=name)
        loggers.append(aim_logger)
    if metrics_file:
        loggers.append(FileLogger(metrics_file, mode="w"))
    logger = MultiLogger(loggers)

    state = None
    for step in tqdm(range(opt_steps), desc="Optimizing oxDNA1 parameters"):
        output = optimizer.step(opt_params, state)
        opt_params = output.opt_params
        state = output.state

        # For umbrella sampling, combine weights across all thermo simulators in
        # each group, this modifies component states in place
        for group in thermo_simulator_groups:
            thermo_reweight_simulators(group, state.component_state)

        # Log metrics
        for ctx, obs in output.observables.items():
            for name in obs:
                logger.log_metric(f"{ctx}.{name}", obs[name], step=step)

    logging.info("Optimization complete!")
    logging.info("Final parameters: %s", opt_params)

    ray.shutdown()
    return opt_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full oxDNA1 reparameterization: mechanical (S_eff, C, g), "
            "structural (diameter, pitch, propeller, rise), persistence length, "
            "and melting temperature properties."
        )
    )
    parser.add_argument(
        "--objectives",
        type=str,
        default=None,
        help=(
            "Comma-separated list of objectives to optimize. "
            f"Mechanical: {', '.join(sorted(MECHANICAL_OBJECTIVES))}. "
            f"Persistence: {', '.join(sorted(PERSISTENCE_OBJECTIVES))}. "
            f"Structural: {', '.join(sorted(STRUCTURAL_OBJECTIVES))}. "
            f"Thermo: {', '.join(sorted(THERMO_OBJECTIVES))}. "
            "If not specified, all objectives are used."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate for optimization (default: 5e-4).",
    )
    parser.add_argument(
        "--opt-steps",
        type=int,
        default=100,
        help="Number of optimization steps (default: 100).",
    )
    parser.add_argument(
        "--oxdna-source",
        type=str,
        default=None,
        help="Path to oxDNA source directory (required for oxDNA based simulators).",
    )
    parser.add_argument(
        "--use-aim",
        action="store_true",
        help="Enable Aim experiment tracking.",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Path to file for logging metrics. If not specified, no file logging.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file to override default targets and simulator settings.",
    )
    parser.add_argument(
        "--dump-config",
        type=str,
        default=None,
        help="Path to dump the default config JSON and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config (with optional overrides from file)
    config = load_config(Path(args.config) if args.config else None)
    if args.dump_config:
        if args.dump_config == "-":
            print(json.dumps(config, indent=2))
        else:
            with Path(args.dump_config).open("w") as f:
                json.dump(config, f, indent=2)
        raise SystemExit

    # Parse and validate objectives
    if args.objectives:
        selected_objectives = {obj.strip() for obj in args.objectives.split(",")}
        invalid = selected_objectives - VALID_OBJECTIVES
        if invalid:
            raise ValueError(
                f"Invalid objective(s): {invalid}. "
                f"Valid options are: {VALID_OBJECTIVES}"
            )
    else:
        selected_objectives = VALID_OBJECTIVES

    # Check if oxDNA source is needed
    oxdna_source = Path(args.oxdna_source).resolve() if args.oxdna_source else None
    need_oxdna = bool(selected_objectives & (STRUCTURAL_OBJECTIVES | PERSISTENCE_OBJECTIVES | THERMO_OBJECTIVES))
    if need_oxdna and oxdna_source is None:
        raise ValueError("--oxdna-source is required when using oxDNA based objectives")

    run_optimization(
        config=config,
        learning_rate=args.learning_rate,
        opt_steps=args.opt_steps,
        selected_objectives=selected_objectives,
        use_aim=args.use_aim,
        oxdna_source_path=oxdna_source,
        metrics_file=Path(args.metrics_file) if args.metrics_file else None,
    )
