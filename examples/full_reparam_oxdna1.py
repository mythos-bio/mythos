"""Full reparameterization of oxDNA1"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import InitVar
from pathlib import Path
from typing import Any

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
from mythos.energy.base import EnergyFunction
from mythos.observables.diameter import Diameter
from mythos.observables.persistence_length import PersistenceLength
from mythos.observables.pitch import PitchAngle, compute_pitch
from mythos.observables.propeller import PropellerTwist
from mythos.observables.rise import Rise
from mythos.observables.stretch_torsion import ExtensionZ, TwistXY, stretch_torsion
from mythos.simulators.base import SimulatorOutput
from mythos.simulators.lammps.lammps_oxdna import LAMMPSoxDNASimulator
from mythos.simulators.oxdna import oxDNASimulator
from mythos.ui.loggers.console import ConsoleLogger
from mythos.ui.loggers.disk import FileLogger
from mythos.ui.loggers.multilogger import MultiLogger
from mythos.utils.types import Params

jax.config.update("jax_enable_x64", val=True)
logging.basicConfig(level=logging.INFO)

# Type alias for DiffTRe loss function return type
LossOutput = tuple[float, tuple[tuple[str, float], dict]]

STRETCH_FORCES = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40]  # pN
STRETCH_TORQUE = 0  # pN·nm
TARGET_STRETCH_MODULUS = 1000.0  # pN
TWIST_TORQUES = [5, 10, 15, 20, 25, 30]
TWIST_FORCE = 2  # pN
TARGET_TORSION_MODULUS = 460  # pN·nm^2
TARGET_TWIST_STRETCH_COUPLING = -90  # pN·nm

# Structural property targets (20bp duplex at 300K)
TARGET_HELIX_DIAMETER = 20.0  # Angstroms (experimental: ~20 Å)
TARGET_HELICAL_PITCH = 3.57  # nm (experimental: 3.4-3.6 nm)
TARGET_PROPELLER_TWIST = -12.6  # degrees
TARGET_RISE = 0.34  # nm (3.4 Å per base pair)
SIGMA_BACKBONE = 0.70  # oxDNA1 default excluded volume sigma for backbone

# Persistence length target (60bp duplex)
TARGET_PERSISTENCE_LENGTH = 50.0  # nm

# Data directory for full reparameterization systems
DATA_ROOT = Path("data/full_reparam_oxdna1")
MECHANICAL_DIR = DATA_ROOT / "mechanical"
MECHANICAL_SYSTEM_DIR = MECHANICAL_DIR / "lammps" / "40bp_duplex"
PERSISTENCE_SYSTEM_DIR = MECHANICAL_DIR / "60bp_duplex"
STRUCTURAL_DIR = DATA_ROOT / "structural"
STRUCTURAL_SYSTEM_DIR = STRUCTURAL_DIR / "20bp_duplex"


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
    stretch_forces: list[float] | None = None,
    stretch_torque: float = STRETCH_TORQUE,
    twist_torques: list[float] | None = None,
    twist_force: float = TWIST_FORCE,
    variables: dict[str, Any] | None = None,
    n_replicas: int = 1,
    **kwargs: Any,
) -> list[LAMMPSStretchSimulator]:
    stretch_forces = stretch_forces or STRETCH_FORCES
    twist_torques = twist_torques or TWIST_TORQUES
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


def create_stretch_torsion_objectives(
    simulators: list[LAMMPSStretchSimulator],
    energy_fn: EnergyFunction,
    top: jdna_top.Topology,
    displacement_fn: Callable,
    kt: float,
    stretch_torque: float = STRETCH_TORQUE,
    twist_force: float = TWIST_FORCE,
    target_stretch_modulus: float = TARGET_STRETCH_MODULUS,
    target_torsion_modulus: float = TARGET_TORSION_MODULUS,
    target_twist_stretch_coupling: float = TARGET_TWIST_STRETCH_COUPLING,
) -> list[jdna_objective.DiffTReObjective]:
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

    # Conversion factor from simulation units to nm (oxDNA length unit is 0.8518 nm)
    length_conversion = 0.8518

    # Precompute force/torque values (constants for differentiation)
    stretch_forces = jnp.array(STRETCH_FORCES, dtype=jnp.float64)
    twist_torques = jnp.array(TWIST_TORQUES, dtype=jnp.float64)
    n_force_segments = len(STRETCH_FORCES)
    n_torque_segments = len(TWIST_TORQUES)

    def compute_moduli_from_traj(
        traj: Any, weights: jnp.ndarray
    ) -> tuple[float, float, float]:
        forces_arr = jnp.array([md["force"] for md in traj.metadata])
        torques_arr = jnp.array([md["torque"] for md in traj.metadata])

        # Compute segment indices using searchsorted (forces/torques are sorted)
        force_segment_ids = jnp.searchsorted(stretch_forces, forces_arr).astype(jnp.int32)
        torque_segment_ids = jnp.searchsorted(twist_torques, torques_arr).astype(jnp.int32)

        # Create boolean masks for filtering (constants, not differentiable)
        stretch_mask = torques_arr == stretch_torque  # held torque for stretch experiments
        twist_mask = forces_arr == twist_force  # held force for twist experiments

        # Compute observables for all states once
        all_extensions = extension_obs(traj) * length_conversion
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
    n_steps: int = 1_000_000,
    snapshot_interval: int = 10_000,
    n_replicas: int = 1,
    **kwargs: Any,
) -> list[oxDNASimulator]:
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
    target_helix_diameter: float = TARGET_HELIX_DIAMETER,
    target_helical_pitch: float = TARGET_HELICAL_PITCH,
    target_propeller_twist: float = TARGET_PROPELLER_TWIST,
    target_rise: float = TARGET_RISE,
    sigma_backbone: float = SIGMA_BACKBONE,
) -> list[jdna_objective.DiffTReObjective]:
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
    n_steps: int = 1_000_000,
    snapshot_interval: int = 10_000,
    n_replicas: int = 1,
    **kwargs: Any,
) -> list[oxDNASimulator]:
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
    target_persistence_length: float = TARGET_PERSISTENCE_LENGTH,
    n_equilibration_steps: int = 20,
) -> jdna_objective.DiffTReObjective:
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
        lp = persistence_obs(traj, weights)  # Returns nm
        loss = jnp.sqrt((lp - target_persistence_length) ** 2)
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


# Valid objective names for command-line selection
MECHANICAL_OBJECTIVES = {"stretch_modulus", "torsional_modulus", "twist_stretch_coupling"}
PERSISTENCE_OBJECTIVES = {"persistence_length"}
STRUCTURAL_OBJECTIVES = {"helix_diameter", "helical_pitch", "propeller_twist", "rise"}
VALID_OBJECTIVES = MECHANICAL_OBJECTIVES | PERSISTENCE_OBJECTIVES | STRUCTURAL_OBJECTIVES


def run_optimization(
    mechanical_input_dir: Path = MECHANICAL_SYSTEM_DIR,
    structural_input_dir: Path = STRUCTURAL_SYSTEM_DIR,
    persistence_input_dir: Path = PERSISTENCE_SYSTEM_DIR,
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

    kt = get_kt_from_string("300K")

    # Determine which objective categories are needed
    need_mechanical = bool(selected_objectives & MECHANICAL_OBJECTIVES)
    need_persistence = bool(selected_objectives & PERSISTENCE_OBJECTIVES)
    need_structural = bool(selected_objectives & STRUCTURAL_OBJECTIVES)

    all_simulators = []
    all_objectives = []
    opt_params = None

    # Setup mechanical objectives (LAMMPS simulators, 40bp duplex)
    if need_mechanical:
        mech_top = jdna_top.from_oxdna_file(mechanical_input_dir / "data.top")
        mech_box = read_box_size_lammps(mechanical_input_dir / "data")
        mech_displacement_fn = jax_md.space.periodic(mech_box)[0]

        mech_energy_fn = dna1_energy.create_default_energy_fn(
            topology=mech_top,
            displacement_fn=mech_displacement_fn,
        ).with_noopt(
            "ss_stack_weights", "ss_hb_weights"
        ).without_terms(
            "BondedExcludedVolume"  # LAMMPS doesn't implement this term
        ).with_params(kt=kt)

        opt_params = mech_energy_fn.opt_params()

        stretch_twist_simulators = create_stretch_twist_simulators(
            input_dir=mechanical_input_dir,
            energy_fn=mech_energy_fn,
            input_file_name="in",
            variables={"T": kt, "nsteps": 1_250_000},
        )

        mechanical_objectives = create_stretch_torsion_objectives(
            simulators=stretch_twist_simulators,
            energy_fn=mech_energy_fn,
            top=mech_top,
            displacement_fn=mech_displacement_fn,
            kt=kt,
        )

        all_simulators.extend(stretch_twist_simulators)
        all_objectives.extend(mechanical_objectives)

    # Setup structural objectives (oxDNA MD simulator, 20bp duplex)
    if need_structural:
        struct_top = jdna_top.from_oxdna_file(structural_input_dir / "sys.top")
        struct_box = read_box_size_oxdna(structural_input_dir / "sys.conf")
        struct_displacement_fn = jax_md.space.periodic(struct_box)[0]

        struct_energy_fn = dna1_energy.create_default_energy_fn(
            topology=struct_top,
            displacement_fn=struct_displacement_fn,
        ).with_noopt(
            "ss_stack_weights", "ss_hb_weights"
        ).with_params(kt=kt)

        if opt_params is None:
            opt_params = struct_energy_fn.opt_params()

        structural_simulators = create_structural_simulators(
            input_dir=structural_input_dir,
            energy_fn=struct_energy_fn,
            source_path=oxdna_source_path,
            n_steps=1_000_000,
            n_replicas=1,
        )

        structural_objectives = create_structural_objectives(
            simulators=structural_simulators,
            energy_fn=struct_energy_fn,
            top=struct_top,
            displacement_fn=struct_displacement_fn,
            kt=kt,
        )

        all_simulators.extend(structural_simulators)
        all_objectives.extend(structural_objectives)

    # Setup persistence length objective (oxDNA MD simulator, 60bp duplex)
    if need_persistence:
        persist_top = jdna_top.from_oxdna_file(persistence_input_dir / "sys.top")
        persist_box = read_box_size_oxdna(persistence_input_dir / "sys.conf")
        persist_displacement_fn = jax_md.space.periodic(persist_box)[0]

        persist_energy_fn = dna1_energy.create_default_energy_fn(
            topology=persist_top,
            displacement_fn=persist_displacement_fn,
        ).with_noopt(
            "ss_stack_weights", "ss_hb_weights"
        ).with_params(kt=kt)

        if opt_params is None:
            opt_params = persist_energy_fn.opt_params()

        persistence_simulators = create_persistence_simulators(
            input_dir=persistence_input_dir,
            energy_fn=persist_energy_fn,
            source_path=oxdna_source_path,
            n_steps=1_000_000,
            n_replicas=1,
        )

        persistence_objective = create_persistence_objective(
            simulators=persistence_simulators,
            energy_fn=persist_energy_fn,
            top=persist_top,
            displacement_fn=persist_displacement_fn,
            kt=kt,
        )

        all_simulators.extend(persistence_simulators)
        all_objectives.append(persistence_objective)

    # Filter objectives to only those selected
    objective_map = {obj.name: obj for obj in all_objectives}
    all_objectives = [objective_map[name] for name in selected_objectives if name in objective_map]
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
        logging.info(
            "  Mechanical targets: S_eff=%.1f pN, C=%.1f pN·nm², g=%.1f pN·nm",
            TARGET_STRETCH_MODULUS,
            TARGET_TORSION_MODULUS,
            TARGET_TWIST_STRETCH_COUPLING,
        )
    if need_structural:
        logging.info(
            "  Structural targets: diameter=%.1f Å, pitch=%.2f nm, propeller=%.1f°, rise=%.2f nm",
            TARGET_HELIX_DIAMETER,
            TARGET_HELICAL_PITCH,
            TARGET_PROPELLER_TWIST,
            TARGET_RISE,
        )
    if need_persistence:
        logging.info("  Persistence length target: Lp=%.1f nm", TARGET_PERSISTENCE_LENGTH)

    loggers = [ConsoleLogger()]
    if use_aim:
        from mythos.ui.loggers.aim import AimLogger
        name = f"full_reparam_oxdna1-{len(all_objectives)}obj"
        if len(all_objectives) == 1:
            name = f"oxdna1-{all_objectives[0].name}"
        aim_logger = AimLogger(experiment=name)
        loggers.append(aim_logger)
    if metrics_file:
        loggers.append(FileLogger(metrics_file))
    logger = MultiLogger(loggers)

    state = None
    for step in tqdm(range(opt_steps), desc="Optimizing oxDNA1 parameters"):
        output = optimizer.step(opt_params, state)
        opt_params = output.opt_params
        state = output.state

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
            "structural (diameter, pitch, propeller, rise), and persistence length properties."
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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
    need_oxdna = bool(selected_objectives & (STRUCTURAL_OBJECTIVES | PERSISTENCE_OBJECTIVES))
    if need_oxdna and oxdna_source is None:
        raise ValueError("--oxdna-source is required when using oxDNA based objectives")

    run_optimization(
        learning_rate=args.learning_rate,
        opt_steps=args.opt_steps,
        selected_objectives=selected_objectives,
        use_aim=args.use_aim,
        oxdna_source_path=oxdna_source,
        metrics_file=Path(args.metrics_file) if args.metrics_file else None,
    )
