"""Stretch-torsion optimization for DNA mechanical properties (S_eff, C, g)."""

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
from mythos.observables.stretch_torsion import ExtensionZ, TwistXY, stretch_torsion
from mythos.simulators.base import SimulatorOutput
from mythos.simulators.lammps.lammps_oxdna import LAMMPSoxDNASimulator
from mythos.ui.loggers.aim import AimLogger
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

# Data directory for full reparameterization systems
DATA_ROOT = Path("data/full_reparam_oxdna1")
MECHANICAL_DIR = DATA_ROOT / "mechanical"
SYSTEM_DIR = MECHANICAL_DIR / "lammps" / "40bp_duplex"


@chex.dataclass(frozen=True, kw_only=True)
class LAMMPSStretchSimulator(LAMMPSoxDNASimulator):
    """LAMMPS simulator that tags trajectory states with force/torque metadata."""

    force: InitVar[float] = None
    torque: InitVar[float] = None

    def __post_init__(self, force: float, torque: float) -> None:
        """Store force/torque in variables dict for metadata tagging."""
        LAMMPSoxDNASimulator.__post_init__(self)
        object.__setattr__(self, "variables", {**self.variables, "force": force, "torque": torque})

    def run_simulation(self, *args, opt_params: Params, **kwargs) -> SimulatorOutput:
        """Run simulation and tag trajectory with force/torque metadata."""
        output = LAMMPSoxDNASimulator.run_simulation(self, *args, params=opt_params, **kwargs)
        tagged_traj = output.observables[0].with_state_metadata(self.variables.copy())
        return SimulatorOutput(observables=[tagged_traj], state=output.state)


def create_stretch_simulators(
    input_dir: Path,
    energy_fn: EnergyFunction,
    forces: list[float] | None = None,
    torque: float = STRETCH_TORQUE,
    variables: dict[str, Any] | None = None,
    n_replicas: int = 1,
    **kwargs: Any,
) -> list[LAMMPSStretchSimulator]:
    """Create n_replicas simulators per force value, all with the same held torque."""
    if forces is None:
        forces = STRETCH_FORCES
    if variables is None:
        variables = {}
    return [
        LAMMPSStretchSimulator(
            input_dir=str(input_dir),
            energy_fn=energy_fn,
            force=f,
            torque=torque,
            name=f"stretch_f{f}_t{torque}_r{r}",
            variables=variables.copy(),
            **kwargs,
        )
        for f in forces
        for r in range(n_replicas)
    ]


def create_twist_simulators(
    input_dir: Path,
    energy_fn: EnergyFunction,
    torques: list[float] | None = None,
    force: float = TWIST_FORCE,
    variables: dict[str, Any] | None = None,
    n_replicas: int = 1,
    **kwargs: Any,
) -> list[LAMMPSStretchSimulator]:
    """Create n_replicas simulators per torque value, all with the same held force."""
    if torques is None:
        torques = TWIST_TORQUES
    if variables is None:
        variables = {}
    return [
        LAMMPSStretchSimulator(
            input_dir=str(input_dir),
            energy_fn=energy_fn,
            force=force,
            torque=t,
            name=f"twist_f{force}_t{t}_r{r}",
            variables=variables.copy(),
            **kwargs,
        )
        for t in torques
        for r in range(n_replicas)
    ]


def read_box_size_lammps(data_path: Path) -> jnp.ndarray:
    """Parse box dimensions [x, y, z] from LAMMPS data file."""
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


def tree_mean(trees: list) -> Any:
    """Compute element-wise mean of a list of pytrees."""
    if len(trees) == 1:
        return trees[0]
    return jax.tree.map(lambda *x: jnp.mean(jnp.stack(x)), *trees)


def create_stretch_torsion_objectives(
    stretch_simulators: list[LAMMPSStretchSimulator],
    twist_simulators: list[LAMMPSStretchSimulator],
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
    """Create 3 DiffTRe objectives for S_eff (stretch), C (torsion), and g (coupling)."""
    transform_fn = energy_fn.energy_fns[0].transform_fn
    n_nucs_per_strand = top.n_nucleotides // 2

    # Define end base pairs for extension measurement
    bp1 = jnp.array([0, 2 * n_nucs_per_strand - 1], dtype=jnp.int32)
    bp2 = jnp.array([n_nucs_per_strand - 1, n_nucs_per_strand], dtype=jnp.int32)

    # Get quartets for twist measurement
    quartets = obs_base.get_duplex_quartets(n_nucs_per_strand)

    # Create observables
    extension_obs = ExtensionZ(
        rigid_body_transform_fn=transform_fn,
        bp1=bp1,
        bp2=bp2,
        displacement_fn=displacement_fn,
    )
    twist_obs = TwistXY(
        rigid_body_transform_fn=transform_fn,
        quartets=quartets,
        displacement_fn=displacement_fn,
    )

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
        """Compute (s_eff, c, g) moduli from trajectory with force/torque metadata."""
        # Pre-compute metadata arrays as constants
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
        """Compute stretch modulus loss."""
        s_eff, c, g = compute_moduli_from_traj(traj, weights)
        loss = jnp.sqrt((s_eff - target_stretch_modulus) ** 2)
        return loss, (("stretch_modulus", s_eff), {"torsional_modulus": c, "twist_stretch_coupling": g})

    def torsional_modulus_loss_fn(traj: Any, weights: jnp.ndarray, *_, **__) -> LossOutput:
        """Compute torsional modulus loss."""
        s_eff, c, g = compute_moduli_from_traj(traj, weights)
        loss = jnp.sqrt((c - target_torsion_modulus) ** 2)
        return loss, (("torsional_modulus", c), {"stretch_modulus": s_eff, "twist_stretch_coupling": g})

    def twist_stretch_coupling_loss_fn(traj: Any, weights: jnp.ndarray, *_, **__) -> LossOutput:
        """Compute twist-stretch coupling loss."""
        s_eff, c, g = compute_moduli_from_traj(traj, weights)
        # Use absolute target since g can be negative
        loss = jnp.sqrt((g - target_twist_stretch_coupling) ** 2)
        return loss, (("twist_stretch_coupling", g), {"stretch_modulus": s_eff, "torsional_modulus": c})

    # Collect required observables from all simulators
    all_simulators = stretch_simulators + twist_simulators
    required_observables = [obs for sim in all_simulators for obs in sim.exposes()]

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


# Valid objective names for command-line selection
VALID_OBJECTIVES = {"stretch_modulus", "torsional_modulus", "twist_stretch_coupling"}


def run_optimization(
    input_dir: Path = SYSTEM_DIR,
    learning_rate: float = 5e-4,
    opt_steps: int = 100,
    selected_objectives: list[str] | None = None,
) -> Params:
    """Run DiffTRe optimization targeting S_eff=1000pN, C=460pN·nm², g=-90pN·nm."""
    import ray
    from mythos.utils.units import get_kt_from_string
    from tqdm import tqdm

    # Load topology and create energy function
    top = jdna_top.from_oxdna_file(input_dir / "data.top")
    box = read_box_size_lammps(input_dir / "data")
    kt = get_kt_from_string("300K")
    displacement_fn = jax_md.space.periodic(box)[0]

    # Create base energy function
    base_energy_fn = dna1_energy.create_default_energy_fn(
        topology=top,
        displacement_fn=displacement_fn,
    ).with_noopt(
        "ss_stack_weights", "ss_hb_weights"
    ).without_terms(
        "BondedExcludedVolume"  # LAMMPS doesn't implement this term
    ).with_params(kt=kt)

    opt_params = base_energy_fn.opt_params()

    # Create simulators for stretch experiments (vary force, hold torque)
    stretch_simulators = create_stretch_simulators(
        input_dir=input_dir,
        energy_fn=base_energy_fn,
        input_file_name="in",
        variables={"T": kt, "nsteps": 1_250_000},
    )

    # Create simulators for twist experiments (vary torque, hold force)
    twist_simulators = create_twist_simulators(
        input_dir=input_dir,
        energy_fn=base_energy_fn,
        input_file_name="in",
        variables={"T": kt, "nsteps": 1_250_000},
    )

    all_simulators = stretch_simulators + twist_simulators

    # Create the stretch-torsion objectives (S_eff, C, g)
    objectives = create_stretch_torsion_objectives(
        stretch_simulators=stretch_simulators,
        twist_simulators=twist_simulators,
        energy_fn=base_energy_fn,
        top=top,
        displacement_fn=displacement_fn,
        kt=kt,
    )

    # Filter objectives if a subset was specified
    if selected_objectives:
        objective_map = {obj.name: obj for obj in objectives}
        objectives = [objective_map[name] for name in selected_objectives]
        logging.info("Using selected objectives: %s", [obj.name for obj in objectives])

    # Setup Ray optimizer
    optimizer = jdna_optimization.RayOptimizer(
        objectives=objectives,
        simulators=all_simulators,
        aggregate_grad_fn=tree_mean,
        optimizer=optax.adam(learning_rate=learning_rate),
    )

    # Run optimization loop
    logging.info(
        "Starting stretch-torsion optimization: targets S_eff=%.1f pN, C=%.1f pN·nm², g=%.1f pN·nm",
        TARGET_STRETCH_MODULUS,
        TARGET_TORSION_MODULUS,
        TARGET_TWIST_STRETCH_COUPLING,
    )

    aim_logger = AimLogger(experiment="stretch_twist_optimization")

    state = None
    for step in tqdm(range(opt_steps), desc="Optimizing mechanical properties"):
        output = optimizer.step(opt_params, state)
        opt_params = output.opt_params
        state = output.state

        # Log metrics
        for ctx, obs in output.observables.items():
            for name in obs:
                logging.info("Step %d: %s.%s = %s", step, ctx, name, obs[name])
                aim_logger.log_metric(f"{ctx}.{name}", obs[name], step=step)

    logging.info("Optimization complete!")
    logging.info("Final parameters: %s", opt_params)

    ray.shutdown()
    return opt_params


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stretch-torsion optimization for DNA mechanical properties (S_eff, C, g)."
    )
    parser.add_argument(
        "--objectives",
        type=str,
        default=None,
        help=(
            "Comma-separated list of objectives to optimize. "
            f"Valid options: {', '.join(sorted(VALID_OBJECTIVES))}. "
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse and validate objectives
    selected_objectives = None
    if args.objectives:
        selected_objectives = [obj.strip() for obj in args.objectives.split(",")]
        invalid = set(selected_objectives) - VALID_OBJECTIVES
        if invalid:
            raise ValueError(
                f"Invalid objective(s): {invalid}. "
                f"Valid options are: {VALID_OBJECTIVES}"
            )

    run_optimization(
        learning_rate=args.learning_rate,
        opt_steps=args.opt_steps,
        selected_objectives=selected_objectives,
    )
