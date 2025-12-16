"""Multi-simulation, multi-objective oxDNA1 full reparameterization script.

This script performs a full reparameterization of the oxDNA1 model using:
- Structural objectives: helix diameter, helical pitch, propeller twist, and rise
- Mechanical objectives: torsional modulus, stretch modulus, and twist-stretch coupling
- Thermodynamic objectives: melting temperature for various DNA systems

All objectives use DiffTRe for gradient estimation. The script uses Ray simulators
and optimizers to distribute workloads. Thermodynamic systems use the multigang
simulator to update umbrella weights as a gang using last_hist information.

Usage:
    python examples/advanced_optimizations/oxdna1_full_reparameterization.py \
        --num_sims=10 --learning_rate=1e-3 --opt_steps=100

Important: Run from the root directory of the repository.
"""

import logging
import typing
from dataclasses import InitVar
from pathlib import Path
from unittest.mock import MagicMock

import chex
import fire
import jax
import jax.numpy as jnp
import jax_md
import mythos.energy.dna1 as dna1_energy
import mythos.input.topology as jdna_top
import mythos.optimization.objective as jdna_objective
import mythos.optimization.optimization as jdna_optimization
import mythos.utils.types as jdna_types
import optax
import pandas as pd
import ray
from mythos.energy.base import EnergyFunction
from mythos.input import oxdna_input
from mythos.observables import base as obs_base
from mythos.observables.diameter import Diameter
from mythos.observables.melting_temp import MeltingTemp
from mythos.observables.persistence_length import PersistenceLength
from mythos.observables.pitch import PitchAngle, compute_pitch
from mythos.observables.propeller import PropellerTwist
from mythos.observables.rise import Rise
from mythos.observables.stretch_torsion import ExtensionZ, TwistXY, stretch_torsion
from mythos.simulators.lammps.lammps_oxdna import LAMMPSoxDNASimulator
from mythos.simulators.oxdna import oxdna
from mythos.simulators.oxdna.oxdna import oxDNASimulator
from mythos.simulators.oxdna.utils import read_energy
from mythos.simulators.ray import RayMultiGangSimulation, RayMultiSimulation, RaySimulation
from mythos.ui.loggers.console import ConsoleLogger
from mythos.ui.loggers.logger import NullLogger
from mythos.ui.loggers.multilogger import MultiLogger
from mythos.utils.units import get_kt, get_kt_from_string
from tqdm import tqdm

jax.config.update("jax_enable_x64", val=True)
logging.basicConfig(level=logging.INFO)
logging.getLogger("jax").setLevel(logging.WARNING)

# =============================================================================
# Data directories
# =============================================================================
DATA_ROOT = Path("data/full_reparam_oxdna1")
STRUCTURAL_DIR = DATA_ROOT / "structural"
MECHANICAL_DIR = DATA_ROOT / "mechanical"
THERMODYNAMIC_DIR = DATA_ROOT / "thermodynamic"

# System definitions
STRUCTURAL_SYSTEMS = ["20bp_duplex", "20bp_duplex_nicked"]
MECHANICAL_OXDNA_SYSTEMS = ["60bp_duplex"]
MECHANICAL_LAMMPS_SYSTEMS = ["lammps/40bp_duplex"]
THERMODYNAMIC_SYSTEMS = [
    "5bp_duplex",
    "5bp_duplex_dangling_end",
    "5bp_duplex_terminal_mismatch",
    "5bp_stem_5bp_loop_hairpin",
    "8bp_duplex",
    "8bp_duplex_1bp_bulge",
    "8bp_duplex_2bp_bubble",
]

STRUCTURAL_SYSTEMS = []
#MECHANICAL_OXDNA_SYSTEMS = []
#MECHANICAL_LAMMPS_SYSTEMS = []
THERMODYNAMIC_SYSTEMS = []


# =============================================================================
# Target values for structural observables (experimental/literature values)
# =============================================================================
TARGET_HELIX_DIAMETER = 23.0  # Angstroms
TARGET_HELICAL_PITCH = 10.5   # bp/turn
TARGET_PROPELLER_TWIST = 21.7 # degrees
TARGET_RISE = 3.4             # Angstroms

# Target values for mechanical observables
TARGET_PERSISTENCE_LENGTH = 47.5  # nm (from PersistenceLength.TARGETS["oxDNA"])
TARGET_STRETCH_MODULUS = 1100.0   # pN
TARGET_TORSIONAL_MODULUS = 440.0  # pN·nm²
TARGET_TWIST_STRETCH_COUPLING = -17.0  # pN·nm

# Target melting temperatures (in simulation units, derived from Celsius)
TARGET_MELTING_TEMPS = {
    "5bp_duplex": get_kt_from_string("31.2C"),
    "5bp_duplex_dangling_end": get_kt_from_string("35.0C"),
    "5bp_duplex_terminal_mismatch": get_kt_from_string("33.0C"),
    "5bp_stem_5bp_loop_hairpin": get_kt_from_string("58.0C"),
    "8bp_duplex": get_kt_from_string("48.2C"),
    "8bp_duplex_1bp_bulge": get_kt_from_string("42.0C"),
    "8bp_duplex_2bp_bubble": get_kt_from_string("38.0C"),
}

# =============================================================================
# Force/Torque sweep for mechanical properties
# =============================================================================
# For stretch experiments: vary force at zero torque
STRETCH_FORCES = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40]  # pN
STRETCH_TORQUE = 0  # pN·nm

# For torsion experiments: vary torque at constant force
TORSION_FORCE = 2  # pN (small constant force to keep duplex extended)
TORSION_TORQUES = [5, 10, 15, 20, 25, 30]  # pN·nm

# Combined list of (force, torque) tuples for all mechanical simulations
MECHANICAL_FORCE_TORQUE_SWEEP = (
    [(f, STRETCH_TORQUE) for f in STRETCH_FORCES] +  # 12 stretch simulations
    [(TORSION_FORCE, t) for t in TORSION_TORQUES]     # 6 torsion simulations
)  # Total: 18 simulations


# =============================================================================
# Helper classes for observable tagging
# =============================================================================
class EnergyInfo(pd.DataFrame):
    """Tagged DataFrame for energy information from umbrella sampling."""
    pass


class HistogramInfo(pd.DataFrame):
    """Tagged DataFrame for histogram information from umbrella sampling."""
    pass


# =============================================================================
# Custom simulators
# =============================================================================
@chex.dataclass(kw_only=True, eq=False)
class oxDNAUmbrellaSampler(oxdna.oxDNASimulator):
    """oxDNA simulator with umbrella sampling support for melting temperature calculations."""

    exposed_observables: typing.ClassVar[list[str]] = ["trajectory", "energy_info", "histogram_info"]

    def run(self, params: jdna_types.Params, meta_data: typing.Any = None) -> tuple:
        traj = oxdna.oxDNASimulator.run(self, params, meta_data)
        energy_df = EnergyInfo(read_energy(self.base_dir))
        hist_df = HistogramInfo(self.get_hist())
        return traj, energy_df, hist_df

    def update_weights(self, weights: pd.DataFrame) -> None:
        """Update umbrella sampling weights from histogram data."""
        weights_file = self.base_dir / self.input_config["weights_file"]
        weights.to_csv(weights_file, sep=" ", header=False)

    def get_hist(self) -> pd.DataFrame:
        """Read the last histogram file from the simulation."""
        hist_file = self.base_dir / self.input_config["last_hist_file"]
        hist_df_columns = ["bind", "mindist", "unbiased"]
        hist_df = pd.read_csv(
            hist_file, names=hist_df_columns, sep=r"\s+", usecols=[0, 1, 3], skiprows=1
        ).set_index(["bind", "mindist"])
        hist_df["unbiased_normed"] = hist_df["unbiased"] / hist_df["unbiased"].sum()
        return hist_df


class oxDNAUmbrellaSamplerGang(RayMultiGangSimulation):
    """Gang simulator for umbrella sampling that updates weights after each run."""

    def pre_run(self, *args, **kwargs) -> None:
        """Hook before all simulations start."""
        pass

    def post_run(self, observables: list[typing.Any], *args, **kwargs) -> list[typing.Any]:
        """Update umbrella weights from combined histograms after all simulations complete."""
        # Combine histogram data from all simulations
        print("OBSERVABLES IN OXDNAGANG post_run: ", observables)
        hist = pd.concat([i for i in observables if isinstance(i, HistogramInfo)])
        hist = hist.reset_index().groupby(["bind", "mindist"]).sum()

        # Compute new weights from combined histogram
        weights = hist.query("unbiased_normed > 0").eval("weights = 1 / unbiased_normed")
        weights["weights"] /= weights["weights"].min()  # for numerical stability
        weights = weights[["weights"]]
        # Fill in zeroed states
        weights = weights.reindex(hist.index, fill_value=0)

        # Update weights in all simulators
        ray.get([
            simulator.call_async("update_weights", weights)
            for simulator in self.simulations
        ])
        return observables


# =============================================================================
# Helper functions
# =============================================================================
def read_box_size(input_dir: Path, sim_config: dict) -> jnp.ndarray:
    """Read box size from configuration file."""
    with input_dir.joinpath(sim_config["conf_file"]).open("r") as f:
        for line in f:
            if line.startswith("b ="):
                box_size = line.split("=")[1].strip().split()
                return jnp.array([float(i) for i in box_size])
    raise ValueError(f"Could not find box size in {input_dir / sim_config['conf_file']}")


def read_box_size_lammps(data_path: Path) -> jnp.ndarray:
    box = [-1, -1, -1]
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


@chex.dataclass(frozen=True)
class OxDNASystemConfig:
    """Configuration bundle for an oxDNA system."""

    energy_fn: EnergyFunction
    topology: jdna_top.Topology
    displacement_fn: typing.Callable
    kT: float
    sim_config: dict


def create_oxdna_system_config(
    base_energy_fn: EnergyFunction,
    system_dir: Path,
    box_size: jnp.ndarray | None = None,
) -> OxDNASystemConfig:
    """Create a complete configuration bundle for an oxDNA system.

    Args:
        base_energy_fn: Base energy function to configure for this system.
        system_dir: Path to the system directory containing input and topology files.
        box_size: Optional box size. If None, will be read from the configuration file.

    Returns:
        OxDNASystemConfig containing energy_fn, topology, displacement_fn, kT, and sim_config.
    """
    sim_config = oxdna_input.read(system_dir / "input")
    top = jdna_top.from_oxdna_file(next(system_dir.glob("*.top")))

    if box_size is None:
        box_size = read_box_size(system_dir, sim_config)
    displacement_fn = jax_md.space.periodic(box_size)[0]

    kT = get_kt_from_string(sim_config["T"])
    energy_fn = base_energy_fn.with_props(
        topology=top,
        displacement_fn=displacement_fn,
    ).with_params(kt=kT)

    return OxDNASystemConfig(
        energy_fn=energy_fn,
        topology=top,
        displacement_fn=displacement_fn,
        kT=kT,
        sim_config=sim_config,
    )


def get_h_bonded_pairs(n_nucs_per_strand: int) -> jnp.ndarray:
    """Get hydrogen-bonded base pairs for a duplex."""
    s1 = list(range(n_nucs_per_strand))
    s2 = list(range(n_nucs_per_strand, 2 * n_nucs_per_strand))
    s2.reverse()
    return jnp.array(list(zip(s1, s2)), dtype=jnp.int32)


# =============================================================================
# Objective creation functions
# =============================================================================
def create_structural_objectives(
    simulator: RayMultiSimulation,
    energy_fn: EnergyFunction,
    top: jdna_top.Topology,
    displacement_fn: typing.Callable,
    opt_params: jdna_types.Params,
    kT: float,
) -> list[jdna_objective.DiffTReObjective]:
    """Create structural objectives (diameter, pitch, propeller twist, rise)."""
    transform_fn = energy_fn.energy_fns[0].transform_fn
    n_nucs_per_strand = top.n_nucleotides // 2
    h_bonded_pairs = get_h_bonded_pairs(n_nucs_per_strand)
    quartets = obs_base.get_duplex_quartets(n_nucs_per_strand)

    # Observable instances
    diameter_obs = Diameter(
        rigid_body_transform_fn=transform_fn,
        h_bonded_base_pairs=h_bonded_pairs,
        displacement_fn=displacement_fn,
    )
    pitch_obs = PitchAngle(
        rigid_body_transform_fn=transform_fn,
        quartets=quartets,
        displacement_fn=displacement_fn,
    )
    propeller_obs = PropellerTwist(
        rigid_body_transform_fn=transform_fn,
        h_bonded_base_pairs=h_bonded_pairs,
    )
    rise_obs = Rise(
        rigid_body_transform_fn=transform_fn,
        quartets=quartets,
        displacement_fn=displacement_fn,
    )

    # Get sigma_backbone from energy function params
    sigma_backbone = energy_fn.params_dict().get("sigma_backbone", 0.7)

    # Loss functions
    def diameter_loss_fn(traj, weights, _energy_fn, _opt_params, _observables):
        obs_values = diameter_obs(traj, sigma_backbone)
        expected = jnp.dot(weights, obs_values)
        loss = jnp.sqrt((expected - TARGET_HELIX_DIAMETER) ** 2)
        return loss, (("helix_diameter", expected), {})

    def pitch_loss_fn(traj, weights, _energy_fn, _opt_params, _observables):
        obs_values = pitch_obs(traj)
        expected_angle = jnp.dot(weights, obs_values)
        expected_pitch = compute_pitch(expected_angle)
        loss = jnp.sqrt((expected_pitch - TARGET_HELICAL_PITCH) ** 2)
        return loss, (("helical_pitch", expected_pitch), {})

    def propeller_loss_fn(traj, weights, _energy_fn, _opt_params, _observables):
        obs_values = propeller_obs(traj)
        expected = jnp.dot(weights, obs_values)
        loss = jnp.sqrt((expected - TARGET_PROPELLER_TWIST) ** 2)
        return loss, (("propeller_twist", expected), {})

    def rise_loss_fn(traj, weights, _energy_fn, _opt_params, _observables):
        obs_values = rise_obs(traj)
        expected = jnp.dot(weights, obs_values)
        loss = jnp.sqrt((expected - TARGET_RISE) ** 2)
        return loss, (("rise", expected), {})

    beta = jnp.array(1 / kT, dtype=jnp.float64)
    common_kwargs = {
        "required_observables": simulator.exposes(),
        "energy_fn": energy_fn,
        "opt_params": opt_params,
        "beta": beta,
        "n_equilibration_steps": 0,
        "min_n_eff_factor": 0.95,
    }

    return [
        jdna_objective.RayObjective(
            jdna_objective.DiffTReObjective,
            name="helix_diameter",
            logging_observables=["loss", "helix_diameter", "neff"],
            grad_or_loss_fn=diameter_loss_fn,
            **common_kwargs,
        ),
        jdna_objective.RayObjective(
            jdna_objective.DiffTReObjective,
            name="helical_pitch",
            logging_observables=["loss", "helical_pitch", "neff"],
            grad_or_loss_fn=pitch_loss_fn,
            **common_kwargs,
        ),
        jdna_objective.RayObjective(
            jdna_objective.DiffTReObjective,
            name="propeller_twist",
            logging_observables=["loss", "propeller_twist", "neff"],
            grad_or_loss_fn=propeller_loss_fn,
            **common_kwargs,
        ),
        jdna_objective.RayObjective(
            jdna_objective.DiffTReObjective,
            name="rise",
            logging_observables=["loss", "rise", "neff"],
            grad_or_loss_fn=rise_loss_fn,
            **common_kwargs,
        ),
    ]


# =============================================================================
# oxDNA Mechanical objective (persistence length)
# =============================================================================

def create_oxdna_mechanical_objective(
    simulator: RayMultiSimulation,
    energy_fn: EnergyFunction,
    top: jdna_top.Topology,
    displacement_fn: typing.Callable,
    opt_params: jdna_types.Params,
    kT: float,
    target_lp: float = TARGET_PERSISTENCE_LENGTH,
) -> jdna_objective.DiffTReObjective:
    """Create a persistence length objective for oxDNA mechanical simulation.

    This objective uses the persistence length observable to measure the
    mechanical stiffness of the DNA duplex. It is appropriate for oxDNA
    simulations where we can directly measure the bending of the helix.

    Args:
        simulator: The oxDNA simulator to use.
        energy_fn: Energy function for the simulation.
        top: Topology of the system.
        displacement_fn: Displacement function for the system.
        opt_params: Optimization parameters.
        kT: Temperature in simulation units.
        target_lp: Target persistence length in nm (default: 47.5 nm).

    Returns:
        DiffTReObjective for persistence length optimization.
    """
    transform_fn = energy_fn.energy_fns[0].transform_fn
    n_nucs_per_strand = top.n_nucleotides // 2
    quartets = obs_base.get_duplex_quartets(n_nucs_per_strand)

    lp_obs = PersistenceLength(
        rigid_body_transform_fn=transform_fn,
        displacement_fn=displacement_fn,
        quartets=quartets,
        truncate=40,  # Standard truncation for Lp fitting
    )

    def persistence_length_loss_fn(traj, weights, _energy_fn, _opt_params, _observables):
        """Compute persistence length loss using weighted trajectory."""
        fit_lp = lp_obs(traj, weights=weights)
        # Use relative squared error for consistent scaling
        loss = ((fit_lp - target_lp) / target_lp) ** 2
        return loss, (("persistence_length", fit_lp), {})

    beta = jnp.array(1 / kT, dtype=jnp.float64)

    return jdna_objective.RayObjective(
        jdna_objective.DiffTReObjective,
        name="persistence_length",
        required_observables=simulator.exposes(),
        logging_observables=["loss", "persistence_length", "neff"],
        grad_or_loss_fn=persistence_length_loss_fn,
        energy_fn=energy_fn,
        opt_params=opt_params,
        beta=beta,
        n_equilibration_steps=0,
        min_n_eff_factor=0.95,
    )


# =============================================================================
# LAMMPS Mechanical simulation with trajectory metadata
# =============================================================================

@chex.dataclass
class LAMMPSMechanicalSimulator(LAMMPSoxDNASimulator):
    """LAMMPS simulator that adds force/torque metadata to trajectory states.

    This extends LAMMPSoxDNASimulator to tag each state in the trajectory with
    the force and torque conditions used for that simulation. The metadata
    can be used with SimulatorTrajectory.filter() to select states by condition.
    """
    force: InitVar[float] = None
    torque: InitVar[float] = None

    def __post_init__(self, force: float, torque: float):
        """Initialize the simulator with force/torque conditions.

        Args:
            force: Applied force in pN.
            torque: Applied torque in pN·nm.
        """
        LAMMPSoxDNASimulator.__post_init__(self)
        self.variables.update({"force": force, "torque": torque})

    def run(self, params: jdna_types.Params, meta_data: typing.Any = None) -> tuple:
        """Run simulation and return trajectory with force/torque metadata."""
        traj = LAMMPSoxDNASimulator.run(self, params, meta_data)
        return traj.with_state_metadata(self.variables)


def create_lammps_mechanical_objectives(
    simulator: RayMultiSimulation,
    energy_fn: EnergyFunction,
    top: jdna_top.Topology,
    displacement_fn: typing.Callable,
    opt_params: jdna_types.Params,
    kT: float,
) -> list[jdna_objective.DiffTReObjective]:
    """Create 3 DiffTRe objectives for LAMMPS mechanical properties.

    Creates separate objectives for:
    1. Stretch modulus (S_eff) - from stretch_torsion
    2. Torsional modulus (C) - from stretch_torsion
    3. Twist-stretch coupling (g) - from stretch_torsion

    The loss functions use SimulatorTrajectory.filter() to select states
    by their force/torque metadata, making the slicing logic simple.

    Args:
        simulators: List of LAMMPSMechanicalSimulator instances (one per condition).
        energy_fn: Energy function for the simulation.
        top: Topology of the system.
        displacement_fn: Displacement function for the system.
        opt_params: Optimization parameters.
        kT: Temperature in simulation units.

    Returns:
        List of 3 DiffTReObjective instances: [stretch, torsion, coupling].
    """
    transform_fn = energy_fn.energy_fns[0].transform_fn
    n_nucs_per_strand = top.n_nucleotides // 2
    quartets = obs_base.get_duplex_quartets(n_nucs_per_strand)

    # Define end base pairs for extension measurement
    bp1 = jnp.array([0, 2 * n_nucs_per_strand - 1], dtype=jnp.int32)
    bp2 = jnp.array([n_nucs_per_strand - 1, n_nucs_per_strand], dtype=jnp.int32)

    twist_obs = TwistXY(rigid_body_transform_fn=transform_fn, quartets=quartets, displacement_fn=displacement_fn)
    extension_obs = ExtensionZ(rigid_body_transform_fn=transform_fn, bp1=bp1, bp2=bp2, displacement_fn=displacement_fn)

    beta = jnp.array(1 / kT, dtype=jnp.float64)

    def compute_moduli_from_traj(traj, weights):
        """Compute stretch_torsion moduli from trajectory with metadata filtering.

        Filters trajectory by force/torque conditions and computes weighted
        observables for stretch_torsion calculation.

        Returns:
            Tuple of (s_eff, c, g) moduli from stretch_torsion.
        """
        idx_forces = [(i, md["force"]) for i, md in enumerate(traj.metadata) if md["torque"] == STRETCH_TORQUE]
        idx, forces = (jnp.array(i) for i in zip(*idx_forces, strict=True))
        weights_force = weights[idx]
        traj_force = traj.slice(idx)
        force_extensions = jnp.multiply(weights_force, 0.8518 * extension_obs(traj_force))

        idx_torques = [(i, md["torque"]) for i, md in enumerate(traj.metadata) if md["force"] == TORSION_FORCE]
        idx, torques = (jnp.array(i) for i in zip(*idx_torques, strict=True))
        weights_torque = weights[idx]
        traj_torque = traj.slice(idx)
        torque_extensions = jnp.multiply(weights_torque, 0.8518 * extension_obs(traj_torque))
        torque_twists = jnp.multiply(weights_torque, twist_obs(traj_torque))

        return stretch_torsion(forces, force_extensions, torques, torque_extensions, torque_twists)

    # -------------------------------------------------------------------------
    # Stretch modulus loss function
    # -------------------------------------------------------------------------
    def stretch_modulus_loss_fn(traj, weights, _energy_fn, _opt_params, _observables):
        """Compute stretch modulus loss."""
        s_eff, _, _ = compute_moduli_from_traj(traj, weights)
        loss = jnp.sqrt((s_eff - TARGET_STRETCH_MODULUS) ** 2)
        return loss, (("stretch_modulus", s_eff), {})

    # -------------------------------------------------------------------------
    # Torsional modulus loss function
    # -------------------------------------------------------------------------
    def torsional_modulus_loss_fn(traj, weights, _energy_fn, _opt_params, _observables):
        """Compute torsional modulus loss."""
        _, c, _ = compute_moduli_from_traj(traj, weights)
        loss = jnp.sqrt((c - TARGET_TORSIONAL_MODULUS) ** 2)
        return loss, (("torsional_modulus", c), {})

    # -------------------------------------------------------------------------
    # Twist-stretch coupling loss function
    # -------------------------------------------------------------------------
    def twist_stretch_coupling_loss_fn(traj, weights, _energy_fn, _opt_params, _observables):
        """Compute twist-stretch coupling loss."""
        _, _, g = compute_moduli_from_traj(traj, weights)
        loss = jnp.sqrt((g - TARGET_TWIST_STRETCH_COUPLING) ** 2)
        return loss, (("twist_stretch_coupling", g), {})

    # Common arguments shared by all mechanical objectives
    common_objective_args = {
        "required_observables": simulator.exposes(),
        "energy_fn": energy_fn,
        "opt_params": opt_params,
        "beta": beta,
        "n_equilibration_steps": 0,
        "min_n_eff_factor": 0.95,
    }

    # Create the three objectives
    stretch_objective = jdna_objective.RayObjective(
        jdna_objective.DiffTReObjective,
        name="stretch_modulus",
        logging_observables=["loss", "stretch_modulus", "torsional_modulus", "twist_stretch_coupling", "neff"],
        grad_or_loss_fn=stretch_modulus_loss_fn,
        **common_objective_args,
    )

    torsional_objective = jdna_objective.RayObjective(
        jdna_objective.DiffTReObjective,
        name="torsional_modulus",
        logging_observables=["loss", "torsional_modulus", "neff"],
        grad_or_loss_fn=torsional_modulus_loss_fn,
        **common_objective_args,
    )

    coupling_objective = jdna_objective.RayObjective(
        jdna_objective.DiffTReObjective,
        name="twist_stretch_coupling",
        logging_observables=["loss", "twist_stretch_coupling", "neff"],
        grad_or_loss_fn=twist_stretch_coupling_loss_fn,
        **common_objective_args,
    )

    return [stretch_objective, torsional_objective, coupling_objective]


def create_melting_temp_objective(
    simulator: RayMultiGangSimulation,
    energy_fn: EnergyFunction,
    opt_params: jdna_types.Params,
    kT: float,
    target_temp: float,
    system_name: str,
) -> jdna_objective.DiffTReObjective:
    """Create a melting temperature objective for a thermodynamic system."""
    kt_range = get_kt(jnp.linspace(280, 350, 20))

    melting_temp_fn = MeltingTemp(
        rigid_body_transform_fn=energy_fn.energy_fns[0].transform_fn,
        sim_temperature=kT,
        temperature_range=kt_range,
        energy_fn=energy_fn,
    )

    def melting_temp_loss_fn(traj, weights, _energy_fn, opt_params, observables):
        # Filter energy and histogram info from observables
        e_info = pd.concat([i for i in observables if isinstance(i, EnergyInfo)])
        obs = melting_temp_fn(
            traj,
            e_info["bond"].to_numpy(),
            e_info["weight"].to_numpy(),
            opt_params,
        )
        expected_tm = jnp.dot(weights, obs).sum()
        loss = jnp.sqrt((expected_tm - target_temp) ** 2)
        if not jnp.isfinite(loss):
            raise ValueError("Non-finite loss encountered in melting temp calculation.")
        return loss, (("melting_temp", expected_tm), {})

    beta = jnp.array(1 / kT, dtype=jnp.float64)
    return jdna_objective.RayObjective(
        jdna_objective.DiffTReObjective,
        name=f"melting_temp_{system_name}",
        required_observables=simulator.exposes(),
        logging_observables=["loss", "melting_temp", "neff"],
        grad_or_loss_fn=melting_temp_loss_fn,
        energy_fn=energy_fn,
        opt_params=opt_params,
        beta=beta,
        n_equilibration_steps=0,
        min_n_eff_factor=0.95,
    )


# =============================================================================
# Main function
# =============================================================================
def main(
    num_sims: int = 1,
    learning_rate: float = 1e-3,
    opt_steps: int = 100,
    oxdna_src: str = "../oxDNA",
    use_aim: bool = False,
):
    """Run multi-simulation, multi-objective oxDNA1 reparameterization.

    Args:
        num_sims: Number of parallel simulations per system.
        learning_rate: Learning rate for Adam optimizer.
        opt_steps: Number of optimization steps.
        oxdna_src: Path to oxDNA source directory.
        use_aim: Whether to use Aim for logging (requires: pip install aim).
    """
    # Initialize Ray
    ray.init(
        runtime_env={
            "env_vars": {"JAX_ENABLE_X64": "True", "JAX_PLATFORM_NAME": "cpu"}
        }
    )

    oxdna_src = Path(oxdna_src).resolve()

    # ==========================================================================
    # Create base energy function (will be customized per system)
    # ==========================================================================
    # construct a base energy function with settings common to all systems. We
    # stub a mock for the topology since it will be set per-system using
    # overrides. This base is used primarily to apply global configurations.
    base_energy_fn = dna1_energy.create_default_energy_fn(
        topology=MagicMock(),  # Placeholder, will set per system
    ).with_noopt("ss_stack_weights", "ss_hb_weights", "kt")

    # Get opt_params once from the base energy function - these are shared across all systems
    opt_params = base_energy_fn.opt_params()

    # ==========================================================================
    # Create simulators and objectives for each system type
    # ==========================================================================
    all_simulators = []
    all_objectives = []

    # --------------------------------------------------------------------------
    # Structural systems (use oxDNASimulator)
    # --------------------------------------------------------------------------
    for system_name in STRUCTURAL_SYSTEMS:
        system_dir = STRUCTURAL_DIR / system_name
        config = create_oxdna_system_config(base_energy_fn, system_dir)
        simulator = RayMultiSimulation.create(
            num_sims,
            oxDNASimulator,
            input_dir=system_dir,
            sim_type=jdna_types.oxDNASimulatorType.DNA1,
            energy_fn=config.energy_fn,
            source_path=oxdna_src,
        )
        all_simulators.append(simulator)

        structural_objectives = create_structural_objectives(
            simulator, config.energy_fn, config.topology, config.displacement_fn, opt_params, config.kT
        )
        all_objectives.extend(structural_objectives)
        logging.info("Created %d structural objectives for %s", len(structural_objectives), system_name)

    # --------------------------------------------------------------------------
    # Mechanical systems (use oxDNASimulator or LAMMPSoxDNASimulator)
    # --------------------------------------------------------------------------
    for system_name in MECHANICAL_OXDNA_SYSTEMS:
        system_dir = MECHANICAL_DIR / system_name
        config = create_oxdna_system_config(base_energy_fn, system_dir)

        simulator = RayMultiSimulation.create(
            num_sims,
            oxDNASimulator,
            input_dir=system_dir,
            sim_type=jdna_types.oxDNASimulatorType.DNA1,
            energy_fn=config.energy_fn,
            source_path=oxdna_src,
        )
        all_simulators.append(simulator)

        # oxDNA mechanical uses persistence length objective
        mechanical_objective = create_oxdna_mechanical_objective(
            simulator, config.energy_fn, config.topology, config.displacement_fn, opt_params, config.kT
        )
        all_objectives.append(mechanical_objective)
        logging.info("Created persistence length objective for %s", system_name)

    # LAMMPS-based mechanical systems with force/torque sweep
    # Each system creates 18 simulators (one per force/torque condition)
    # and 3 DiffTRe objectives (stretch modulus, torsional modulus, coupling)
    for system_name in MECHANICAL_LAMMPS_SYSTEMS:
        system_dir = MECHANICAL_DIR / system_name
        # LAMMPS systems have different file structure
        top = jdna_top.from_oxdna_file(system_dir / "data.top")
        box = read_box_size_lammps(system_dir / "data")
        # For LAMMPS, temperature is typically set in the input file
        kT = get_kt_from_string("296.15K")  # Default temperature
        displacement_fn = jax_md.space.periodic(box)[0]

        energy_fn = base_energy_fn.with_props(
            topology=top,
            displacement_fn=displacement_fn,
        ).with_params(kt=kT)

        # Create simulators for all 18 force/torque conditions defined in sweep
        mechanical_simulator = RayMultiSimulation(
            simulations=[
                RaySimulation.create(
                    LAMMPSMechanicalSimulator,
                    input_dir=system_dir,
                    input_file_name="in",
                    energy_fn=energy_fn,
                    force=force,
                    torque=torque,
                    variables={"T": kT},
                )
                for force, torque in MECHANICAL_FORCE_TORQUE_SWEEP
            ]
        )
        all_simulators.append(mechanical_simulator)

        # Create 3 DiffTRe objectives: stretch modulus, torsional modulus, twist-stretch coupling
        # All objectives share the same simulators and filter by condition type
        mechanical_objectives = create_lammps_mechanical_objectives(
            mechanical_simulator, energy_fn, top, displacement_fn, opt_params, kT
        )
        all_objectives.extend(mechanical_objectives)

        logging.info("Created %d mechanical objectives for %s", len(mechanical_objectives), system_name)

    # --------------------------------------------------------------------------
    # Thermodynamic systems (use oxDNAUmbrellaSamplerGang for weight updates)
    # --------------------------------------------------------------------------
    for system_name in THERMODYNAMIC_SYSTEMS:
        system_dir = THERMODYNAMIC_DIR / system_name
        config = create_oxdna_system_config(base_energy_fn, system_dir)
        simulator = oxDNAUmbrellaSamplerGang.create(
            num_sims,
            oxDNAUmbrellaSampler,
            input_dir=system_dir,
            sim_type=jdna_types.oxDNASimulatorType.DNA1,
            energy_fn=config.energy_fn,
            source_path=oxdna_src,
        )
        logging.debug("MultiGangExposes: %s", simulator.exposes())
        all_simulators.append(simulator)

        target_temp = TARGET_MELTING_TEMPS[system_name]
        tm_objective = create_melting_temp_objective(
            simulator, config.energy_fn, opt_params, config.kT, target_temp, system_name
        )
        all_objectives.append(tm_objective)
        logging.info("Created melting temp objective for %s (target: %s)", system_name, target_temp)

    # ==========================================================================
    # Aggregate gradient function
    # ==========================================================================
    def tree_mean(trees:tuple[jdna_types.PyTree]) -> jdna_types.PyTree:
        if len(trees) <= 1:
            return trees[0]
        return jax.tree.map(lambda *x: jnp.mean(jnp.stack(x)), *trees)

    # ==========================================================================
    # Setup optimizer
    # ==========================================================================

    optimizer = jdna_optimization.RayMultiOptimizer(
        objectives=all_objectives,
        simulators=all_simulators,
        aggregate_grad_fn=tree_mean,
        optimizer=optax.adam(learning_rate=learning_rate),
    )

    # ==========================================================================
    # Setup logging
    # ==========================================================================
    if use_aim:
        from mythos.ui.loggers.aim import AimLogger
        aim_logger = AimLogger()
        aim_logger.aim_run.set("learning_rate", learning_rate)
        aim_logger.aim_run.set("num_sims", num_sims)
        aim_logger.aim_run.set("opt_steps", opt_steps)
        aim_logger.aim_run.set("num_objectives", len(all_objectives))
        aim_logger.aim_run.set("num_simulators", len(all_simulators))
    else:
        aim_logger = NullLogger()
    console_logger = ConsoleLogger()
    logger = MultiLogger([aim_logger, console_logger])

    # ==========================================================================
    # Run optimization loop
    # ==========================================================================
    logging.info(f"Starting optimization with {len(all_objectives)} objectives and {len(all_simulators)} simulators")

    opt_params = base_energy_fn.opt_params()
    for step in tqdm(range(opt_steps), desc="Optimizing"):
        try:
            opt_state, opt_params, grads = optimizer.step(opt_params)

            # Log metrics from all objectives
            for objective in all_objectives:
                for name, value in objective.logging_observables().items():
                    logger.log_metric(f"{objective._name}/{name}", value, step=step)

            optimizer = optimizer.post_step(
                optimizer_state=opt_state,
                opt_params=opt_params,
            )
        except Exception as e:
            logging.error(f"Error at step {step}: {e}")
            raise

    logging.info("Optimization complete!")
    logging.info(f"Final parameters: {opt_params}")

    # Cleanup Ray
    ray.shutdown()

    return opt_params


if __name__ == "__main__":
    fire.Fire(main)
