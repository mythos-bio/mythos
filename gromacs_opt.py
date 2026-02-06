"""GROMACS Martini2 simulation setup with composed energy functions.

This script demonstrates:
1. Preprocessing GROMACS topology files using grompp to expand macros
2. Using GromacsParamsParser to extract Martini2 force field parameters
3. Creating Bond, Angle, and LJ energy configurations from the topology
4. Composing these into a unified energy function
5. Setting up a GromacsSimulator for running MD simulations
"""

import itertools
import logging
import shutil
import tempfile
from pathlib import Path

import jax.numpy as jnp
import lipyphilic as lpp
import MDAnalysis
import optax
from mythos.energy.base import ComposedEnergyFunction
from mythos.energy.martini.m2 import LJ, Angle, AngleConfiguration, LJConfiguration
from mythos.energy.martini.m2.bond import Bond, BondConfiguration
from mythos.input.gromacs_input import read_params_from_topology
from mythos.optimization.objective import DiffTReObjective
from mythos.optimization.optimization import SimpleOptimizer
from mythos.simulators.gromacs import GromacsSimulator
from mythos.utils.helpers import run_command

DATA_DIR = Path("data/templates/martini/m2/DMPC/273K")
TOPO_FILE = "topol_pp.top"
OUTPUT_PREFIX = "output"


def preprocess_topology(data_dir: Path):
    cmd = ["gmx", "grompp",
           "-p", "topol.top",
           "-f", "md.mdp",
           "-c", "membrane.gro",
           "-pp", TOPO_FILE,
           "-o", OUTPUT_PREFIX
    ]
    run_command(cmd, cwd=data_dir, log_prefix="topology_preprocess")


def get_system_info_from_structure(data_dir: Path) -> dict:
    # Try to load from representative TPR if available, otherwise use GRO
    u = MDAnalysis.Universe(data_dir / f"{OUTPUT_PREFIX}.tpr")

    atom_types = tuple(u.atoms.types)

    # Extract bond information
    bond_names = tuple(
        f"{u.atoms[b[0]].resname}_{u.atoms[b[0]].name}_{u.atoms[b[1]].name}"
        for b in u.bonds.indices
    )
    bonded_neighbors = jnp.array(u.bonds.indices)
    unbonded = set(itertools.combinations(range(len(u.atoms)), 2))
    unbonded -= {tuple(i) for i in bonded_neighbors.tolist()}
    unbonded -= {tuple(reversed(i)) for i in bonded_neighbors.tolist()}
    unbonded_neighbors = jnp.array(list(unbonded))

    # Extract angle information
    angle_names = tuple(
        f"{u.atoms[a[0]].resname}_{u.atoms[a[0]].name}_{u.atoms[a[1]].name}_{u.atoms[a[2]].name}"
        for a in u.angles.indices
    )
    angles = jnp.array(u.angles.indices)

    return {
        "atom_types": atom_types,
        "bond_names": bond_names,
        "angle_names": angle_names,
        "angles": angles,
        "bonded_neighbors": bonded_neighbors,
        "unbonded_neighbors": unbonded_neighbors,
    }


def create_martini2_energy_function(data_dir: Path) -> ComposedEnergyFunction:
    preprocess_topology(data_dir)
    params = read_params_from_topology(data_dir / TOPO_FILE)
    lj_params = params["nonbond_params"]
    bond_params = params["bond_params"]
    angle_params = params["angle_params"]
    sys_info = get_system_info_from_structure(data_dir)

    # Convert angle theta0 from degrees to radians for the configuration
    angle_params_rad = {}
    for key, value in angle_params.items():
        if key.startswith("angle_theta0_"):
            angle_params_rad[key] = jnp.deg2rad(value)
        else:
            angle_params_rad[key] = value

    # Create configurations
    lj_config = LJConfiguration(**lj_params)
    bond_config = BondConfiguration(**bond_params)
    angle_config = AngleConfiguration(**angle_params_rad)

    lj_fn = LJ(params=lj_config, **sys_info)
    bond_fn = Bond(params=bond_config, **sys_info)
    angle_fn = Angle(params=angle_config, **sys_info)

    # Compose into a single energy function
    return ComposedEnergyFunction(
        energy_fns=[bond_fn, angle_fn, lj_fn]
    )


def calculate_membrane_thickness(
    universe: MDAnalysis.Universe,
    lipid_sel: str = "name GL1 GL2",
    thickness_sel: str = "name PO4",
):
    leaflets = lpp.AssignLeaflets(universe=universe, lipid_sel=lipid_sel)
    leaflets.run()
    # Calculate membrane thickness
    thicknesses = lpp.analysis.MembThickness(
        universe=universe, lipid_sel=thickness_sel, leaflets=leaflets.leaflets
    )
    thicknesses.run()
    return thicknesses.memb_thickness


TARGET_THICKNESS = 32  # nm
def thickness_loss(traj, weights, ef, opt_params, observables):
    # filter universe types
    universes = [obs for obs in observables if isinstance(obs, MDAnalysis.Universe)]
    all_thickness = jnp.hstack([calculate_membrane_thickness(u) for u in universes])
    expected_thickness = jnp.dot(weights, all_thickness)
    loss = jnp.sqrt((TARGET_THICKNESS - expected_thickness)**2)
    return loss, (("thickness", expected_thickness),())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    energy_fn = create_martini2_energy_function(DATA_DIR)

    simulator = GromacsSimulator(
        input_dir=DATA_DIR,
        energy_fn=energy_fn,
        equilibration_steps=10000,
        simulation_steps=50000,
        binary_path="/Users/arik/ws/ssec/gromacs-2025.4/build/bin/gmx",  # Adjust as needed
    )

    thickness_obj = DiffTReObjective(
        energy_fn=energy_fn,
        grad_or_loss_fn=thickness_loss,
        required_observables=simulator.exposes(),
        name="MembraneThickness",
        beta=1/(0.0083144621 * 273)
    )

    opt = SimpleOptimizer(
        simulator=simulator,
        objective=thickness_obj,
        optimizer=optax.adam(learning_rate=1e-3),
    )

    # Print summary
    print("\n=== Setup Complete ===")
    print(f"Data directory: {DATA_DIR}")
    print(f"Energy function terms: {[type(fn).__name__ for fn in energy_fn.energy_fns]}")
    print(f"Simulator input dir: {simulator.input_dir}")

    opt_params = energy_fn.opt_params()
    opt_state = None
    for step in range(20):
        opt_out = opt.step(params=opt_params, state=opt_state)
        opt_state = opt_out.state
        opt_params = opt_out.opt_params
        metrics = opt_out.observables
        print(f"Step {step}: observables = {metrics}")


