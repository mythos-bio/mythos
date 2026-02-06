"""GROMACS Martini2 simulation setup with composed energy functions.

This script demonstrates:
1. Preprocessing GROMACS topology files using grompp to expand macros
2. Using GromacsParamsParser to extract Martini2 force field parameters
3. Creating Bond, Angle, and LJ energy configurations from the topology
4. Composing these into a unified energy function
5. Setting up a GromacsSimulator for running MD simulations
"""

import logging
import shutil
import tempfile
from pathlib import Path
import lipyphilic as lpp

import jax.numpy as jnp
import MDAnalysis
import optax

from mythos.energy.base import ComposedEnergyFunction
from mythos.energy.martini.m2 import Angle, AngleConfiguration, LJ, LJConfiguration
from mythos.energy.martini.m2.bond import Bond, BondConfiguration
from mythos.input.gromacs_input import read_params_from_topology
from mythos.optimization.objective import DiffTReObjective
from mythos.optimization.optimization import SimpleOptimizer
from mythos.simulators.gromacs import GromacsSimulator
from mythos.utils.helpers import run_command


DATA_DIR = Path("data/templates/martini/m2/DMPC/273K")
PREPROCESSED_TOPOLOGY_FILE = "_pp_topol.top"


def preprocess_topology(
    data_dir: Path,
    topology_file: str = "topol.top",
    mdp_file: str = "md.mdp",
    structure_file: str = "membrane.gro",
    output_file: str = PREPROCESSED_TOPOLOGY_FILE,
    gmx_binary: str | None = None,
) -> Path:
    """Preprocess a GROMACS topology file using grompp to expand macros.

    Args:
        data_dir: Directory containing the GROMACS input files
        topology_file: Name of the topology file
        mdp_file: Name of the MDP file
        structure_file: Name of the structure file
        output_file: Name for the preprocessed output file
        gmx_binary: Path to gmx binary (auto-detected if None)

    Returns:
        Path to the preprocessed topology file
    """
    gmx = gmx_binary or shutil.which("gmx")
    if gmx is None:
        raise FileNotFoundError(
            "GROMACS binary not found. Please install GROMACS or provide path."
        )

    output_path = data_dir / output_file

    # Use grompp with -pp flag to preprocess topology
    cmd = [
        gmx,
        "grompp",
        "-p", topology_file,
        "-f", mdp_file,
        "-c", structure_file,
        "-pp", output_file,
        "-maxwarn", "10",  # Allow warnings during preprocessing
    ]

    run_command(cmd, cwd=data_dir, log_prefix="topology_preprocess")

    if not output_path.exists():
        raise FileNotFoundError(
            f"Preprocessed topology file not created: {output_path}"
        )

    print(f"Created preprocessed topology: {output_path}")
    return output_path


def get_system_info_from_structure(data_dir: Path) -> dict:
    """Extract system information from GROMACS structure files using MDAnalysis.

    Returns dictionary with:
        - atom_types: tuple of atom type strings
        - bond_names: tuple of bond name strings (MOLTYPE_ATOM1_ATOM2)
        - angle_names: tuple of angle name strings (MOLTYPE_ATOM1_ATOM2_ATOM3)
        - angles: array of angle triplets (atom indices)
        - bonded_neighbors: array of bonded pairs
        - unbonded_neighbors: array of non-bonded pairs for LJ
    """
    # Try to load from representative TPR if available, otherwise use GRO
    tpr_file = data_dir / "representative.tpr"
    gro_file = data_dir / "membrane.gro"

    if tpr_file.exists():
        u = MDAnalysis.Universe(str(tpr_file))
    else:
        u = MDAnalysis.Universe(str(gro_file))

    atom_types = tuple(u.atoms.types)

    # Extract bond information
    bond_names = tuple(
        f"{u.atoms[b[0]].resname}_{u.atoms[b[0]].name}_{u.atoms[b[1]].name}"
        for b in u.bonds.indices
    )
    bonded_neighbors = jnp.array(u.bonds.indices)

    # Extract angle information
    angle_names = tuple(
        f"{u.atoms[a[0]].resname}_{u.atoms[a[0]].name}_{u.atoms[a[1]].name}_{u.atoms[a[2]].name}"
        for a in u.angles.indices
    )
    angles = jnp.array(u.angles.indices)

    # For unbonded neighbors, we need all pairs that are not bonded
    # This is computed during energy function creation based on cutoff
    n_atoms = len(u.atoms)
    all_pairs = set()
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            all_pairs.add((i, j))

    bonded_set = {tuple(sorted(b)) for b in u.bonds.indices}
    unbonded_pairs = list(all_pairs - bonded_set)
    unbonded_neighbors = jnp.array(unbonded_pairs) if unbonded_pairs else jnp.empty((0, 2), dtype=int)

    return {
        "atom_types": atom_types,
        "bond_names": bond_names,
        "angle_names": angle_names,
        "angles": angles,
        "bonded_neighbors": bonded_neighbors,
        "unbonded_neighbors": unbonded_neighbors,
    }


def create_martini2_energy_function(
    data_dir: Path,
    preprocessed_topology: Path | None = None,
) -> ComposedEnergyFunction:
    """Create a composed energy function with all Martini2 terms.

    If preprocessed_topology is not provided, this will run grompp to
    preprocess the topology file first.

    Parses the topology file and creates:
    - Bond energy function (harmonic bonds)
    - Angle energy function (G96 cosine-based angles)
    - LJ energy function (Lennard-Jones non-bonded)

    Args:
        data_dir: Directory containing GROMACS input files
        preprocessed_topology: Path to preprocessed topology file (optional)

    Returns:
        ComposedEnergyFunction containing all three energy terms
    """
    # Preprocess topology if not provided
    if preprocessed_topology is None:
        preprocessed_topology = preprocess_topology(data_dir)

    # Parse parameters from preprocessed topology
    print(f"Parsing preprocessed topology: {preprocessed_topology}")
    params = read_params_from_topology(preprocessed_topology)

    lj_params = params["nonbond_params"]
    bond_params = params["bond_params"]
    angle_params = params["angle_params"]

    print(f"  Found {len(lj_params) // 2} LJ pair types")
    print(f"  Found {len(bond_params) // 2} bond types")
    print(f"  Found {len(angle_params) // 2} angle types")

    # Get system structure information
    print("Loading system structure...")
    sys_info = get_system_info_from_structure(data_dir)

    print(f"  {len(sys_info['atom_types'])} atoms")
    print(f"  {len(sys_info['bond_names'])} bonds")
    print(f"  {len(sys_info['angle_names'])} angles")

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

    # Create individual energy functions
    common_kwargs = {
        "atom_types": sys_info["atom_types"],
        "bond_names": sys_info["bond_names"],
        "angle_names": sys_info["angle_names"],
        "angles": sys_info["angles"],
        "bonded_neighbors": sys_info["bonded_neighbors"],
        "unbonded_neighbors": sys_info["unbonded_neighbors"],
    }

    lj_fn = LJ(params=lj_config, **common_kwargs)
    bond_fn = Bond(params=bond_config, **common_kwargs)
    angle_fn = Angle(params=angle_config, **common_kwargs)

    # Compose into a single energy function
    composed_energy_fn = ComposedEnergyFunction(
        energy_fns=[bond_fn, angle_fn, lj_fn]
    )

    print("Created composed energy function with Bond + Angle + LJ terms")
    return composed_energy_fn


def create_gromacs_simulator(
    data_dir: Path,
    energy_fn: ComposedEnergyFunction,
) -> GromacsSimulator:
    """Create a GROMACS simulator for the given data directory.

    Args:
        data_dir: Path to directory containing GROMACS input files
        energy_fn: The composed energy function with all Martini2 terms

    Returns:
        Configured GromacsSimulator instance
    """
    simulator = GromacsSimulator(
        input_dir=data_dir,
        energy_fn=energy_fn,
        equilibration_steps=10000,
        simulation_steps=50000,
        binary_path="/Users/arik/ws/ssec/gromacs-2025.4/build/bin/gmx",  # Adjust as needed
    )

    print(f"Created GromacsSimulator for {data_dir}")
    return simulator


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
    # Check if a preprocessed topology already exists, otherwise preprocess
    preprocessed_path = DATA_DIR / PREPROCESSED_TOPOLOGY_FILE

    if preprocessed_path.exists():
        print(f"Using existing preprocessed topology: {preprocessed_path}")
    else:
        print("Preprocessed topology not found, will run grompp to create it...")
        # This will raise an error if GROMACS is not available
        preprocessed_path = None

    # Create the composed energy function from topology
    # This will preprocess the topology file first using grompp if needed
    energy_fn = create_martini2_energy_function(DATA_DIR, preprocessed_path)

    # Create the GROMACS simulator
    simulator = create_gromacs_simulator(DATA_DIR, energy_fn)

    # Print summary
    print("\n=== Setup Complete ===")
    print(f"Data directory: {DATA_DIR}")
    print(f"Energy function terms: {[type(fn).__name__ for fn in energy_fn.energy_fns]}")
    print(f"Simulator input dir: {simulator.input_dir}")

    #simout = simulator.run()
    #thickness = calculate_membrane_thickness(simout.observables[1])  # MDAnalysis Universe is the second observable
    #print("Thickness shape:", thickness.shape)
    #print("Average membrane thickness (nm):", jnp.mean(thickness))

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

    opt_params = energy_fn.opt_params()
    opt_state = None
    for step in range(20):
        opt_out = opt.step(params=opt_params, state=opt_state)
        opt_state = opt_out.state
        opt_params = opt_out.opt_params
        metrics = opt_out.observables
        print(f"Step {step}: observables = {metrics}")


