r"""Optimize Martini M2 parameters via DiffTRe using bottom-up Wasserstein distance matching.

Matches simulated bond-distance and triplet-angle distributions against
atomistic reference distributions (stored as .npy files) by minimizing the
mean Wasserstein distance.

Usage examples
--------------
Discover available bond and angle names from the topology::

    python m2_bottom_up_opt.py \
        --input-dir data/templates/martini/m2/DMPC/273K \
        --list-bonds --list-angles

Optimize with one bond and one angle distribution::

    python m2_bottom_up_opt.py \
        --input-dir data/templates/martini/m2/DMPC/273K \
        --bond DMPC_NC3_PO4 ref_bonds/DMPC_NC3_PO4.npy \
        --angle DMPC_NC3_PO4_GL1 ref_angles/DMPC_NC3_PO4_GL1.npy \
        --opt-steps 50

Optimize with multiple bonds, parallel sims, and file logging::

    python m2_bottom_up_opt.py \
        --input-dir data/templates/martini/m2/DMPC/273K \
        --bond DMPC_NC3_PO4 ref_bonds/DMPC_NC3_PO4.npy \
        --bond DMPC_GL1_GL2 ref_bonds/DMPC_GL1_GL2.npy \
        --angle DMPC_NC3_PO4_GL1 ref_angles/DMPC_NC3_PO4_GL1.npy \
        --opt-steps 50 --num-sims 2 --learning-rate 5e-4 \
        --temperature 273 --metrics-file results.log
"""

import argparse
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import MDAnalysis
import numpy as np
import optax
from mythos.energy.base import ComposedEnergyFunction
from mythos.energy.martini.base import MartiniTopology
from mythos.energy.martini.m2 import LJ, Angle, AngleConfiguration, Bond, BondConfiguration, LJConfiguration
from mythos.input.gromacs_input import read_params_from_topology
from mythos.observables.bond_distances import BondDistancesMapped
from mythos.observables.triplet_angles import TripletAnglesMapped
from mythos.observables.wasserstein import WassersteinDistanceMapped
from mythos.optimization.objective import DiffTReObjective
from mythos.optimization.optimization import RayOptimizer
from mythos.simulators.gromacs.gromacs import GromacsSimulator
from mythos.ui.loggers.console import ConsoleLogger
from mythos.ui.loggers.disk import FileLogger
from mythos.ui.loggers.multilogger import MultiLogger
from mythos.utils.helpers import run_command

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize Martini M2 parameters via DiffTRe using bottom-up "
                    "Wasserstein distance matching against reference bond and angle distributions"
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing input files (topol.top, md.mdp, membrane.gro)")
    parser.add_argument("--bond", nargs=2, action="append", metavar=("BOND_NAME", "NPY_FILE"),
                        help="A bond name (e.g. DMPC_NC3_PO4) and a .npy reference distribution file. "
                             "May be specified multiple times.")
    parser.add_argument("--angle", nargs=2, action="append", metavar=("ANGLE_NAME", "NPY_FILE"),
                        help="An angle name (e.g. DMPC_NC3_PO4_GL1) and a .npy reference distribution file. "
                             "May be specified multiple times.")
    parser.add_argument("--bond-ref-units", choices=["nm", "angstrom"], default="angstrom",
                        help="Units of bond reference distributions. If 'angstrom', values are "
                             "converted to nm. (default: angstrom)")
    parser.add_argument("--angle-ref-units", choices=["radian", "degree"], default="radian",
                        help="Units of angle reference distributions. If 'degree', values are "
                             "converted to radians. (default: radian)")
    parser.add_argument("--num-sims", type=int, default=1,
                        help="Number of parallel GROMACS simulations (default: 1)")
    parser.add_argument("--equilibration-steps", type=int, default=200_000,
                        help="Number of equilibration steps (default: 200000)")
    parser.add_argument("--simulation-steps", type=int, default=500_000,
                        help="Number of simulation steps (default: 500000)")
    parser.add_argument("--snapshot-steps", type=int, default=10_000,
                        help="Coordinate output frequency (default: 10000)")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Adam optimizer learning rate (default: 5e-4)")
    parser.add_argument("--temperature", type=float, default=273.0,
                        help="Temperature in Kelvin for beta calculation (default: 273)")
    parser.add_argument("--gromacs-binary", type=Path, default=None,
                        help="Path to GROMACS binary (default: search in PATH)")
    parser.add_argument("--metrics-file", type=Path, default=None,
                        help="File to write metrics to. Defaults to console only.")
    parser.add_argument("--opt-steps", type=int, default=100,
                        help="Number of optimization steps (default: 100)")
    parser.add_argument("--list-bonds", action="store_true",
                        help="Print available bond names from the topology and exit.")
    parser.add_argument("--list-angles", action="store_true",
                        help="Print available angle names from the topology and exit.")
    return parser.parse_args()


def get_universe_and_params(data_dir: Path, gromacs_binary: Path | None = None):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cmd = [gromacs_binary or "gmx", "grompp",
            "-p", "topol.top",
            "-f", "md.mdp",
            "-c", "membrane.gro",
            "-pp", tmpdir / "preprocessed.top",
            "-o", tmpdir / "preprocessed.tpr",
        ]
        run_command(cmd, cwd=data_dir, log_prefix="topology_preprocess")
        universe = MDAnalysis.Universe(tmpdir / "preprocessed.tpr")
        universe.transfer_to_memory()
        params = read_params_from_topology(tmpdir / "preprocessed.top")
    return universe, params


def load_reference_distributions(args):
    """Load bond and angle reference distributions from .npy files."""
    bond_v_map: dict[str, jnp.ndarray] = {}
    for bond_name, npy_file in (args.bond or []):
        npy_path = Path(npy_file)
        ref = np.load(npy_path)
        if args.bond_ref_units == "angstrom":
            ref = ref * 0.1  # Angstrom -> nm
        bond_v_map[bond_name] = jnp.array(ref)

    angle_v_map: dict[str, jnp.ndarray] = {}
    for angle_name, npy_file in (args.angle or []):
        npy_path = Path(npy_file)
        ref = np.load(npy_path)
        if args.angle_ref_units == "degree":
            ref = np.deg2rad(ref)
        angle_v_map[angle_name] = jnp.array(ref)

    return bond_v_map, angle_v_map


if __name__ == "__main__":
    args = parse_args()

    # Handle --list-bonds / --list-angles
    if args.list_bonds or args.list_angles:
        universe, _ = get_universe_and_params(args.input_dir, args.gromacs_binary)
        top = MartiniTopology.from_universe(universe)

        if args.list_bonds:
            print("Available bond names:")
            for name in sorted(set(top.bond_names)):
                count = top.bond_names.count(name)
                print(f"  {name}  ({count} bond{'s' if count > 1 else ''})")
        if args.list_angles:
            print("Available angle names:")
            for name in sorted(set(top.angle_names)):
                count = top.angle_names.count(name)
                print(f"  {name}  ({count} angle{'s' if count > 1 else ''})")
        raise SystemExit(0)

    if not args.bond and not args.angle:
        raise SystemExit("Error: at least one --bond or --angle is required "
                         "(or use --list-bonds / --list-angles).")

    bond_v_map, angle_v_map = load_reference_distributions(args)

    universe, parameters = get_universe_and_params(args.input_dir, args.gromacs_binary)
    top = MartiniTopology.from_universe(universe)

    energy_fn = ComposedEnergyFunction(energy_fns=[
        LJ.from_topology(topology=top, params=LJConfiguration(**parameters["nonbond_params"])),
        Bond.from_topology(topology=top, params=BondConfiguration(**parameters["bond_params"])),
        Angle.from_topology(topology=top, params=AngleConfiguration(**parameters["angle_params"])),
    ])

    # Build Wasserstein observables for bonds and angles
    wasserstein_observables: list[WassersteinDistanceMapped] = []
    if bond_v_map:
        wasserstein_observables.append(WassersteinDistanceMapped(
            observable=BondDistancesMapped(
                topology=top, bond_names=tuple(bond_v_map.keys()),
            ),
            v_distribution_map=bond_v_map,
        ))
    if angle_v_map:
        wasserstein_observables.append(WassersteinDistanceMapped(
            observable=TripletAnglesMapped(
                topology=top, angle_names=tuple(angle_v_map.keys()),
            ),
            v_distribution_map=angle_v_map,
        ))

    n_total = sum(len(obs.v_distribution_map) for obs in wasserstein_observables)

    simulator = GromacsSimulator.create_n(args.num_sims,
        input_dir=args.input_dir,
        energy_fn=energy_fn.with_props(unbonded_neighbors=None),
        equilibration_steps=args.equilibration_steps,
        simulation_steps=args.simulation_steps,
        input_overrides={
            "nstxout": args.snapshot_steps,
            "ref-t": args.temperature,
        },
        binary_path=args.gromacs_binary,
    )

    def wasserstein_loss(traj, weights, *_):
        total = jnp.float64(0.0)
        for obs in wasserstein_observables:
            w_distances = obs(traj, weights)
            for v in w_distances.values():
                total = total + v
        loss = jnp.sqrt(total / n_total)
        return loss, (("wasserstein_mean", loss), ())

    beta = 1 / (0.0083144621 * args.temperature)
    wasserstein_obj = DiffTReObjective(
        energy_fn=energy_fn,
        grad_or_loss_fn=wasserstein_loss,
        required_observables=[i for s in simulator for i in s.exposes()],
        name="WassersteinBottomUp",
        beta=beta,
    )

    loggers = [ConsoleLogger()]
    if args.metrics_file is not None:
        loggers.append(FileLogger(args.metrics_file))

    opt = RayOptimizer(
        simulators=simulator,
        objectives=[wasserstein_obj],
        optimizer=optax.adam(learning_rate=args.learning_rate),
        aggregate_grad_fn=lambda x: x[0],
        logger=MultiLogger(loggers),
    )

    opt_params = energy_fn.opt_params()

    # Print summary
    print("\n=== Setup Complete ===")
    print(f"Data directory: {args.input_dir}")
    print(f"Energy function terms: {[type(fn).__name__ for fn in energy_fn.energy_fns]}")
    print(f"Number of parallel simulations: {args.num_sims}")
    print(f"Objective: {wasserstein_obj.__class__.__qualname__}({wasserstein_obj.name})")
    if bond_v_map:
        print(f"  Bond distributions: {list(bond_v_map.keys())}")
    if angle_v_map:
        print(f"  Angle distributions: {list(angle_v_map.keys())}")
    print(f"Optimizing {len(opt_params)} parameters")

    opt.run(params=opt_params, n_steps=args.opt_steps)
