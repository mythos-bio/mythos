import argparse
import tempfile
from pathlib import Path

import jax.numpy as jnp
import MDAnalysis
import optax
from mythos.energy.base import ComposedEnergyFunction
from mythos.energy.martini.base import MartiniTopology
from mythos.energy.martini.m2 import LJ, Angle, AngleConfiguration, Bond, BondConfiguration, LJConfiguration
from mythos.input.gromacs_input import read_params_from_topology
from mythos.observables.membrane_thickness import MembraneThickness
from mythos.optimization.objective import DiffTReObjective
from mythos.optimization.optimization import RayOptimizer
from mythos.simulators.gromacs.gromacs import GromacsSimulator
from mythos.ui.loggers.console import ConsoleLogger
from mythos.ui.loggers.disk import FileLogger
from mythos.ui.loggers.multilogger import MultiLogger
from mythos.utils.helpers import run_command


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize Martini M2 membrane thickness via DiffTRe"
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing input files (topol.top, md.mdp, membrane.gro)")
    parser.add_argument("--num-sims", type=int, default=1,
                        help="Number of parallel GROMACS simulations (default: 1)")
    parser.add_argument("--target-thickness", type=float, required=True,
                        help="Target membrane thickness value for the loss function")
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
                        help="File to write metrics to. defaults to console only.")
    parser.add_argument("--opt-steps", type=int, default=100,
                        help="Number of optimization steps (default: 100)")
    return parser.parse_args()


def get_universe_and_params(data_dir: Path, gromacs_binary: Path|None = None):
    # Preprocess topology and create a representative tpr topology file for
    # creating universe for use with energy funtion and observables.
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
        # ensure we can read data after temp dir is deleted and in remote
        # context, since MDAnalysis keeps handle to file otherwise.
        universe.transfer_to_memory()
        params = read_params_from_topology(tmpdir / "preprocessed.top")
    return universe, params

if __name__ == "__main__":
    args = parse_args()

    universe, parameters = get_universe_and_params(args.input_dir, args.gromacs_binary)
    top = MartiniTopology.from_universe(universe)
    energy_fn = ComposedEnergyFunction(energy_fns=[
        LJ.from_topology(topology=top, params=LJConfiguration(**parameters["nonbond_params"])),
        Bond.from_topology(topology=top, params=BondConfiguration(**parameters["bond_params"])),
        Angle.from_topology(topology=top, params=AngleConfiguration(**parameters["angle_params"])),
    ])

    simulator = GromacsSimulator.create_n(args.num_sims,
        input_dir=args.input_dir,
        energy_fn=energy_fn.with_props(unbonded_neighbors=None), # don't use and overhead for ray task
        equilibration_steps=args.equilibration_steps,
        simulation_steps=args.simulation_steps,
        input_overrides={
            "nstxout": args.snapshot_steps,
            "ref-t": args.temperature,
        },
        binary_path=args.gromacs_binary,
    )

    thickness_obs = MembraneThickness(
        topology=universe, lipid_sel="name GL1 GL2", thickness_sel="name PO4"
    )

    def thickness_loss(traj, weights, *_):
        all_thickness = thickness_obs(traj)
        expected_thickness = jnp.dot(weights, all_thickness)
        loss = jnp.sqrt((args.target_thickness - expected_thickness)**2)
        return loss, (("thickness", expected_thickness),())

    thickness_obj = DiffTReObjective(
        energy_fn=energy_fn,
        grad_or_loss_fn=thickness_loss,
        required_observables=[i for s in simulator for i in s.exposes()],
        name="MembraneThickness",
        beta=1/(0.0083144621 * args.temperature)
    )

    opt = RayOptimizer(
        simulators=simulator,
        objectives=[thickness_obj],
        optimizer=optax.adam(learning_rate=args.learning_rate),
        aggregate_grad_fn=lambda x: x[0],  # only single objective.
    )

    opt_params = energy_fn.opt_params()

    # Print summary
    print("\n=== Setup Complete ===")
    print(f"Data directory: {args.input_dir}")
    print(f"Energy function terms: {[type(fn).__name__ for fn in energy_fn.energy_fns]}")
    print(f"Number of parallel simulations: {args.num_sims}")
    print(f"Objective: {thickness_obj.__class__.__qualname__}({thickness_obj.name}) "
          f"with target {args.target_thickness} nm")
    print(f"Optimizing {len(opt_params)} parameters")

    # metrics logging, set --metrics-file to enable file logging and set
    # destination file
    loggers = [ConsoleLogger()]
    if args.metrics_file is not None:
        loggers.append(FileLogger(args.metrics_file))
    mlogger = MultiLogger(loggers)

    opt_state = None
    for step in range(args.opt_steps):
        opt_out = opt.step(params=opt_params, state=opt_state)
        opt_state = opt_out.state
        opt_params = opt_out.opt_params
        metrics = opt_out.observables
        for c, obs in metrics.items():
            for m, val in obs.items():
                mlogger.log_metric(f"{c}.{m}", val, step)
