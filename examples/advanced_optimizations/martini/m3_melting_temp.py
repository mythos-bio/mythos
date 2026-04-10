"""membrane melting temperature optimization via DiffTRe.

Evaluation script (not for check-in) that runs 28 GROMACS simulations at
different temperatures in parallel via Ray, then optimises Martini force-field
parameters so that the fitted melting temperature matches the experimental
target.

DiffTRe computes per-state Boltzmann weights using the per-state temperature
stored on each ``SimulatorTrajectory``. Each GROMACS simulator reads its
reference temperature from the MDP file (overridden via ``input_overrides``).

Usage::

    python m3_melting_temp.py \
        --input-dir gromacs_input_dir \
        --opt-steps 50 --learning-rate 5e-4

Prerequisites:
    * A GROMACS input directory with the standard Martini template files
      (topol.top, md.mdp, membrane.gro, index.ndx).
    * GROMACS installed and available as ``gmx`` on PATH (or via --gromacs-binary).
"""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import MDAnalysis
import optax
from jax import numpy as jnp
from mythos.energy.base import ComposedEnergyFunction
from mythos.energy.martini.base import MartiniTopology
from mythos.energy.martini.m2 import (
    Angle,
    AngleConfiguration,
    Bond,
    BondConfiguration,
)
from mythos.input.gromacs_input import read_params_from_topology
from mythos.observables.membrane_melting_temp import (
    MembraneMeltingTemp,
)
from mythos.optimization.objective import DiffTReObjective
from mythos.optimization.optimization import OptimizerOutput, RayOptimizer
from mythos.simulators.gromacs.gromacs import GromacsSimulator
from mythos.ui.loggers.console import ConsoleLogger
from mythos.ui.loggers.disk import FileLogger
from mythos.ui.loggers.multilogger import MultiLogger
from mythos.utils.helpers import run_command

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

logger = logging.getLogger(__name__)

# Boltzmann constant in kJ/(mol*K) -- GROMACS energy units
KB = 0.0083144621

# target melting temperature (K)
DEFAULT_TARGET_TM = 314.0

# Simulation temperatures to run (K) -- from jax-martini experiment
DEFAULT_SIM_TEMPS: tuple[float, ...] = (
    291, 292.5, 294, 295.5, 297, 298.5, 300, 301.5,
    303, 304.5, 306, 307.5, 309, 310.5, 312, 313.5,
    315, 316.5, 318, 319.5, 321, 322.5, 324, 325.5,
    327, 328.5, 330, 331.5,
)


def preprocess_topology(
    data_dir: Path, gromacs_binary: str | None = None,
) -> tuple[MDAnalysis.Universe, dict[str, Any]]:
    """Run grompp to get a preprocessed topology & TPR, return (Universe, params)."""
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        gmx = gromacs_binary or shutil.which("gmx")
        if gmx is None:
            raise ValueError("GROMACS binary not found")
        cmd = [
            gmx, "grompp",
            "-p", "topol.top",
            "-f", "md.mdp",
            "-c", "membrane.gro",
            "-n", "index.ndx",
            "-pp", str(tmpdir / "preprocessed.top"),
            "-o", str(tmpdir / "preprocessed.tpr"),
        ]
        run_command(cmd, cwd=data_dir, log_prefix="topology_preprocess")
        universe = MDAnalysis.Universe(tmpdir / "preprocessed.tpr")
        universe.transfer_to_memory()
        params = read_params_from_topology(tmpdir / "preprocessed.top")
    return universe, params


def build_energy_fn(
    universe: MDAnalysis.Universe, params: dict[str, Any],
) -> tuple[MartiniTopology, ComposedEnergyFunction]:
    """Build a ComposedEnergyFunction from a preprocessed topology."""
    top = MartiniTopology.from_universe(universe)
    energy_fn = ComposedEnergyFunction(energy_fns=[
        # LJ is not used in wet m3
        Bond.from_topology(topology=top, params=BondConfiguration(**params["bond_params"])),
        Angle.from_topology(topology=top, params=AngleConfiguration(**params["angle_params"])),
    ])
    return top, energy_fn


def build_tm_loss_fn(
    tm_observable: MembraneMeltingTemp,
    target_tm: float,
):
    """Build a DiffTRe-compatible loss function for melting temperature.

    DiffTRe now computes per-state Boltzmann weights using the per-state
    temperature on the trajectory, so no beta-ratio power-correction is
    needed here.  The weights are passed directly to
    :class:`MembraneMeltingTemp` which handles APL computation,
    segmentation, sigmoid fitting, and Tm extraction.
    """

    def tm_loss(traj, weights, *_):
        # MembraneMeltingTemp handles APL, segmentation, sigmoid fit
        tm = tm_observable(traj, weights=weights)
        loss = (tm - target_tm) ** 2
        return loss, (("tm", tm), ())

    return tm_loss


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="membrane melting temperature optimisation via DiffTRe."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Path to GROMACS input directory with Martini template files.",
    )
    parser.add_argument(
        "--lipid-sel", type=str, default="name GL1 GL2",
        help="MDAnalysis selection string for lipid tail beads (default: 'name GL1 GL2').",
    )
    parser.add_argument(
        "--target-tm", type=float, default=DEFAULT_TARGET_TM,
        help=f"Target melting temperature in Kelvin (default: {DEFAULT_TARGET_TM} K).",
    )
    parser.add_argument(
        "--sim-temps", type=float, nargs="+", default=DEFAULT_SIM_TEMPS,
        help=f"Simulation temperatures in Kelvin (default: {DEFAULT_SIM_TEMPS}).",
    )
    parser.add_argument(
        "--simulation-steps", type=int, default=500_000,
        help="Number of MD steps per simulation (default: 500000).",
    )
    parser.add_argument(
        "--equilibration-steps", type=int, default=0,
        help="Equilibration snapshots to discard per trajectory (default: 0).",
    )
    parser.add_argument(
        "--snapshot-steps", type=int, default=1000,
        help="Snapshot frequency in MD steps (default: 1000).",
    )
    parser.add_argument(
        "--opt-steps", type=int, default=50,
        help="Number of optimisation iterations (default: 50).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-4,
        help="Adam learning rate (default: 5e-4).",
    )
    parser.add_argument(
        "--biphasic-equil-steps", type=int, default=0,
        help="Number of biphasic equilibration steps before production (default: 0 = off).",
    )
    parser.add_argument(
        "--gromacs-binary", type=str, default=None,
        help="Path to gmx binary (default: search PATH).",
    )
    parser.add_argument(
        "--metrics-file", type=Path, default=None,
        help="Optional path to write metrics CSV log (default: console only).",
    )
    return parser.parse_args()


def main() -> None:
    """Run melting temperature optimisation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    # 1. Preprocess topology & build energy function
    print(f"Preprocessing topology from {args.input_dir} ...")
    universe, params = preprocess_topology(args.input_dir, args.gromacs_binary)
    _, energy_fn = build_energy_fn(universe, params)

    # 2. Melting temperature observable
    tm_observable = MembraneMeltingTemp(
        topology=universe,
        lipid_sel=args.lipid_sel,
        temperatures=jnp.array(args.sim_temps)*KB,
    )

    # 3. Create one simulator per temperature
    simulators = []
    for i, temp in enumerate(args.sim_temps):
        simulators.append(GromacsSimulator(
            name=f"sim.{i}.temp_{temp:.1f}K",
            input_dir=args.input_dir,
            energy_fn=energy_fn.with_props(unbonded_neighbors=None),
            simulation_steps=args.simulation_steps,
            equilibration_steps=0,
            input_overrides={"nstxout": args.snapshot_steps, "ref-t": temp},
            binary_path=Path(args.gromacs_binary) if args.gromacs_binary else None,
        ))

    # 4. Build loss function and DiffTRe objective
    tm_loss = build_tm_loss_fn(tm_observable, args.target_tm*KB)
    required_obs = tuple(name for sim in simulators for name in sim.exposes())

    objective = DiffTReObjective(
        energy_fn=energy_fn,
        grad_or_loss_fn=tm_loss,
        required_observables=required_obs,
        logging_observables=("loss", "tm", "neff"),
        name="MeltingTemp",
        n_equilibration_steps=args.equilibration_steps,
        max_valid_opt_steps=5,
    )

    # 5. Optimizer
    opt = RayOptimizer(
        simulators=simulators,
        objectives=[objective],
        optimizer=optax.adam(learning_rate=args.learning_rate),
        aggregate_grad_fn=lambda grads: grads[0],
        logger=MultiLogger([ConsoleLogger(), FileLogger(args.metrics_file)])
            if args.metrics_file else ConsoleLogger(),
    )

    opt_params = energy_fn.opt_params()

    print("\n=== Melting Temperature Optimisation ===")
    print(f"  Target Tm:        {args.target_tm} K")
    print(f"  Sim temperatures: {len(args.sim_temps)} ({args.sim_temps[0]}-{args.sim_temps[-1]} K)")
    print(f"  Parallel sims:    {len(simulators)}")
    print(f"  Sim steps:        {args.simulation_steps}")
    print(f"  Opt steps:        {args.opt_steps}")
    print(f"  Learning rate:    {args.learning_rate}")
    print(f"  Parameters:       {len(opt_params)}\n")

    # callback to add a logging observable for the melting temp in K
    def opt_callback(optimizer_output: OptimizerOutput, step: int) -> tuple[OptimizerOutput, bool]:
        melting = optimizer_output.observables["MeltingTemp"]
        melting["tm_k"] = melting["tm"] / KB
        return optimizer_output, True

    output = opt.run(params=opt_params, n_steps=args.opt_steps, callback=opt_callback)

    print("\n=== Optimisation Complete ===")
    print(f"  Final params: {output.opt_params}")


if __name__ == "__main__":
    main()
