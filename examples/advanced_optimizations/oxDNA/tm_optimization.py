"""An example of running a melting temperature simulation using oxDNA.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

python tm_opt_ray.py
"""
import functools
import itertools
import logging
import typing
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import jax_dna.energy as jdna_energy
import jax_dna.energy.dna1 as jdna1_energy
import jax_dna.input.topology as jdna_top
import jax_dna.optimization.objective as jdna_objective
import jax_dna.optimization.optimization as jdna_optimization
import jax_dna.utils.types as jdna_types
import jax_md
import optax
import pandas as pd
import ray
from jax_dna.input import oxdna_input
from jax_dna.observables.melting_temp import MeltingTemp
from jax_dna.simulators import oxdna
from jax_dna.simulators.oxdna.utils import read_energy
from jax_dna.ui.loggers.console import ConsoleLogger
from jax_dna.ui.loggers.logger import NullLogger
from jax_dna.ui.loggers.multilogger import MultiLogger
from jax_dna.utils.units import get_kt, get_kt_from_string

jax.config.update("jax_enable_x64", True)
logging.basicConfig(level=logging.INFO)
logging.getLogger("jax").setLevel(logging.WARNING)


def main(
        num_sims: int = 10,
        learning_rate: float = 1e-3,
        opt_steps: int = 100,
        input_dir: str = "data/templates/tm-6bp-2op",
        target_temp: str = "31.2C",
        oxdna_src: str = "../oxDNA",
        use_aim: bool = False,
    ):
    input_dir = Path(input_dir)
    oxdna_src = Path(oxdna_src).resolve()
    try:
        target_temp = get_kt_from_string(target_temp)
    except ValueError:  # assume it is in simulation units
        target_temp = float(target_temp)
    kt_range = get_kt(jnp.linspace(280, 350, 20))
    top = jdna_top.from_oxdna_file(input_dir / "sys.top")
    sim_config = oxdna_input.read(input_dir / "input")
    kT = get_kt_from_string(sim_config["T"])
    with input_dir.joinpath(sim_config["conf_file"]).open("r") as f:
        for line in f:
            if line.startswith("b ="):
                box_size = line.split("=")[1].strip().split()
                box_size = jnp.array([float(i) for i in box_size])
                break

    # Setup the energy functions and configs
    _, energy_config = jdna1_energy.default_configs()
    energy_fns = jdna1_energy.default_energy_fns()
    energy_configs = []
    opt_params = []

    for ec in jdna1_energy.default_energy_configs():
        if isinstance(ec, jdna1_energy.StackingConfiguration):
            ec = ec.replace( non_optimizable_required_params=(
                    "ss_stack_weights",
                    "kt",
                ),
                kt=kT)
            opt_params.append(ec.opt_params)
            energy_configs.append(ec)
        elif isinstance(ec, jdna1_energy.HydrogenBondingConfiguration):
            ec = ec.replace(non_optimizable_required_params=("ss_hb_weights") )
            opt_params.append(ec.opt_params)
            energy_configs.append(ec)
        else:
            energy_configs.append(ec)
            opt_params.append(ec.opt_params)

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        jdna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    energy_fn_builder_fn = jdna_energy.energy_fn_builder(
        energy_fns=energy_fns,
        energy_configs=energy_configs,
        transform_fn=transform_fn,
        displacement_fn=jax_md.space.periodic(box_size)[0]
    )

    # This seems like a common pattern
    top = jdna_top.from_oxdna_file(input_dir / "sys.top")
    def obj_energy_fn_builder(params: jdna_types.Params) -> callable:
        return jax.vmap(
            lambda trajectory: energy_fn_builder_fn(params)(
                trajectory.rigid_body,
                seq=jnp.array(top.seq),
                bonded_neighbors=top.bonded_neighbors,
                unbonded_neighbors=top.unbonded_neighbors.T,
            )
        )

    # This is a convenience wrapper just to use for filtering the observables
    # for the type we desire. Observables looks generic, while some objective
    # like the difftre expects trajectories. To keep generality we probably want
    # some way to tag observables, so we can filter relevant ones.
    #
    # For this difftre has been modified to filter SimulatorTrajectory for its
    # building of the combined trajectory, enabling us to pass more observables.
    class EnergyInfo(pd.DataFrame):
        pass

    # Create a simple class for running the simulator remotely, while also
    # modifying the run function to return both trajectory and energy data, for
    # use with the melting temp objective. The chex.dataclass decorator is not
    # compatible with ray actors, so we keep the underlying simulator as an
    # internal attribute.
    @ray.remote
    class RaySimulator:
        def __init__(self, **kwargs):
            self.simulator = oxdna.oxDNASimulator(**kwargs)

        def run(self, params, meta_data=None):
            traj = self.simulator.run(params, meta_data)
            energy_df = EnergyInfo(read_energy(self.simulator.base_dir))
            # note we directly pass the python objects to objective here as
            # opposed to files.
            return traj, energy_df

        def get_hist(self):
            hist_file = self.simulator.base_dir / self.simulator.input_config["last_hist_file"]
            hist_df_columns = ["bind", "mindist", "unbiased"]
            hist_df = pd.read_csv(hist_file, names=hist_df_columns, sep="\s+", usecols=[0,1,3], skiprows=1) \
                .set_index(["bind", "mindist"])
            hist_df["unbiased_normed"] = hist_df["unbiased"] / hist_df["unbiased"].sum()
            return hist_df

        def update_weights(self, weights):
            weights_file = self.simulator.base_dir / self.simulator.input_config["weights_file"]
            weights.to_csv(weights_file, sep=" ", header=False)

    # Make a wrapper class to run all of these remote simulators as though they
    # are a single simulator, but implement the simulator interface that has
    # exposes function so it can be used in the optimizer
    class MultiRaySimulator:
        def __init__(self, simulators):
            self.simulators = simulators

        def run(self, params, meta_data=None):
            # Run all the simulators in parallel and wait for all to be finished
            # before gathering results here
            futures = [sim.run.remote(params, meta_data) for sim in self.simulators]
            results = ray.get(futures)
            # Flatten the list, [traj, energy, traj, energy, ...]
            observables = list(itertools.chain.from_iterable(results))
            # Prior to next run, update the umbrella weights based on the histograms
            self.update_weights()
            return observables

        def update_weights(self):
            hist = ray.get([simulator.get_hist.remote() for simulator in self.simulators])
            hist = pd.concat(hist).reset_index().groupby(["bind", "mindist"]).sum()
            weights = hist.query("unbiased_normed > 0").eval("weights = 1 / unbiased_normed")
            weights["weights"] /= weights["weights"].min()  # for numerical stability
            weights = weights[["weights"]]
            # fill in zeroed states
            weights = weights.reindex(hist.index, fill_value=0)
            # Update these in all simulators
            ray.get([simulator.update_weights.remote(weights) for simulator in self.simulators])

        def exposes(self):
            # each simulator returns 2 observables: traj and energy, but doesn't
            # really matter what they are called here, just they are unique
            return [f"obs-{i}" for i in range(2*len(self.simulators))]

    # Construct multi simulator, it expects a list of simulator actors to be
    # passed in
    multi_simulator = MultiRaySimulator([
        RaySimulator.options(num_cpus=1).remote(
            input_dir=input_dir,
            sim_type=jdna_types.oxDNASimulatorType.DNA1,
            energy_configs=energy_configs,
            source_path=oxdna_src,
        )
        for _ in range(num_sims)
    ])

    # Setup the melting temp function, loss and objective
    melting_temp_fn = MeltingTemp(
        rigid_body_transform_fn=transform_fn,
        sim_temperature=kT,  # in sim units
        temperature_range=kt_range,  # in sim units
        energy_config=energy_configs,
        energy_fn_builder=obj_energy_fn_builder,
    )

    def melting_temp_loss_fn(
        trajectory: jax_md.rigid_body.RigidBody,
        weights: jnp.ndarray,
        _energy_model: jdna_energy.base.ComposedEnergyFunction,
        opt_params: jdna_types.Params,
        observables: list,
    ) -> tuple[float, tuple[str, typing.Any]]:
        # The objective passes along all observables, we filter out just the
        # energy info from there, which is needed for melting temp calculation.
        e_info = pd.concat([i for i in observables if isinstance(i, EnergyInfo)])
        obs = melting_temp_fn(trajectory, e_info["bond"].to_numpy(), e_info["weight"].to_numpy(), opt_params)

        expected_melting_temp = jnp.dot(weights, obs).sum()
        loss = (expected_melting_temp - target_temp) ** 2
        loss = jnp.sqrt(loss)
        if not jnp.isfinite(loss):
            # There is no recovery from this...
            raise ValueError("Non-finite loss encountered.")
        return loss, (("melting_temp", expected_melting_temp), {})

    melting_temp_objective = jdna_objective.DiffTReObjective(
        name="prop_twist",
        required_observables=multi_simulator.exposes(),
        needed_observables=multi_simulator.exposes(), # do we need to supply both?
        logging_observables=["loss", "melting_temp", "neff"],
        grad_or_loss_fn=melting_temp_loss_fn,
        energy_fn_builder=obj_energy_fn_builder,
        opt_params=opt_params,
        min_n_eff_factor=0.95,
        beta=jnp.array(1 / kT, dtype=jnp.float64),
        n_equilibration_steps=0,
    )

    optimizer = jdna_optimization.SimpleOptimizer(
        objective=melting_temp_objective,
        simulator=multi_simulator,
        optimizer=optax.adam(learning_rate=learning_rate),
    )

    # Create loggers, we've added an AIM logger (https://aimstack.io/) for
    # logging to the tracking and visualization system they provide. For this,
    #  pip install aim
    # is required. We've also added a multi-logger that logs to any number of
    # loggers at once, here we use it to log to both the console and AIM.
    if use_aim:
        from jax_dna.ui.loggers.aim import AimLogger
        aim_logger = AimLogger()
        aim_logger.aim_run.set("learning_rate", learning_rate)
        aim_logger.aim_run.set("num_sims", num_sims)
        aim_logger.aim_run.set("opt_steps", opt_steps)
        aim_logger.aim_run.set("target_temp", target_temp)
        aim_logger.aim_run.set("input_dir", str(input_dir))
        aim_logger.aim_run.set("oxdna_src", str(oxdna_src))
    else:
        aim_logger = NullLogger()
    console_logger = ConsoleLogger()
    logger = MultiLogger([aim_logger, console_logger])
    # Aim has the concept of run parameters, add the learning rate to this one

    for i in range(opt_steps):
        state, opt_params, _ = optimizer.step(opt_params)

        for metric, value in optimizer.objective.logging_observables():
            logger.log_metric(metric, value, i)

        optimizer.post_step(state, opt_params)


if __name__ == "__main__":
    fire.Fire(main)
