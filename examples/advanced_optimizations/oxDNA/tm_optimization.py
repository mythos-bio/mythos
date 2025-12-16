"""An example of running a melting temperature simulation using oxDNA.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

python examples/advanced_optimizations/oxDNA/tm_optimization.py
"""
import logging
import typing
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import jax_md
import mythos.energy as jdna_energy
import mythos.energy.dna1 as jdna1_energy
import mythos.input.topology as jdna_top
import mythos.optimization.objective as jdna_objective
import mythos.optimization.optimization as jdna_optimization
import mythos.utils.types as jdna_types
import optax
import pandas as pd
import ray
from mythos.input import oxdna_input
from mythos.observables.melting_temp import MeltingTemp
from mythos.simulators import oxdna
from mythos.simulators.oxdna.utils import read_energy
from mythos.simulators.ray import RayMultiGangSimulation
from mythos.ui.loggers.console import ConsoleLogger
from mythos.ui.loggers.logger import NullLogger
from mythos.ui.loggers.multilogger import MultiLogger
from mythos.utils.units import get_kt, get_kt_from_string

jax.config.update("jax_enable_x64", True)
logging.basicConfig(level=logging.INFO)
logging.getLogger("jax").setLevel(logging.WARNING)


def main(
        num_sims: int = 10,
        learning_rate: float = 1e-3,
        opt_steps: int = 100,
        input_dir: str = "data/templates/tm-6bp-2op",
        target_temp: str | float = "31.2C",
        oxdna_src: str = "../oxDNA",
        use_aim: bool = False,
    ):
    input_dir = Path(input_dir)
    oxdna_src = Path(oxdna_src).resolve()
    if isinstance(target_temp, str):
        target_temp = get_kt_from_string(target_temp)
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

    top = jdna_top.from_oxdna_file(input_dir / "sys.top")
    energy_fn = jdna1_energy.create_default_energy_fn(
        topology=top,
        displacement_fn=jax_md.space.periodic(box_size)[0],
    ).with_noopt("ss_stack_weights", "ss_hb_weights", "kt"
    ).with_params(kt = kT)

    opt_params = energy_fn.opt_params()

    # This is a convenience wrapper just to use for filtering the observables
    # for the type we desire. Observables looks generic, while some objective
    # like the difftre expects trajectories. To keep generality we probably want
    # some way to tag observables, so we can filter relevant ones.
    #
    # For this difftre has been modified to filter SimulatorTrajectory for its
    # building of the combined trajectory, enabling us to pass more observables.
    class EnergyInfo(pd.DataFrame):
        pass

    class HistogramInfo(pd.DataFrame):
        pass

    class oxDNAUmbrellaSampler(oxdna.oxDNASimulator):
        exposed_observables: typing.ClassVar[list[str]] = ["trajectory", "energy_info", "histogram_info"]

        def run(self, params: jdna_types.Params, meta_data: typing.Any = None) -> jax_md.rigid_body.RigidBody:
            traj = super().run(params, meta_data)
            energy_df = EnergyInfo(read_energy(self.base_dir))
            hist_df = HistogramInfo(self.get_hist())
            return traj, energy_df, hist_df

        def update_weights(self, weights: pd.DataFrame) -> None:
            weights_file = self.base_dir / self.input_config["weights_file"]
            weights.to_csv(weights_file, sep=" ", header=False)

        def get_hist(self) -> pd.DataFrame:
            hist_file = self.base_dir / self.input_config["last_hist_file"]
            hist_df_columns = ["bind", "mindist", "unbiased"]
            hist_df = pd.read_csv(hist_file, names=hist_df_columns, sep="\s+", usecols=[0,1,3], skiprows=1) \
                .set_index(["bind", "mindist"])
            hist_df["unbiased_normed"] = hist_df["unbiased"] / hist_df["unbiased"].sum()
            return hist_df

    class oxDNAUmbrellaSamplerGang(RayMultiGangSimulation):
        def pre_run(self, *args, **kwargs) -> None:
            pass

        def post_run(self, observables: list[typing.Any], *args, **kwargs) -> list[typing.Any]:
            hist = pd.concat([i for i in observables if isinstance(i, HistogramInfo)])
            hist = hist.reset_index().groupby(["bind", "mindist"]).sum()
            weights = hist.query("unbiased_normed > 0").eval("weights = 1 / unbiased_normed")
            weights["weights"] /= weights["weights"].min()  # for numerical stability
            weights = weights[["weights"]]
            # fill in zeroed states
            weights = weights.reindex(hist.index, fill_value=0)
            # Update these in all simulators
            ray.get([simulator.call_async("update_weights", weights) for simulator in self.simulations])
            return observables

    multi_simulator = oxDNAUmbrellaSamplerGang.create(
        num_sims,
        oxDNAUmbrellaSampler,
        input_dir=input_dir,
        sim_type=jdna_types.oxDNASimulatorType.DNA1,
        energy_fn=energy_fn,
        source_path=oxdna_src,
    )

    # Setup the melting temp function, loss and objective
    melting_temp_fn = MeltingTemp(
        rigid_body_transform_fn=1, # not used
        sim_temperature=kT,  # in sim units
        temperature_range=kt_range,  # in sim units
        energy_fn=energy_fn
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
        logging_observables=["loss", "melting_temp", "neff"],
        grad_or_loss_fn=melting_temp_loss_fn,
        energy_fn=energy_fn,
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
        from mythos.ui.loggers.aim import AimLogger
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

        for metric, value in optimizer.objective.logging_observables().items():
            logger.log_metric(metric, value, i)

        optimizer.post_step(state, opt_params)


if __name__ == "__main__":
    fire.Fire(main)
