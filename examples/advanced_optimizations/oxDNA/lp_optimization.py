"""An example of running persistence length optimization using oxDNA and DiffTRe."""

import logging
import typing
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import jax_md
import mythos.energy.dna1 as dna1_energy
import mythos.input.topology as jdna_top
import mythos.optimization.objective as jdna_objective
import mythos.optimization.optimization as jdna_optimization
from mythos.simulators.oxdna.oxdna import oxDNASimulator
from mythos.simulators.ray import RayMultiSimulation
import mythos.utils.types as jdna_types
import optax
import ray
from mythos.input import oxdna_input
from mythos.observables import base
from mythos.observables.persistence_length import PersistenceLength
from mythos.ui.loggers.console import ConsoleLogger
from mythos.ui.loggers.logger import NullLogger
from mythos.ui.loggers.multilogger import MultiLogger
from mythos.utils.units import get_kt_from_string
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


# Logging configurations =======================================================
logging.basicConfig(level=logging.INFO)
logging.getLogger("jax").setLevel(logging.WARNING)

TARGET_LP = 40.0  # nm, or should this be 47.5?

def main(
    num_sims:int=10,
    learning_rate:float=1e-3,
    opt_steps:int=100,
    target_lp:float=TARGET_LP,
    input_dir: str = "data/templates/simple-helix-60bp-oxdna1",
    oxdna_path: str = "../oxDNA",
    use_aim: bool = False
):
    # The coordination of objectives and simulators is done through Ray actors.
    # So we need to initialize a ray server
    ray.init(runtime_env={"env_vars": {"JAX_ENABLE_X64": "True", "JAX_PLATFORM_NAME": "cpu"}})

    oxdna_path = Path(oxdna_path).resolve()
    input_dir = Path(input_dir).resolve()
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
    energy_fn = dna1_energy.create_default_energy_fn(
        topology=top,
        displacement_fn=jax_md.space.periodic(box_size)[0],
    ).with_noopt("ss_stack_weights", "ss_hb_weights", "kt"
    ).with_params(kt = kT)

    transform_fn = energy_fn.energy_fns[0].transform_fn  # all are same

    opt_params = energy_fn.opt_params()

    multi_simulator = RayMultiSimulation.create(
        num_sims,
        oxDNASimulator,
        input_dir=input_dir,
        sim_type=jdna_types.oxDNASimulatorType.DNA1,
        energy_fn=energy_fn,
        source_path=oxdna_path,
    )

    lp_fn = PersistenceLength(
        rigid_body_transform_fn=transform_fn,
        displacement_fn=jax_md.space.periodic(box_size)[0],
        quartets=base.get_duplex_quartets(int(top.n_nucleotides / 2)),
        truncate=40,
    )

    def lp_loss_fn(
        traj: jax_md.rigid_body.RigidBody,
        weights: jnp.ndarray,
        *_args,
        **_kwargs,
    ) -> tuple[float, tuple[str, typing.Any]]:
        fit_lp = lp_fn(traj, weights=weights)
        loss = (fit_lp - target_lp) ** 2
        loss = jnp.sqrt(loss)
        return loss, (("persistence_length", fit_lp), {})

    persistence_length_objective = jdna_objective.DiffTReObjective(
        name="persistence_length",
        required_observables=multi_simulator.exposes(),
        logging_observables=["loss", "persistence_length", "neff"],
        grad_or_loss_fn=lp_loss_fn,
        energy_fn=energy_fn,
        opt_params=opt_params,
        min_n_eff_factor=0.95,
        beta=jnp.array(1 / kT, dtype=jnp.float64),
        n_equilibration_steps=0,
        max_valid_opt_steps=10,
    )
    # ==========================================================================

    opt = jdna_optimization.SimpleOptimizer(
        objective=persistence_length_objective,
        simulator=multi_simulator,
        optimizer = optax.adam(learning_rate=1e-3),
    )
    # ==========================================================================

    if use_aim:
        from mythos.ui.loggers.aim import AimLogger
        aim_logger = AimLogger()
        aim_logger.aim_run.set("lp_target", target_lp)
        aim_logger.aim_run.set("lp_learning_rate", learning_rate)
        aim_logger.aim_run.set("lp_num_sims", num_sims)
        aim_logger.aim_run.set("lp_input_dir", str(input_dir))
        aim_logger.aim_run.set("lp_oxdna_path", str(oxdna_path))
    else:
        aim_logger = NullLogger()
    console_logger = ConsoleLogger()
    logger = MultiLogger([aim_logger, console_logger])


    # Run optimization =========================================================
    for i in tqdm(range(opt_steps), desc="Optimizing"):
        opt_state, opt_params, grads = opt.step(opt_params)

        for (name, value) in opt.objective.logging_observables().items():
            logger.log_metric(name, value, step=i)

        opt = opt.post_step(
            optimizer_state=opt_state,
            opt_params=opt_params,
        )

if __name__=="__main__":
    fire.Fire(main)
