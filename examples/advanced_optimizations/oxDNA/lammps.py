"""An example of running a simulation using LAMMPS for oxDNA model optimization.

Important: This requires the fire library to be installed, use ``pip install
fire``.

Running this example:

``python -m examples.advanced_optimizations.oxDNA.lammps``

See help message for more details.

``python -m examples.advanced_optimizations.oxDNA.lammps --help``
"""


import logging
from pathlib import Path
import typing

import fire
import jax
import jax.numpy as jnp
import mythos.energy.dna1 as dna1_energy
import mythos.observables as jd_obs
import mythos.optimization.objective as jdna_objective
import mythos.optimization.optimization as jdna_optimization
import mythos.optimization.simulator as jdna_simulator
import mythos.utils.types as jdna_types
import jax_md
import optax
from mythos.input import topology
from mythos.simulators.lammps.lammps_oxdna import LAMMPSoxDNASimulator
from mythos.ui.loggers.console import ConsoleLogger
from mythos.ui.loggers.logger import NullLogger
from mythos.ui.loggers.multilogger import MultiLogger

jax.config.update("jax_enable_x64", True)
TARGET = jd_obs.propeller.TARGETS["oxDNA"]

def main(
        n_opt_steps:int=25,
        target:float=TARGET,
        use_aim: bool = False,
        input_dir: str = "data/templates/simple-helix-60bp-oxdna1-lammps",
        learning_rate: float = 1e-3
    ):
    logging.basicConfig(level=logging.INFO)
    input_dir = Path(input_dir).resolve()
    _, sim_config = dna1_energy.default_configs()
    kT = 0.1  # Must match that used in LAMMPS

    top = topology.from_oxdna_file(input_dir / "sys.top")
    energy_fn = dna1_energy.create_default_energy_fn(
        topology=top,
    ).with_noopt(
        "ss_stack_weights", "ss_hb_weights"
    ).without_terms(
        "BondedExcludedVolume"
    )

    simulator = LAMMPSoxDNASimulator(
        input_dir=input_dir,
        energy_fn=energy_fn,
    )

    def simulator_fn(
        params: jdna_types.Params,
        _meta: jdna_types.MetaData,
    ) -> tuple[str, str]:
        return [simulator.run(params)]

    obs_trajectory = "trajectory"

    trajectory_simulator = jdna_simulator.BaseSimulator(
        name="oxdna-sim",
        fn=simulator_fn,
        exposes = [obs_trajectory],
        meta_data = {},
    )

    prop_twist_fn = jd_obs.propeller.PropellerTwist(
        rigid_body_transform_fn=energy_fn.energy_fns[0].transform_fn,
        h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
    )

    def prop_twist_loss_fn(
        traj: jax_md.rigid_body.RigidBody,
        weights: jnp.ndarray,
        *_args,
        **_kwargs,
    ) -> tuple[float, tuple[str, typing.Any]]:
        obs = prop_twist_fn(traj)
        expected_prop_twist = jnp.dot(weights, obs)
        loss = (expected_prop_twist - target)**2
        loss = jnp.sqrt(loss)
        return loss, (("prop_twist", expected_prop_twist), {})

    opt_params = energy_fn.opt_params()

    propeller_twist_objective = jdna_objective.DiffTReObjective(
        name = "DiffTRe",
        required_observables = [obs_trajectory],
        needed_observables = [obs_trajectory],
        logging_observables = ["loss", "prop_twist"],
        grad_or_loss_fn = prop_twist_loss_fn,
        energy_fn = energy_fn,
        opt_params = opt_params,
        min_n_eff_factor = 0.95,
        beta = jnp.array(1/kT),
        n_equilibration_steps = 1000,
        max_valid_opt_steps=100,
    )

    opt = jdna_optimization.SimpleOptimizer(
        objective=propeller_twist_objective,
        simulator=trajectory_simulator,
        optimizer = optax.adam(learning_rate=learning_rate),
    )

    if use_aim:
        from mythos.ui.loggers.aim import AimLogger
        aim_run = AimLogger(experiment="oxdna-lammps-propeller-twist")
        aim_logger = AimLogger(aim_run = aim_run)
    else:
        aim_logger = NullLogger()
    console_logger = ConsoleLogger()
    logger = MultiLogger([aim_logger, console_logger])
    # Aim has the concept of run parameters, add the learning rate to this one

    for i in range(n_opt_steps):
        opt_state, opt_params, _ = opt.step(opt_params)

        log_values = propeller_twist_objective.logging_observables()
        for (name, value) in log_values:
            logger.log_metric(f"{name}", value, step=i)

        opt = opt.post_step(
            optimizer_state=opt_state,
            opt_params=opt_params,
        )


if __name__=="__main__":
    fire.Fire(main)