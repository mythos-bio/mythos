"""An example of running a simulation using LAMMPS for oxDNA model optimization.

Important: This requires the fire library to be installed, use ``pip install
fire``.

Running this example:

``python -m examples.advanced_optimizations.oxDNA.lammps``

See help message for more details.

``python -m examples.advanced_optimizations.oxDNA.lammps --help``
"""


import functools
import logging
import typing

import fire
import jax
import jax.numpy as jnp
import jax_dna.energy as jdna_energy
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.observables as jd_obs
import jax_dna.optimization.objective as jdna_objective
import jax_dna.optimization.optimization as jdna_optimization
import jax_dna.optimization.simulator as jdna_simulator
from jax_dna.ui.loggers.logger import NullLogger
import jax_dna.utils.types as jdna_types
import jax_md
import optax
from jax_dna.input import topology
from jax_dna.simulators.lammps.lammps_oxdna import LAMMPSoxDNASimulator
from jax_dna.ui.loggers.console import ConsoleLogger
from jax_dna.ui.loggers.multilogger import MultiLogger

jax.config.update("jax_enable_x64", True)
TARGET = jd_obs.propeller.TARGETS["oxDNA"]

def main(
        n_opt_steps:int=25,
        target:float=TARGET,
        use_aim: bool = False,
        input_dir: str = "lammps_inputs",
        learning_rate: float = 1e-3
    ):
    logging.basicConfig(level=logging.INFO)

    simulation_config, energy_config = dna1_energy.default_configs()
    kT = simulation_config["kT"]

    energy_fns = dna1_energy.default_energy_fns()
    energy_fn_configs = dna1_energy.default_energy_configs()
    opt_params = []
    for ec in energy_fn_configs:
        opt_params.append(
            ec.opt_params
        )

    for op in opt_params:
        if "ss_stack_weights" in op:
            del op["ss_stack_weights"]
        if "eps_backbone" in op:
            del op["eps_backbone"]

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    energy_fn_builder_fn = jdna_energy.energy_fn_builder(
        energy_fns=energy_fns,
        energy_configs=energy_fn_configs,
        transform_fn=transform_fn,
    )

    topology_fname = "data/templates/simple-helix/sys.top"
    top = topology.from_oxdna_file(topology_fname)

    def energy_fn_builder(params: jdna_types.Params) -> callable:
        return jax.vmap(
            lambda trajectory: energy_fn_builder_fn(params)(
                trajectory.rigid_body,
                seq=jnp.array(top.seq),
                bonded_neighbors=top.bonded_neighbors,
                unbonded_neighbors=top.unbonded_neighbors.T,
            )
        )

    simulator = LAMMPSoxDNASimulator(
        input_dir=input_dir,
        energy_configs=energy_fn_configs,
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
        rigid_body_transform_fn=transform_fn,
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


    propeller_twist_objective = jdna_objective.DiffTReObjective(
        name = "DiffTRe",
        required_observables = [obs_trajectory],
        needed_observables = [obs_trajectory],
        logging_observables = ["loss", "prop_twist"],
        grad_or_loss_fn = prop_twist_loss_fn,
        energy_fn_builder = energy_fn_builder,
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
        from jax_dna.ui.loggers.aim import AimLogger
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