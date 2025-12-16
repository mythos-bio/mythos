"""An example of running a multi trajectory simulation using oxDNA.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

``python -m examples.simulations.oxdna.oxDNA``
"""
import functools
import logging
import os
from pathlib import Path
import shutil
import typing
import warnings
import jax
import jax.numpy as jnp
import jax_md
import optax
import ray
from tqdm import tqdm

import mythos.energy as jdna_energy
import mythos.energy.dna1 as dna1_energy
import mythos.input.toml as toml_reader
import mythos.input.topology as jdna_top
import mythos.input.trajectory as jdna_traj
import mythos.input.tree as jdna_tree
import mythos.observables as jd_obs
import mythos.optimization.simulator as jdna_simulator
import mythos.optimization.objective as jdna_objective
import mythos.optimization.optimization as jdna_optimization
import mythos.simulators.oxdna as oxdna
import mythos.simulators.io as jdna_sio
import mythos.utils.types as jdna_types
import mythos.ui.loggers.console as console_logger


jax.config.update("jax_enable_x64", True)


# Logging configurations =======================================================
logging.basicConfig(level=logging.DEBUG, filename="opt.log", filemode="w")
objective_logging_config = {
    "level":logging.DEBUG,
    "filename":"objective.log",
    "filemode":"w",
}
simulator_logging_config = objective_logging_config | {"filename": "simulator.log"}
# ==============================================================================


# To combine the gradients of multiple objectives, we can use a mean, however
# this example only has one objective, so it will remain unchanged.
def tree_mean(trees:tuple[jdna_types.PyTree]) -> jdna_types.PyTree:
    if len(trees) <= 1:
        return trees[0]
    return jax.tree.map(lambda *x: jnp.mean(jnp.stack(x)), *trees)


def main():

    # The coordination of objectives and simulators is done through Ray actors.
    # So we need to initialize a ray server
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        runtime_env={
            "env_vars": {
                "JAX_ENABLE_X64": "True",
                "JAX_PLATFORM_NAME": "cpu",
            }
        }
    )


    # Input configuration ======================================================
    optimization_config = {
        "n_steps": 20,
        "oxdna_build_threads": 4,
        "log_every": 10,
        "n_oxdna_runs": 3,
        "oxdna_use_cached_build": False,
    }

    simulator_logging_config = {
        "filename": "simulator.log",
        "level": logging.DEBUG,
        "filemode": "w",
    }

    kT = toml_reader.parse_toml("mythos/input/dna1/default_simulation.toml")["kT"]
    geometry = toml_reader.parse_toml("mythos/input/dna1/default_energy.toml")["geometry"]

    template_dir = Path("data/templates/simple-helix")
    topology_fname = template_dir / "sys.top"

    cwd = Path(os.getcwd())
    # ==========================================================================


    # Energy Function ==========================================================
    energy_fns = dna1_energy.default_energy_fns()
    energy_configs = []
    opt_params = []

    for ec in dna1_energy.default_energy_configs():
        # We are only interested in the stacking configuration
        # However we don't want to optimize ss_stack_weights and kt
        if isinstance(ec, dna1_energy.StackingConfiguration):
            ec = ec.replace(
                non_optimizable_required_params=(
                    "ss_stack_weights",
                    "kt",
                )
            )
            opt_params.append(ec.opt_params)
            energy_configs.append(ec)
        else:
            energy_configs.append(ec)
            opt_params.append({})

    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    energy_fn_builder_fn = jdna_energy.energy_fn_builder(
        energy_fns=energy_fns,
        energy_configs=energy_configs,
        transform_fn=transform_fn,
    )

    top = jdna_top.from_oxdna_file(topology_fname)
    def energy_fn_builder(params: jdna_types.Params) -> callable:
        return jax.vmap(
            lambda trajectory: energy_fn_builder_fn(params)(
                trajectory.rigid_body,
                seq=jnp.array(top.seq),
                bonded_neighbors=top.bonded_neighbors,
                unbonded_neighbors=top.unbonded_neighbors.T,
            )
            / top.n_nucleotides
        )
    # ==========================================================================


    # Simulators ================================================================
    sim_outputs_dir = cwd / "sim_outputs"
    sim_outputs_dir.mkdir(parents=True, exist_ok=True)

    def make_simulator(id:str, oxdna_bin: Path = None, oxdna_src: Path = None) -> jdna_simulator.BaseSimulator:

        simulator = oxdna.oxDNASimulator(
            input_dir=template_dir,
            sim_type=jdna_types.oxDNASimulatorType.DNA1,
            energy_configs=energy_configs,
            n_build_threads=optimization_config["oxdna_build_threads"],
            logger_config=simulator_logging_config | {"filename": f"simulator_{id}.log"},
            source_path=oxdna_src,
            binary_path=oxdna_bin,
            ignore_params=bool(oxdna_bin),  # we handle the build using same params
        )

        def simulator_fn(
            params: jdna_types.Params,
            meta: jdna_types.MetaData,
        ) -> tuple[str, str]:
            return [simulator.run(params)]

        return jdna_simulator.SimulatorActor.remote(
            name=id,
            fn=simulator_fn,
            exposes=[f"traj-{id}"],
            meta_data={},
        )


    sim_ids = [f"sim{i}" for i in range(optimization_config["n_oxdna_runs"])]
    traj_ids = [f"traj-{id}" for id in sim_ids]

    # If we share the same source path (e.g. we run on a single machine or a
    # cluster with a shared file system), we can pre-build the binary once and
    # provide the location of that binary to each simulator. Note it is the
    # users responsibility to call the build function at each step.
    if optimization_config["oxdna_use_cached_build"]:
        builder = oxdna.oxDNASimulator(
            source_path="../oxDNA/",
            input_dir=template_dir,
            energy_configs=energy_configs,
            sim_type=None
        )
        simulators = [make_simulator(id, oxdna_bin=builder.binary_path) for id in sim_ids]
    else:
        simulators = [make_simulator(id, oxdna_src="../oxDNA/") for id in sim_ids]
    # ==========================================================================



    # Objective ================================================================
    prop_twist_fn = jd_obs.propeller.PropellerTwist(
        rigid_body_transform_fn=transform_fn,
        h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]]),
    )

    def prop_twist_loss_fn(
        traj: jax_md.rigid_body.RigidBody,
        weights: jnp.ndarray,
        energy_model: jdna_energy.base.ComposedEnergyFunction,
        _opt_params,
        _observables,
    ) -> tuple[float, tuple[str, typing.Any]]:
        obs = prop_twist_fn(traj)
        expected_prop_twist = jnp.dot(weights, obs)
        loss = (expected_prop_twist - jd_obs.propeller.TARGETS["oxDNA"]) ** 2
        loss = jnp.sqrt(loss)
        return loss, (("prop_twist", expected_prop_twist), {})

    propeller_twist_objective = jdna_objective.DiffTReObjectiveActor.remote(
        name="prop_twist",
        required_observables=traj_ids,
        needed_observables=traj_ids,
        logging_observables=["loss", "prop_twist", "neff"],
        grad_or_loss_fn=prop_twist_loss_fn,
        energy_fn_builder=energy_fn_builder,
        opt_params=opt_params,
        min_n_eff_factor=0.95,
        beta=jnp.array(1 / kT, dtype=jnp.float64),
        n_equilibration_steps=0,
        max_valid_opt_steps=10,
    )
    # ==========================================================================



    # Logger ===================================================================
    logger = console_logger.ConsoleLogger()

    # ==========================================================================


    # Optimization =============================================================
    objectives = [propeller_twist_objective]

    opt = jdna_optimization.Optimization(
        objectives=objectives,
        simulators=simulators,
        optimizer = optax.adam(learning_rate=1e-3),
        aggregate_grad_fn=tree_mean,
        logger=logger,
    )
    # ==========================================================================



    # Run optimization =========================================================
    for i in tqdm(range(optimization_config["n_steps"]), desc="Optimizing"):
        if optimization_config["oxdna_use_cached_build"]:
            builder.build(new_params=opt_params)

        opt_state, opt_params, _ = opt.step(opt_params)

        for objective in opt.objectives:
            log_values = ray.get(objective.logging_observables.remote())
            for (name, value) in log_values:
                logger.log_metric(name, value, step=i)

        opt = opt.post_step(
            optimizer_state=opt_state,
            opt_params=opt_params,
        )

    if optimization_config["oxdna_use_cached_build"]:
        builder.cleanup()

if __name__ == "__main__":
    main()
