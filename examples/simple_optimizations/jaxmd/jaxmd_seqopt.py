"""An example of running a simple optimization using jax_md.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

``python -m examples.simple_optimizations.jaxmd.jaxmd``
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dna.energy.dna1 as jdna_energy
from jax_dna.input.sequence_dependence import read_ss_weights
import jax_dna.input.topology as jdna_top
import jax_dna.input.trajectory as jdna_traj
import jax_dna.observables as jdna_obs
from jax_dna.observables.base import get_duplex_quartets
from jax_dna.observables.persistence_length import PersistenceLength
import jax_dna.simulators.jax_md as jdna_jaxmd
import jax_dna.utils.types as jdna_types
import jax_md
import numpy as np
import optax
from jax_dna.input.sequence_constraints import dseq_to_pseq, from_bps
from jax_dna.optimization.objective import DiffTReObjective
from jax_dna.optimization.optimization import SimpleOptimizer

jax.config.update("jax_enable_x64", True)

def main():
    # configs specific to this file
    run_config = {
        "n_sim_steps": 50_000,
        "n_opt_steps": 50,
        "learning_rate": 0.0001,
    }

    experiment_dir = Path("data/templates/simple-helix-60bp-oxdna1").resolve()

    topology = jdna_top.from_oxdna_file(experiment_dir / "sys.top")
    initial_positions = (
        jdna_traj.from_file(
            experiment_dir / "init.conf",
            topology.strand_counts,
            is_oxdna=False,
        )
        .states[0]
        .to_rigid_body()
    )

    experiment_config, _ = jdna_energy.default_configs()

    dt = experiment_config["dt"]
    kT = experiment_config["kT"]
    diff_coef = experiment_config["diff_coef"]
    rot_diff_coef = experiment_config["rot_diff_coef"]

    # These are special values for the jax_md simulator
    gamma = jax_md.rigid_body.RigidBody(
        center=jnp.array([kT / diff_coef], dtype=jnp.float64),
        orientation=jnp.array([kT / rot_diff_coef], dtype=jnp.float64),
    )
    mass = jax_md.rigid_body.RigidBody(
        center=jnp.array([experiment_config["nucleotide_mass"]], dtype=jnp.float64),
        orientation=jnp.array([experiment_config["moment_of_inertia"]], dtype=jnp.float64),
    )

    transform_fn = jdna_energy.default_transform_fn()

    # The jax_md simulator needs an energy function. We can use the default
    # energy functions and configurations for dna1 simulations. For more
    # information on energy functions and configurations, see the documentation.
    displacement_fn = jax_md.space.free()[0]

    bps = jnp.array([[i, topology.n_nucleotides-i-1] for i in range(topology.n_nucleotides // 2)], dtype=jnp.int32)
    sc = from_bps(topology.n_nucleotides, bps)
    pseq = dseq_to_pseq(topology.seq, sc)

    ss_params = read_ss_weights("ss_weights.txt")

    energy_fn = jdna_energy.create_default_energy_fn(
        topology=topology,
        displacement_fn=displacement_fn,
    ).with_params(
        pseq=pseq, pseq_constraints=sc, **ss_params,
    )

    simulator = jdna_jaxmd.JaxMDSimulator(
        energy_fn=energy_fn,
        simulator_params=jdna_jaxmd.StaticSimulatorParams(
            seq=jnp.array(topology.seq),
            mass=mass,
            bonded_neighbors=topology.bonded_neighbors,
            checkpoint_every=500,
            dt=dt,
            kT=kT,
            gamma=gamma,
        ),
        space=jax_md.space.free(),
        simulator_init=jax_md.simulate.nvt_langevin,
        neighbors=jdna_jaxmd.NoNeighborList(unbonded_nbrs=topology.unbonded_neighbors),
    )

    # ==========================================================================
    # Up until this point this is identical to running a simulation, save for
    # the definition of `params`, to run an optimization we need to define a few
    # more things: a loss function, a function that computes the loss given a
    # set of parameters, and  an optimizer
    # ==========================================================================

    # The ObservableLossFn class is a convenience wrapper for computing the the
    # loss of an observable. In this case, we are using the propeller twist and
    # the loss is squared error. the ObservableLossFn class implements __call__
    # that takes the output of the simulation, the target, and weights and
    # returns the loss and the measured observable.

    lp = PersistenceLength(
        rigid_body_transform_fn=transform_fn,
        displacement_fn=displacement_fn,
        quartets=get_duplex_quartets(int(topology.n_nucleotides / 2)),
        truncate=40,
    )

    def lp_loss(traj, weights, *args, **kwargs):
        obs = lp(traj, weights=weights)
        loss = jnp.sqrt((obs - 40.0)**2)
        return loss, (("lp", obs), {})

    # we just optimmize over the sequence, not all of the other params
    # considered for optimization by default in energy_fn.opt_params()
    params = {"pseq": pseq}

    objective = DiffTReObjective(
        name="lp_objective",
        energy_fn=energy_fn,
        grad_or_loss_fn=lp_loss,
        logging_observables=["lp"],
        required_observables=["traj"],
        needed_observables=["traj"],
        opt_params=params,
        beta=jnp.array(1 / kT, dtype=jnp.float64),
        n_equilibration_steps=100,
        max_valid_opt_steps=10,
    )

    class MySimulator:
        def __init__(self, simulator, initial_positions, key, nsteps):
            self.simulator = simulator
            self.initial_positions = initial_positions
            self.steps = nsteps
            self.key = key
            def sim_run(params, key):
                return simulator.run(params, self.initial_positions, self.steps, key)
            self.sim_fun = jax.jit(sim_run)

        def exposes(self):
            return ["traj"]

        def run(self, params):
            self.key, subkey = jax.random.split(self.key)
            traj = self.sim_fun(params, subkey)
            return [traj]


    optimizer = SimpleOptimizer(
        simulator=MySimulator(simulator, initial_positions, jax.random.key(1234), run_config["n_sim_steps"]),
        objective=objective,
        optimizer=optax.adam(learning_rate=run_config["learning_rate"]),
    )

    # Gumbel temp schedule for the probabilistic sequence
    gumbel_temp_schedule = jnp.linspace(0.8, 0.1, run_config["n_opt_steps"])

    def normalize_pseq(pseq: jdna_types.Probabilistic_Sequence, temp: float) -> jdna_types.Probabilistic_Sequence:
        """Turn logits into probabilities using a gumbel-softmax at given temp."""
        up = jax.nn.softmax(pseq[0] / temp)
        bp = jax.nn.softmax(pseq[1] / temp)
        return up, bp

    for i in range(run_config["n_opt_steps"]):
        # normalize according to schedule at each step
        params["pseq"] = normalize_pseq(params["pseq"], gumbel_temp_schedule[i])
        print(f"    {params['pseq']}")
        opt_state, params, grads = optimizer.step(params)
        print(f"grads: {grads}")
        for obs, val in objective.logging_observables():
            print(f"Step {i}: {obs}: {val}")
        print(f" p  {params['pseq']}")
        optimizer.post_step(opt_state, params)


if __name__=="__main__":
    main()
