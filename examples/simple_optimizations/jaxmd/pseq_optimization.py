"""Probabilistic sequence optimization using DiffTRe with Gumbel softmax annealing.

Optimizes the DNA sequence of a single-stranded system to achieve a target
end-to-end distance (or radius of gyration). Uses JaxMD simulator with the
DiffTRe algorithm from ``mythos.optimization.objective`` and the
``SimpleOptimizer`` from ``mythos.optimization.optimization``.

The optimizer updates raw (logit) values for the probabilistic sequence.
After each step, a callback projects these values back to valid probability
distributions via ``softmax(values / temperature)``. The Gumbel softmax
temperature is annealed from high to low over the optimization, gradually
sharpening the distributions toward discrete (one-hot) sequences.

Important: This assumes that the current working directory is the root
directory of the repository. i.e. this file was invoked using:

``python -m examples.simple_optimizations.jaxmd.pseq_optimization``

or with arguments::

    python -m examples.simple_optimizations.jaxmd.pseq_optimization --system ss20 --target 2.0
    python -m examples.simple_optimizations.jaxmd.pseq_optimization --system ss100 --observable rg --target 5.0
"""

import argparse
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax_md
import mythos.energy.dna1 as jdna_energy
import mythos.input.sequence_constraints as jdna_sc
import mythos.input.sequence_dependence as jdna_seqdep
import mythos.input.topology as jdna_top
import mythos.input.trajectory as jdna_traj
import mythos.optimization.objective as jdna_objective
import mythos.optimization.optimization as jdna_optimization
import mythos.simulators.base as jdna_sim_base
import mythos.simulators.jax_md as jdna_jaxmd
import mythos.utils.constants as jdna_const
import numpy as np
import optax
from typing_extensions import override

jax.config.update("jax_enable_x64", True)


def pseq_to_argmax_sequence(
    pseq: tuple[jnp.ndarray, jnp.ndarray],
    sc: jdna_sc.SequenceConstraints,
) -> str:
    """Convert a probabilistic sequence to a discrete sequence string via argmax.

    Args:
        pseq: Tuple of (unpaired_pseq, bp_pseq).
        sc: Sequence constraints describing the pairing structure.

    Returns:
        A string representing the most likely sequence.
    """
    unpaired_pseq, bp_pseq = pseq
    seq_chars = ["?"] * sc.n_nucleotides

    # Fill in unpaired nucleotides
    for idx in range(sc.n_nucleotides):
        up_idx = int(sc.idx_to_unpaired_idx[idx])
        if up_idx >= 0:
            nt_idx = int(jnp.argmax(unpaired_pseq[up_idx]))
            seq_chars[idx] = jdna_const.DNA_ALPHA[nt_idx]

    # Fill in base-paired nucleotides
    for bp_idx in range(sc.n_bp):
        bp_type_idx = int(jnp.argmax(bp_pseq[bp_idx]))
        bp_type = jdna_const.BP_TYPES[bp_type_idx]  # e.g. "AT", "TA", "GC", "CG"
        nt1_idx, nt2_idx = int(sc.bps[bp_idx, 0]), int(sc.bps[bp_idx, 1])
        seq_chars[nt1_idx] = bp_type[0]
        seq_chars[nt2_idx] = bp_type[1]

    return "".join(seq_chars)


def e2e_distance_single(body, displacement_fn):
    """Compute end-to-end distance for a single state."""
    disp = displacement_fn(body.center[0], body.center[-1])
    return jax_md.space.distance(disp)


def rg_single(body, displacement_fn):
    """Compute radius of gyration for a single state."""
    R = body.center
    com = jnp.mean(R, axis=0)
    drs = jax.vmap(displacement_fn, in_axes=(0, None))(R, com)
    rs = jax_md.space.distance(drs)
    return jnp.sqrt(jnp.mean(jnp.sum(rs**2, axis=-1)))


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Optimize DNA sequence via DiffTRe with SimpleOptimizer.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="ss20",
        choices=["ss20", "ss100"],
        help="Single-stranded system to use (default: ss20)",
    )
    parser.add_argument(
        "--observable",
        type=str,
        default="e2e",
        choices=["e2e", "rg"],
        help="Observable to optimize: end-to-end distance or radius of gyration (default: e2e)",
    )
    parser.add_argument("--target", type=float, default=2.0, help="Target observable value in oxDNA units (default: 2.0)")
    parser.add_argument("--n-sim-steps", type=int, default=20_000, help="Simulation steps per iteration (default: 20000)")
    parser.add_argument("--n-opt-steps", type=int, default=50, help="Number of optimization steps (default: 50)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate (default: 0.05)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--temp-start", type=float, default=2.0, help="Starting Gumbel softmax temperature (default: 2.0)")
    parser.add_argument("--temp-end", type=float, default=0.1, help="Ending Gumbel softmax temperature (default: 0.1)")
    parser.add_argument("--min-neff", type=float, default=0.8, help="Minimum normalized n_eff before resimulating (default: 0.8)")
    parser.add_argument("--max-reweight-steps", type=int, default=10, help="Max optimizer steps before forced resimulation (default: 10)")
    return parser


def main():
    args = get_parser().parse_args()

    # =========================================================================
    # Load topology and initial configuration
    # =========================================================================
    experiment_dir = Path(f"data/templates/{args.system}")

    topology = jdna_top.from_oxdna_file(experiment_dir / "sys.top")
    initial_positions = (
        jdna_traj.from_file(
            experiment_dir / "init.conf",
            topology.strand_counts,
            is_5p_3p=False,
        )
        .states[0]
        .to_rigid_body()
    )

    n_nucleotides = topology.n_nucleotides

    # =========================================================================
    # Define sequence constraints
    # =========================================================================
    # Single-stranded systems have no base pairs — every nucleotide is
    # independently optimizable.
    bps = np.zeros((0, 2), dtype=np.int32)
    sc = jdna_sc.from_bps(n_nucleotides, bps)

    # Initialize pseq using the helper. dseq_to_pseq handles edge cases
    # (e.g. zero base pairs) by inserting dummy arrays needed for JAX tracing.
    init_pseq = jdna_sc.dseq_to_pseq(topology.seq, sc)

    # =========================================================================
    # Setup energy function
    # =========================================================================
    experiment_config, _ = jdna_energy.default_configs()

    dt = experiment_config["dt"]
    kT = experiment_config["kT"]
    diff_coef = experiment_config["diff_coef"]
    rot_diff_coef = experiment_config["rot_diff_coef"]

    gamma = jax_md.rigid_body.RigidBody(
        center=jnp.array([kT / diff_coef], dtype=jnp.float64),
        orientation=jnp.array([kT / rot_diff_coef], dtype=jnp.float64),
    )
    mass = jax_md.rigid_body.RigidBody(
        center=jnp.array([experiment_config["nucleotide_mass"]], dtype=jnp.float64),
        orientation=jnp.array([experiment_config["moment_of_inertia"]], dtype=jnp.float64),
    )

    displacement_fn = jax_md.space.free()[0]

    # Load sequence-specific stacking and hydrogen-bonding weights.
    # These are important for sequence optimization — the default
    # sequence-averaged weights would not capture base-pair-dependent
    # interactions that drive meaningful sequence changes.
    ss_weights = jdna_seqdep.read_ss_weights(
        Path("data/seq-specific/seq_oxdna1.txt")
    )

    # Create the energy function with sequence-specific weights baked in.
    # Only pseq needs to appear in opt_params (the part we differentiate
    # w.r.t.).
    energy_fn = jdna_energy.create_default_energy_fn(
        topology=topology,
        displacement_fn=displacement_fn,
    ).with_params(
        pseq_constraints=sc,
        **ss_weights
    )

    # =========================================================================
    # Initial optimization parameters and temperature schedule
    # =========================================================================
    # Start from uniform logits (large constant, like the reference). The
    # softmax projection in the callback normalizes these to valid pseq
    # distributions each step.
    init_unpaired_pseq, init_bp_pseq = init_pseq
    init_logits = (
        jnp.full(init_unpaired_pseq.shape, 100.0, dtype=jnp.float64),
        jnp.full(init_bp_pseq.shape, 100.0, dtype=jnp.float64),
    )
    # Project initial logits to pseq via softmax at the starting temperature
    temperatures = np.linspace(args.temp_start, args.temp_end, args.n_opt_steps)
    opt_params = {"pseq": (
        jax.nn.softmax(init_logits[0] / temperatures[0]),
        jax.nn.softmax(init_logits[1] / temperatures[0]),
    )}

    # =========================================================================
    # Build the simulator (wrapped for SimpleOptimizer protocol)
    # =========================================================================
    jaxmd_simulator = jdna_jaxmd.JaxMDSimulator(
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
        neighbors=jdna_jaxmd.NoNeighborList(
            unbonded_nbrs=topology.unbonded_neighbors,
        ),
    )

    # Wrap JaxMDSimulator to conform to the Simulator protocol expected by
    # SimpleOptimizer: run(opt_params, **state) -> SimulatorOutput.
    key = jax.random.key(args.seed)

    @chex.dataclass(frozen=True, kw_only=True)
    class JaxMDSimulatorWrapper(jdna_sim_base.Simulator):
        """Wraps JaxMDSimulator for use with SimpleOptimizer."""

        inner: jdna_jaxmd.JaxMDSimulator
        init_state: jax_md.rigid_body.RigidBody
        n_steps: int
        key: jax.random.PRNGKey

        @override
        def run(self, *_args, key=None, **_kwargs) -> jdna_sim_base.SimulatorOutput:
            key = key if key is not None else self.key
            key, subkey = jax.random.split(key)
            # First positional arg from SimpleOptimizer is opt_params
            sim_opt_params = _args[0] if _args else {}
            traj = self.inner.run(sim_opt_params, self.init_state, self.n_steps, subkey)
            return jdna_sim_base.SimulatorOutput(
                observables=[traj],
                state={"key": key},
            )

    simulator = JaxMDSimulatorWrapper(
        name="jaxmd-sim",
        inner=jaxmd_simulator,
        init_state=initial_positions,
        n_steps=args.n_sim_steps,
        key=key,
    )

    # =========================================================================
    # Select observable and build DiffTRe objective
    # =========================================================================
    if args.observable == "e2e":
        single_obs_fn = lambda body: e2e_distance_single(body, displacement_fn)
        obs_name = "E2E"
    else:
        single_obs_fn = lambda body: rg_single(body, displacement_fn)
        obs_name = "Rg"

    target_obs = jnp.array(args.target, dtype=jnp.float64)
    obs_key = simulator.exposes()[0]  # trajectory observable key

    def loss_fn(
        traj,
        weights: jnp.ndarray,
        _energy_model,
        _opt_params,
        _observables,
    ) -> tuple[float, tuple[tuple[str, jnp.ndarray], dict]]:
        """Reweighted observable loss for DiffTRe."""
        per_state_obs = jax.vmap(single_obs_fn)(traj)
        expected_obs = jnp.dot(weights, per_state_obs)
        loss = (expected_obs - target_obs) ** 2
        return loss, ((obs_name, expected_obs), {})

    objective = jdna_objective.DiffTReObjective(
        name="pseq_opt",
        required_observables=(obs_key,),
        logging_observables=("loss", "neff", obs_name),
        grad_or_loss_fn=loss_fn,
        energy_fn=energy_fn,
        min_n_eff_factor=args.min_neff,
        n_equilibration_steps=0,
        max_valid_opt_steps=args.max_reweight_steps,
    )

    # =========================================================================
    # Build optimizer and run
    # =========================================================================
    simple_optimizer = jdna_optimization.SimpleOptimizer(
        objective=objective,
        simulator=simulator,
        optimizer=optax.adam(learning_rate=args.lr),
    )

    print(f"System: {args.system} ({n_nucleotides} nt, {sc.n_unpaired} unpaired, {sc.n_bp} bp)")
    print(f"Observable: {obs_name}, Target: {args.target}")
    print(f"Optimization: {args.n_opt_steps} steps, {args.n_sim_steps} sim steps, lr={args.lr}")
    print(f"Gumbel temperature: {args.temp_start} -> {args.temp_end}")
    print(f"DiffTRe: min_neff={args.min_neff}, max_reweight_steps={args.max_reweight_steps}")
    print()

    def log_callback(optimizer_output, step):
        """Project pseq via Gumbel softmax annealing and print progress.

        After each Adam update, the raw pseq values may no longer be valid
        probability distributions. This callback projects them back via
        softmax(values / temperature), where temperature decreases over
        steps.
        """
        # Project to valid distributions via softmax with annealed temperature
        temp = temperatures[min(step + 1, len(temperatures) - 1)]
        up, bp = optimizer_output.opt_params["pseq"]
        projected_pseq = (jax.nn.softmax(up / temp), jax.nn.softmax(bp / temp))
        optimizer_output = optimizer_output.replace(
            opt_params={"pseq": projected_pseq},
        )

        obs = optimizer_output.observables.get("pseq_opt", {})
        loss = obs.get("loss", float("nan"))
        measured = obs.get(obs_name, float("nan"))
        neff = obs.get("neff", float("nan"))
        pseq = projected_pseq
        seq_str = pseq_to_argmax_sequence(pseq, sc)
        seq_display = f"{seq_str[:10]}...{seq_str[-10:]}" if len(seq_str) > 30 else seq_str
        print(
            f"Step {step:3d} | Loss: {loss:.4f} | {obs_name}: {measured:.4f} | "
            f"n_eff: {neff:.3f} | Temp: {temp:.3f} | Seq: {seq_display}"
        )
        return optimizer_output, True  # return projected output for next step

    output = simple_optimizer.run(opt_params, n_steps=args.n_opt_steps, callback=log_callback)

    # =========================================================================
    # Final result
    # =========================================================================
    final_pseq = output.opt_params["pseq"]
    final_seq = pseq_to_argmax_sequence(final_pseq, sc)
    print(f"\nOptimized sequence: {final_seq}")


if __name__ == "__main__":
    main()
