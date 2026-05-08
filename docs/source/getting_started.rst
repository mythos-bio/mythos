Getting Started
===============

.. _installation:

Installation
------------

We recommend using a fresh virtual environment via tools like `conda` or `uv`
with Python 3.11+:

.. code-block:: bash

    uv venv mythos --python 3.11
    source mythos/bin/activate


.. code-block:: bash

    conda create -y -n mythos python=3.11
    conda activate mythos

Depending on your hardware, you may want to install the GPU-accelerated version
of JAX. See the `JAX installation docs
<https://docs.jax.dev/en/latest/installation.html>`_ for details. If you don't
need GPU support, you can skip straight to installing ``mythos`` (which installs
the CPU version of JAX automatically).

Install ``mythos`` using pip:

.. code-block:: bash

    pip install git+https://github.com/mythos-bio/mythos.git

Some simulator backends have additional dependencies:

- **oxDNA**: install separately. See :doc:`simulators`.
- **GROMACS**: install separately, ensure ``gmx`` is on ``PATH``. See :doc:`simulators`.
- **LAMMPS**: install separately with ``CG-DNA`` package. See :doc:`simulators`.


Quick Overview
--------------

The two primary use cases for ``mythos`` are:

1. **Running differentiable simulations** using one of several backends (JAX-MD, oxDNA,
   GROMACS, LAMMPS). See :doc:`simulators` for the full list.
2. **Optimizing force field parameters** against target observable values, which can be ``top-down'' (experimentally derived) or ``bottom-up'' (derived from more fine-grained models).
   See :doc:`optimization` for the framework details.

Energy functions define the physics of the biomolecular model and typically house the
parameters being optimized (though optimization with respect to other simulation parameters, such as temperature, number of particles, etc. is possible). See :doc:`energy_functions` for currently available models and how to extend them or develop a bespoke model.

Observables compute measurable quantities (e.g. helical pitch, persistence length, membrane thickness, bond and angle distributions, etc.) from trajectories. Observables are packaged into objectives and fed into loss functions. See :doc:`observables` for the full catalog and API.

.. _basic-optimization:

Example: oxDNA Optimization with DiffTRe
-----------------------------------------

Below is a minimal example of optimizing oxDNA energy function parameters
against a propeller twist target using the DiffTRe algorithm. This
demonstrates how the core pieces — simulator, energy function, observable,
loss function, objective, and optimizer — fit together.

.. code-block:: python

    import functools
    import typing
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import optax

    import mythos.energy.dna1 as dna1_energy
    import mythos.observables as jd_obs
    import mythos.simulators.oxdna as oxdna
    from mythos.input import topology
    from mythos.optimization.objective import DiffTReObjective
    from mythos.optimization.optimization import SimpleOptimizer

    jax.config.update("jax_enable_x64", True)

    # --- Simulator environment and energy function ---
    input_dir = Path("data/templates/simple-helix")
    top = topology.from_oxdna_file(input_dir / "sys.top")

    energy_fn = dna1_energy.create_default_energy_fn(top)

    simulator = oxdna.oxDNASimulator(
        input_dir=input_dir,
        energy_fn=energy_fn,
        source_path=Path("../oxDNA").resolve(),
    )

    # --- Observable & loss function ---
    prop_twist_obs = jd_obs.propeller.PropellerTwist(
        rigid_body_transform_fn=transform_fn,
        h_bonded_base_pairs=jnp.array([
            [1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]
        ]),
    )
    target = 21.7 # degrees

    def loss_fn(traj, weights, **_args):  # We only use traj and weights
        measured = jnp.dot(weights, prop_twist_obs(traj))
        loss = jnp.sqrt((measured - target) ** 2)
        return loss, (("prop_twist", measured), {})

    # --- Objective & optimizer ---
    objective = DiffTReObjective(
        name="propeller_twist",
        required_observables=simulator.exposes(),
        energy_fn=energy_fn,
        grad_or_loss_fn=loss_fn,
        n_equilibration_steps=0,
        min_n_eff_factor=0.95,
    )

    opt = SimpleOptimizer(
        objective=objective,
        simulator=simulator,
        optimizer=optax.adam(learning_rate=1e-3),
    )
    opt.run(opt_params, n_steps=100)


More Examples
-------------

The `examples directory <https://github.com/mythos-bio/mythos/tree/master/examples>`_
in the repository contains runnable examples for a variety of scenarios:

**Simulations:**

- `JAX-MD simulation <https://github.com/mythos-bio/mythos/tree/master/examples/jaxmd/simulation.ipynb>`_
- `oxDNA simulation <https://github.com/mythos-bio/mythos/tree/master/examples/oxdna/simulation.ipynb>`_
- `GROMACS (Martini) workflow <https://github.com/mythos-bio/mythos/tree/master/examples/martini/bottom_up_optimization.ipynb>`_

**Optimizations:**

- `Simple optimization (JAX-MD) <https://github.com/mythos-bio/mythos/tree/master/examples/jaxmd/propeller_twist_optimization.ipynb>`_
  — direct differentiation through a JAX-MD simulation
- `oxDNA DiffTRe optimization <https://github.com/mythos-bio/mythos/tree/master/examples/oxdna/propeller_twist_optimization.ipynb>`_
  — single and multi-trajectory DiffTRe, persistence length, melting temperature
- `MARTINI optimization (GROMACS) <https://github.com/mythos-bio/mythos/tree/master/examples/martini/bottom_up_optimization.ipynb>`_
  — bottom-up fitting, membrane thickness, melting temperature

**Custom energy functions:**

- See :doc:`energy_functions` for custom energy function development patterns.


Where to Go Next
-----------------

- :doc:`simulators` — available simulation backends and setup
- :doc:`energy_functions` — energy models and how to extend them
- :doc:`observables` — observable API, catalog, and loss functions
- :doc:`optimization` — optimization lifecycle, DiffTRe, and the optimizer framework
- :doc:`examples` — runnable notebooks
