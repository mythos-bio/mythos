Ray Optimizer
=============

The ``RayOptimizer`` runs multiple simulators and objectives in parallel (on a
local machine or a cluster of remote machines) using `Ray <https://ray.io>`_.
This page covers practical recommendations for getting the most out of it.

For the basic API and gradient aggregation example, see :ref:`ray-optimizer` on
the :doc:`optimization` page.

.. contents:: On this page
   :local:
   :depth: 2


Initializing a Ray Session
---------------------------

Ray will automatically connect to an existing cluster or start a local session
when it encounters the first remote call. To pass options, your script should
call ``ray.init()`` as appropriate prior to running the optimizer (or any remote
calls).

JAX environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^

Ray workers are separate processes. Any ``jax.config`` calls you make in the
driver process (e.g., ``jax.config.update("jax_enable_x64", True)``) have **no
effect** inside workers. You must either start your workers with the desired
environment variables set, or pass them through ``runtime_env`` environment
variables:

.. code-block:: python

    import ray

    ray.init(
        runtime_env={
            "env_vars": {
                "JAX_ENABLE_X64": "True",      # 64-bit precision
                "JAX_PLATFORM_NAME": "cpu",     # force CPU (optional)
            },
        },
    )

Other useful ``ray.init`` options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    ray.init(
        num_cpus=32,                 # limit visible CPUs
        ignore_reinit_error=True,    # safe to call ray.init() multiple times
        log_to_driver=True,          # forward worker logs to the driver
    )


Resource Hints
--------------

Both ``Simulator`` and ``Objective`` inherit from ``SchedulerUnit``, which
accepts an optional generic ``scheduler_hints`` parameter. The ``RayOptimizer``
translates these hints into Ray task options when dispatching remote calls.

.. code-block:: python

    from mythos.utils.scheduler import SchedulerHints

    simulator = oxdna.oxDNASimulator(
        ...,
        scheduler_hints=SchedulerHints(
            num_cpus=4,        # Sets OMP_NUM_THREADS=4, a commonly used environment for controlling cpu parallelism
            num_gpus=0,        # For NVIDIA devices, sets CUDA_VISIBLE_DEVICES to partition GPUs for fulfilling requests
            mem_mb=8192,       # 8 GB — converted to bytes for Ray
            max_retries=2,
        ),
    )

Hints are generic, and the ``RayOptimizer`` translates several common fields
into ray options. Those ray-specific options that are not listed can be passed
via the ``custom`` field specifying "ray" as the engine:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``num_cpus``
     - ``int``
     - Number of CPUs to reserve for this task.
   * - ``num_gpus``
     - ``float``
     - Number of GPUs. Fractional values (e.g. ``0.5``) allow GPU sharing.
   * - ``mem_mb``
     - ``int``
     - Memory reservation in megabytes. Converted to bytes for Ray.
   * - ``max_retries``
     - ``int``
     - Number of retry attempts on task failure.
   * - ``custom``
     - ``dict``
     - Engine-specific options. For Ray, use ``{"ray": {...}}``
       (e.g., ``{"ray": {"scheduling_strategy": "SPREAD"}}``).

``remote_options_default`` on the ``RayOptimizer`` itself sets baseline Ray
options that apply to **all** tasks. Per-unit ``scheduler_hints`` override
these defaults:

.. code-block:: python

    optimizer = RayOptimizer(
        ...,
        remote_options_default={"num_cpus": 2},  # default for all tasks
    )


Creating Multiple Simulator Instances
--------------------------------------

Use ``Simulator.create_n()`` to create multiple instances of the same
simulator with unique names. This is the typical pattern for running
multiple independent trajectories:

.. code-block:: python

    simulators = oxdna.oxDNASimulator.create_n(
        n=8,
        name="oxdna-sim",
        input_dir="data/templates/simple-helix",
        sim_type=jdna_types.oxDNASimulatorType.DNA1,
        energy_configs=energy_fn_configs,
        scheduler_hints=SchedulerHints(num_cpus=4, mem_mb=4096),
    )

Each simulator gets a name like ``oxdna-sim.0``, ``oxdna-sim.1``, etc., and
each exposes its own uniquely qualified observables
(e.g., ``trajectory.oxDNASimulator.oxdna-sim.0``). Wire these into your
objectives via ``required_observables``.

.. note::

    Names across simulators and objectives must be unique. These names are
    used for routing observables in addition to tagging the task in ray for
    observability. The optimizer will raise an error if it detects any naming
    conflicts.


Memory Considerations
---------------------

When combining trajectories from many parallel simulators into a single
objective, memory can become a bottleneck. Here are some strategies:

Trajectory concatenation
^^^^^^^^^^^^^^^^^^^^^^^^

``DiffTReObjective`` concatenates all required trajectories into a single
array using ``SimulatorTrajectory.concat()`` before computing reference
energies. If you have *N* simulators each producing *T* frames of *M*
particles, the concatenated trajectory has shape ``(N × T, M, ...)``. This
can easily exceed available memory for large systems.

Recommendations:

- **Reduce frame count per simulator** rather than reducing the number of
  simulators. Shorter trajectories are cheaper to concatenate and
  reweight, while still benefiting from parallel sampling.
- **Set ``mem_mb``** on both simulators and objectives via
  ``scheduler_hints`` so Ray can schedule tasks onto nodes with sufficient
  memory.

Gradient computation
^^^^^^^^^^^^^^^^^^^^

DiffTRe computes gradients by evaluating the energy function on every frame
of the concatenated trajectory. For large systems, this is the most
memory-intensive step because JAX must hold the full computation graph for
autodiff.

If you encounter out-of-memory errors during gradient computation:

- Use fewer frames (shorter simulations or sub-sampling).
- Set ``JAX_PLATFORM_NAME=cpu`` to avoid GPU memory limits for the
  reweighting step, while still using GPU for the simulation binary.
- Consider splitting objectives across more simulators with fewer frames
  each, and aggregating gradients via ``aggregate_grad_fn``.
- Set **map_checkpoint** to ``True`` and tune **map_batch_size** on the energy
  function to enable gradient checkpointing during the traced loss computation.

Gradient Aggregation
--------------------

``aggregate_grad_fn`` receives a list of gradient pytrees — one per objective,
in the same order as the ``objectives`` list — and must return a single pytree
of the same structure.

Common patterns:

**Mean (equal weighting):**

.. code-block:: python

    import operator
    import jax

    def tree_mean(grads):
        summed = jax.tree.map(operator.add, *grads)
        return jax.tree.map(lambda x: x / len(grads), summed)

**Weighted sum:**

.. code-block:: python

    def weighted_grads(grads, weights=(0.7, 0.3)):
        scaled = [jax.tree.map(lambda g: g * w, g) for g, w in zip(grads, weights)]
        return jax.tree.map(operator.add, *scaled)

**Single objective (passthrough):**

.. code-block:: python

    aggregate_grad_fn = lambda grads: grads[0]


Execution Model
---------------

Understanding how ``RayOptimizer`` dispatches work can help with debugging
and performance tuning.

- Simulators and objectives are dispatched as **Ray tasks** (stateless remote
  function calls), not Ray actors.
- Each simulator task returns one ``ObjectRef`` per exposed observable, plus
  one for the simulator state. These refs are passed *by reference* to
  objective tasks — data is not copied through the driver.
- Objectives resolve their input refs inside the worker via ``ray.get()``.
- The optimizer uses a **reactive scheduling loop**: it checks which
  objectives have all their required observables available, dispatches those
  that are ready, and waits for any result to come back before checking
  again.
- An objective may signal that it is **not ready** (e.g., it needs fresh
  simulation data). The optimizer will re-run the necessary simulators and
  retry, up to a maximum of 2 attempts per step.

Callbacks
^^^^^^^^^

``optimizer.run()`` accepts an optional ``callback`` function that is called
after each step:

.. code-block:: python

    def my_callback(optimizer_output, step):
        for component, obs in optimizer_output.observables.items():
            for name, value in obs.items():
                print(f"step {step} | {component}.{name} = {value}")
        continue_running = True
        return None, continue_running  # (modified_output | None, keep_going)

    optimizer.run(params, n_steps=100, callback=my_callback)

Return ``(None, False)`` to trigger early stopping.

NaN detection
^^^^^^^^^^^^^

The optimizer automatically checks for NaN/Inf values in gradients after
each step and raises ``RuntimeError`` if any are found. This typically
indicates a learning rate that is too high or a numerical precision issue
(see the ``JAX_ENABLE_X64`` note above).


Observability and Debugging
===========================

Ray's dashboard is a powerful tool for monitoring task execution, resource
usage, and debugging. ``mythos`` declares a dependency on ``ray``, which doesn't
automatically install the dashboard components. Use ``ray[default]`` or
``ray[dashboard]`` to get the full experience::

    pip install "ray[default]"

When using the dashboard, it is also important to start the cluster separately::

    ray start --head

Your application will automatically connect to the running cluster if your
application starts on the same machine. If you are running on a different
machine, you can specify the address of the head node in ``ray.init()``::

    ray.init(address="<head-node-ip>:6379")

For more details on using the dashboard and configuring Ray clusters, see the
`Ray documentation <https://docs.ray.io/>`_.