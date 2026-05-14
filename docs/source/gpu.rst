GPU Acceleration
================

``mythos`` can leverage GPUs at multiple levels: the simulation backend
(oxDNA CUDA), the JAX runtime (for energy evaluation and gradient computation),
and the Ray scheduler (for distributing GPU resources across workers). This page
covers configuration for each.

.. contents:: On this page
   :local:
   :depth: 2


oxDNA CUDA Backend
------------------

The oxDNA simulator supports a CUDA backend that runs the simulation on an
NVIDIA GPU. This can dramatically accelerate individual simulations, especially
for large systems.

Enabling the CUDA backend
^^^^^^^^^^^^^^^^^^^^^^^^^

Set the ``backend`` to ``CUDA`` in your oxDNA input file:

.. code-block:: text

    backend = CUDA
    CUDA_device = <device_id>

When ``mythos`` detects ``backend = CUDA`` in the input configuration, it
automatically passes ``-DCUDA=ON`` to CMake during the oxDNA build step.

Build requirements
^^^^^^^^^^^^^^^^^^

The CUDA backend requires:

- An NVIDIA GPU with a supported compute capability
- The CUDA toolkit installed and available on the build node (``nvcc`` on
  ``PATH``)
- A compatible C++ compiler (e.g., ``gcc``)

If you are building on an HPC cluster, you will typically need to load CUDA
and compiler modules before running your optimization:

.. code-block:: bash

    module load gcc/9.3.0 cuda/11.8 cmake/3.27.9

.. note::

   When using the CUDA backend with the ``RayOptimizer``, ensure that CUDA is
   available on **every worker node** that may run an oxDNA simulation task.
   See `oxDNA documentation <https://lorenzo-rovigatti.github.io/oxDNA/install.html>`_
   for full build instructions and supported GPU architectures.


JAX GPU Usage
-------------

JAX can use GPUs for energy function evaluation and gradient computation
(DiffTRe reweighting). By default, JAX will use a GPU if one is available.

Installing JAX with GPU support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default ``jax`` package is CPU-only. To enable GPU support, install the
CUDA-enabled variant:

.. code-block:: bash

    pip install "jax[cuda12]"

See the `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
for other CUDA versions, ROCm support, and troubleshooting.

Controlling the JAX platform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a GPU-enabled JAX installation is present and a GPU is allocated via
``SchedulerHints``, JAX will automatically use the GPU — no extra
configuration is needed.

To force JAX onto CPU instead (useful when GPU memory is limited and you want
to reserve it entirely for the simulation backend), set ``JAX_PLATFORM_NAME``
in the Ray worker environment. This is necessary because ``jax.config`` calls
in the driver process have no effect inside workers:

.. code-block:: python

    ray.init(
        runtime_env={
            "env_vars": {
                "JAX_ENABLE_X64": "True",
                "JAX_PLATFORM_NAME": "cpu",
            },
        },
    )

.. tip::

   A common pattern is to run the **oxDNA simulation** on the GPU (CUDA
   backend) while running the **JAX gradient computation** on the CPU. This
   avoids competition for GPU memory between the simulation binary and JAX's
   autodiff graph.


Other Simulators
-----------------

GROMACS
^^^^^^^

GROMACS supports GPU acceleration when built with CUDA or OpenCL. Since
``mythos`` invokes the ``gmx`` binary directly, GPU usage depends on how
GROMACS was built and configured on your system. Consult the
`GROMACS installation guide <https://manual.gromacs.org/current/install-guide/index.html>`_
for building with GPU support. Once installed, GPU offloading is typically
controlled via ``gmx mdrun`` flags (e.g., ``-nb gpu``, ``-pme gpu``).

LAMMPS
^^^^^^

LAMMPS supports GPU acceleration through several packages (``GPU``,
``KOKKOS``, ``INTEL``). As with GROMACS, ``mythos`` calls the ``lmp`` binary
directly, so GPU support depends on your LAMMPS build. See the
`LAMMPS GPU documentation <https://docs.lammps.org/Speed_gpu.html>`_ for
build instructions and runtime configuration.

For both GROMACS and LAMMPS, use ``num_gpus`` in ``SchedulerHints`` to ensure
Ray allocates GPU resources appropriately for these tasks.


GPU Allocation with Ray Scheduler Hints
----------------------------------------

The ``RayOptimizer`` uses ``SchedulerHints`` to tell Ray how many GPUs each
task requires. Ray uses this information to partition available GPUs across
workers and set ``CUDA_VISIBLE_DEVICES`` accordingly.

Setting ``num_gpus``
^^^^^^^^^^^^^^^^^^^^

Specify GPU requirements per simulator or objective:

.. code-block:: python

    from mythos.utils.scheduler import SchedulerHints

    simulator = oxdna.oxDNASimulator(
        ...,
        scheduler_hints=SchedulerHints(
            num_cpus=4,
            num_gpus=1,       # reserve 1 GPU for this task
            mem_mb=8192,
        ),
    )

Fractional GPU sharing
^^^^^^^^^^^^^^^^^^^^^^

If your simulations are small enough that multiple can share a single GPU,
use fractional values:

.. code-block:: python

    scheduler_hints=SchedulerHints(
        num_gpus=0.5,  # two tasks can share one GPU
    )

Ray will schedule up to two tasks with ``num_gpus=0.5`` on the same GPU.

.. note::

   Fractional GPU sharing relies on tasks fitting within GPU memory
   simultaneously. If tasks exceed the GPU's memory when co-scheduled,
   you will see CUDA out-of-memory errors. Use ``num_gpus=1`` to guarantee
   exclusive GPU access per task.

For full details on scheduler hints, including ``mem_mb``, ``max_retries``,
and ``custom`` options, see the :doc:`ray_optimizer` page.


Slurm GPU Partitions
---------------------

When running on an HPC cluster with GPU nodes, request a GPU partition and
allocate GPUs in your sbatch script:

.. code-block:: bash

    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1
    #SBATCH --tasks-per-node=1
    #SBATCH --cpus-per-task=8

Ensure that the CUDA toolkit is loaded and that your ``scheduler_hints``
match the number of GPUs allocated per node. See :doc:`slurm` for the full
Slurm setup guide.
