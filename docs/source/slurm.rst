Running on Slurm HPC Systems
============================

``mythos`` with the ``RayOptimizer`` can scale across multiple nodes on
Slurm-managed HPC clusters. This page covers practical advice for writing
``sbatch`` scripts and tuning resource allocation.

.. contents:: On this page
   :local:
   :depth: 2


sbatch Script Example
---------------------

The following script launches a multi-node Ray cluster on Slurm and runs an
optimization using ``ray symmetric-run`` (requires Ray 2.50+):

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=mythos-oxdna-lp
    #SBATCH --tasks-per-node=1
    #SBATCH --cpus-per-task=16
    #SBATCH --nodes=4
    #SBATCH --partition=cpu
    #SBATCH --time=02:00:00

    set -x

    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    port=6379
    ip_head=$head_node:$port
    export ip_head
    echo "IP Head: $ip_head"

    module load gcc/9.3.0 cmake/3.27.9  # replace with your cluster's module names
    . ~/mythos/.venv/bin/activate  # replace with your python virtual/conda environment

    srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
      ray symmetric-run \
      --address "$ip_head" \
      --min-nodes "$SLURM_JOB_NUM_NODES" \  # Min nodes waits for all nodes to join before starting
      --num-cpus="${SLURM_CPUS_PER_TASK}" \  # Per-worker logical CPU count
      -- \
      python -u persistence_length_optimization.py


Key sbatch Settings
-------------------

tasks-per-node
^^^^^^^^^^^^^^

Always set ``--tasks-per-node=1``. Each Slurm node runs **one** Ray worker
process; Ray handles task-level parallelism internally. Setting this to a
higher value will launch multiple competing Ray workers on the same node,
leading to resource contention and unpredictable behavior.

cpus-per-task
^^^^^^^^^^^^^

Set ``--cpus-per-task`` to the number of CPU cores each node should expose to
Ray. This value is passed to ``ray symmetric-run`` via ``--num-cpus`` so that
Ray knows how many CPUs are available per node for scheduling tasks.

nodes
^^^^^

The total number of parallel simulators is determined by the total CPU count
across all nodes (``cpus-per-task × nodes``) divided by the CPUs allocated per
simulator via ``scheduler_hints``. For oxDNA, each simulation typically runs
best with a single CPU (``num_cpus=1``), so four nodes with 16 cores each can
run up to 64 simulators in parallel.

Note that a single simulator cannot span multiple nodes — if a simulator
requires more CPUs than ``cpus-per-task``, it will not be schedulable. Ensure
that the per-simulator ``num_cpus`` in ``scheduler_hints`` does not exceed
``cpus-per-task``.


Ray Cluster Startup
-------------------

The script above uses ``ray symmetric-run``, which handles starting and
connecting all workers to the head node automatically. The key flags are:

- ``--address "$ip_head"`` — the head node address, derived from Slurm's
  ``SLURM_JOB_NODELIST``.
- ``--min-nodes "$SLURM_JOB_NUM_NODES"`` — wait for all allocated nodes to
  join before starting the workload.
- ``--num-cpus`` — tells Ray how many CPUs each worker should advertise
  (should match ``--cpus-per-task``).

Module Loading
^^^^^^^^^^^^^^

If the cluster nodes require specific compiler toolchains or build tools (e.g.,
``gcc``, ``cmake``). The oxDNA simulator recompiles its binary during
optimization, so ``cmake`` and a C++ compiler must be available on every node.

Memory Considerations
---------------------

oxDNA Build Memory
^^^^^^^^^^^^^^^^^^

The oxDNA source compilation (triggered each optimization step when parameters
change) can require significant memory — potentially several gigabytes per
build. If you observe out-of-memory errors during the build phase, consider:

- Requesting higher-memory nodes or partitions (via Slurm's ``--mem`` or
  ``--mem-per-cpu`` options, the default may be insufficient).
- Reducing ``n_build_threads`` on the ``oxDNASimulator`` to lower peak memory
  during parallel compilation.
- Setting ``mem_mb`` in ``scheduler_hints`` so Ray avoids scheduling too many
  concurrent builds on the same node.

.. code-block:: python

    from mythos.utils.scheduler import SchedulerHints

    simulator = oxdna.oxDNASimulator(
        ...,
        n_build_threads=4,
        scheduler_hints=SchedulerHints(
            num_cpus=4,
            mem_mb=8192,  # reserve 8 GB per task
        ),
    )

Trajectory and Gradient Memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When many simulators feed trajectories into a single objective, memory can grow
quickly. See the :ref:`Memory Considerations <memory-considerations>` section
of the :doc:`ray_optimizer` page for strategies on managing trajectory
concatenation and gradient computation memory.

Other considerations
^^^^^^^^^^^^^^^^^^^^

Other simulators, objectives, and callbacks may have their own memory
requirements. Typically either the OS or Slurm will kill the job if it exceeds
available or allocated memory, and the logs will show OOM kill messages.

Some options for diagnosing:

* Use Slurm's memory monitoring tools (e.g., ``sacct`` with the ``MaxRSS``
  field)
* Check Ray's dashboard for worker resource usage and correlate which tasks are
  running when OOM kills occur.
* Adjust Slurm's memory allocation and Ray's scheduler hints iteratively on
  a scaled-down sample workflow to find a stable configuration

See :ref:`observability-and-debugging` on the :doc:`ray_optimizer` page for
information on using the Ray dashboard to monitor resource usage, including
how to access it via SSH port forwarding on Slurm clusters.


Troubleshooting
---------------

Further advice on using ray on Slurm can be found in the Ray documentation:
`Deploying on Slurm <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_.

Workers fail to join
^^^^^^^^^^^^^^^^^^^^

If the job times out waiting for workers, verify that all nodes can reach the
head node on port 6379 (or your chosen port). Some clusters have firewall rules
between compute nodes that may block Ray's communication ports, or the port is
already bound by another process, and another should be chosen.

Build failures on worker nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If oxDNA compilation fails on worker nodes but succeeds locally, ensure that
the required modules (``gcc``, ``cmake``) are loaded in the sbatch script.
Worker nodes may not have the same default module set as login nodes.

Out-of-memory kills
^^^^^^^^^^^^^^^^^^^^

If Slurm's OOM killer terminates your job, check whether the issue is during
oxDNA compilation or during gradient computation. Use ``mem_mb`` in
``scheduler_hints`` to help Ray distribute memory-intensive tasks across nodes,
and consider requesting a higher-memory partition.
