Simulators
==========

``mythos`` supports multiple simulation backends. Each simulator exposes one or
more **observables** (trajectories, scalars, etc.) that can be consumed by
:doc:`objectives <optimization>` during optimization.

For a walkthrough of running your first simulation, see :doc:`getting_started`.

.. contents:: Supported Simulators
   :local:
   :depth: 1


JaxMDSimulator
--------------

The ``JaxMDSimulator`` runs molecular dynamics entirely in Python using
`JAX-MD <https://github.com/jax-md/jax-md>`_. Because the simulation is
implemented in JAX, gradients can be computed via automatic differentiation,
enabling direct optimization of energy function parameters without surrogate
methods.

**Key features:**

- Full JAX autodiff through the simulation trajectory
- Pure-Python energy functions composed with the ``+`` operator
- Dynamic neighbor list updates each step
- Gradient checkpointing support to manage memory
- NVT Langevin dynamics with rigid-body support

**Supported energy models:** :ref:`dna1 <energy-dna1>`, :ref:`dna2 <energy-dna2>`,
:ref:`rna2 <energy-rna2>`, :ref:`na1 <energy-na1>`

See :doc:`autoapi/mythos/simulators/jax_md/index` for the full API reference.

**Examples:**
`simulation <https://github.com/mythos-bio/mythos/tree/master/examples/simulations/jaxmd>`_ ·
`optimization <https://github.com/mythos-bio/mythos/tree/master/examples/simple_optimizations/jaxmd>`_


oxDNASimulator
--------------

The ``oxDNASimulator`` wraps the `oxDNA <https://oxdna.org/>`_ C++/CUDA
simulation code. Parameters are injected by modifying the oxDNA source
(``model.h``) and recompiling the binary. Gradients are obtained via the
:ref:`DiffTRe <difftre>` algorithm (Boltzmann reweighting of reference
trajectories).

**Key features:**

- Supports DNA1 and DNA2 interaction models
- Umbrella sampling via ``oxDNAUmbrellaSampler`` for melting temperature
  calculations
- Parameters baked into compiled binary (recompilation on update)

**Supported energy models:** :ref:`dna1 <energy-dna1>`, :ref:`dna2 <energy-dna2>`

.. note::

   You must install oxDNA yourself. Pass the path to the oxDNA binary
   (simulation mode) or source directory (optimization mode)as parameters when
   constructing the simulator. See the `oxDNA documentation
   <https://lorenzo-rovigatti.github.io/oxDNA/install.html>`_ for installation
   instructions.

See :doc:`autoapi/mythos/simulators/oxdna/index` for the full API reference.

**Examples:**
`simulation <https://github.com/mythos-bio/mythos/tree/master/examples/simulations/oxdna>`_ ·
`advanced optimization <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/oxDNA>`_


GromacsSimulator
----------------

The ``GromacsSimulator`` runs `GROMACS <https://www.gromacs.org/>`_ simulations
for coarse-grained systems, particularly MARTINI lipid membranes. Energy
function parameters are injected into the GROMACS topology (``.top``) file
before each simulation via
:func:`~mythos.input.gromacs_input.replace_params_in_topology`. Gradients are
obtained via :ref:`DiffTRe <difftre>`.

**Key features:**

- Two-phase simulation: equilibration followed by production
- MDP parameter overrides for runtime configuration
- Topology-file parameter injection for optimization
- Reads ``.trr`` trajectories via MDAnalysis

**Supported energy models:** :ref:`martini/m2 <energy-martini-m2>`,
:ref:`martini/m3 <energy-martini-m3>`

.. note::

   Requires a working GROMACS installation. The ``gmx`` binary must be
   available on your ``PATH``. See the
   `GROMACS installation guide <https://manual.gromacs.org/current/install-guide/index.html>`_
   for instructions.

See :doc:`autoapi/mythos/simulators/gromacs/index` for the full API reference.

**Examples:**
`simulation <https://github.com/mythos-bio/mythos/tree/master/examples/simulations/gromacs>`_ ·
`MARTINI optimization <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/martini>`_


LAMMPSoxDNASimulator
--------------------

The ``LAMMPSoxDNASimulator`` uses `LAMMPS <https://www.lammps.org/>`_ as a
backend for running oxDNA-style simulations. Parameters are mapped from mythos
energy function names to LAMMPS ``pair_coeff`` / ``bond_coeff`` directives via
a ``REPLACEMENT_MAP`` and substituted into the LAMMPS input file.

**Key features:**

- LAMMPS-native implementation of oxDNA potentials
- Text-file parameter substitution (no recompilation)
- Temperature variable support

**Supported energy models:** :ref:`dna1 <energy-dna1>`

.. warning::

   LAMMPS does not implement ``BondedExcludedVolume``. This energy term should
   be excluded from the energy function when using this simulator.

.. note::

   Requires a working LAMMPS installation with the ``ASPHERE`` and
   ``CG-DNA`` packages enabled. The ``lmp`` binary must be available on your
   ``PATH``. See the
   `LAMMPS installation guide <https://docs.lammps.org/Install.html>`_
   for instructions.

See :doc:`autoapi/mythos/simulators/lammps/index` for the full API reference.


Creating a Custom Simulator
---------------------------

You can add your own simulation backend by subclassing the base simulator
classes in :mod:`mythos.simulators.base`.

There are two base classes to choose from:

- **``Simulator``** — for in-process simulators (e.g. JAX-based). Override
  the ``run()`` method directly.
- **``InputDirSimulator``** — for file-based simulators that shell out to an
  external binary (e.g. GROMACS, LAMMPS). Implement the
  ``run_simulation(input_dir, ...)`` method; the base class handles copying
  input files to a temporary directory automatically.

Both are frozen ``chex`` dataclasses and must return a
:class:`~mythos.simulators.base.SimulatorOutput`.

Minimal example using ``InputDirSimulator``:

.. code-block:: python

    from pathlib import Path
    from typing import Any

    import chex
    from typing_extensions import override

    from mythos.simulators.base import InputDirSimulator, SimulatorOutput


    @chex.dataclass(frozen=True, kw_only=True)
    class MySimulator(InputDirSimulator):
        """A custom file-based simulator."""

        # Optional: override the default exposed observables
        # exposed_observables = ["trajectory", "energy"]

        @override
        def run_simulation(
            self, input_dir: Path, *args, opt_params: dict[str, Any], **kwargs
        ) -> SimulatorOutput:
            # 1. Write opt_params into input files in input_dir
            # 2. Run the external binary (subprocess, etc.)
            # 3. Read back results
            trajectory = ...  # load from output files
            return SimulatorOutput(observables=[trajectory])

Key points:

- Set ``exposed_observables`` (class variable) to declare which observables
  your simulator produces. Defaults to ``["trajectory"]``.
- The ``exposes()`` method returns fully qualified names of the form
  ``"{observable}.{ClassName}.{name}"`` — these are the strings you reference
  in ``DiffTReObjective.required_observables``.
- Use ``Simulator.create_n(n, name, ...)`` to create multiple instances with
  unique names for parallel optimization with ``RayOptimizer``.

See :doc:`autoapi/mythos/simulators/base/index` for the full base class API.
