Energy Functions
================

Energy functions are at the core of ``mythos``. They define how the potential
energy of a molecular system is computed from particle positions and topology.
During optimization, energy function **parameters** are the quantities being
tuned to match experimental observables.

How energy functions are used depends on the simulator backend:

- **jax_md**: Energy functions are pure JAX callables — they compute energies
  directly in Python, and JAX automatic differentiation provides gradients
  through the simulation.
- **External simulators** (oxDNA, GROMACS, LAMMPS): Energy functions serve as
  **parameter containers** — ``mythos`` writes the parameter values into the
  simulator's input files, but the simulator's own code computes the actual
  energies.

.. contents:: On this page
   :local:
   :depth: 2

.. _energy-overview:

jax_md vs. External Simulators
------------------------------

The table below summarizes the key differences when working with energy
functions across simulator backends:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Aspect
     - jax_md
     - oxDNA / LAMMPS
     - GROMACS
   * - Parameter flow
     - In-memory (JAX arrays)
     - Source recompilation (oxDNA) or text substitution (LAMMPS)
     - Topology file substitution
   * - Gradient method
     - Full JAX autodiff
     - DiffTRe (Boltzmann reweighting)
     - DiffTRe (Boltzmann reweighting)
   * - Energy computation
     - Python (JAX JIT)
     - External binary
     - External binary
   * - Update cost
     - Negligible
     - 10–60 s recompilation (oxDNA), ~0.1 s (LAMMPS)
     - ~0.1 s (grompp)
   * - Composability
     - ``+`` operator, ``ComposedEnergyFunction``
     - N/A
     - N/A
   * - Validation
     - Type-safe (Python)
     - Manual — must match simulator's implementation
     - Manual — must match simulator's implementation


Core Classes
------------

All energy-related base classes live in :doc:`autoapi/mythos/energy/base/index`
and :doc:`autoapi/mythos/energy/configuration/index`.

``EnergyFunction``
  Abstract base. Defines the interface: ``__call__(body) → float``, parameter
  management (``with_params()``, ``opt_params()``), and composition.

``BaseEnergyFunction``
  Primary implementation. Holds a ``params`` configuration, displacement
  function, sequence, and topology. Subclasses implement
  ``compute_energy(nucleotide)``.

``ComposedEnergyFunction``
  Linear superposition of multiple energy functions: :math:`E = \sum_i E_i`.
  Built using the ``+`` operator. Parameters share a **global namespace** —
  calling ``with_params(kT=0.1)`` updates ``kT`` in all constituent functions.

``QualifiedComposedEnergyFunction``
  Like ``ComposedEnergyFunction`` but with **namespaced** parameters
  (e.g., ``Fene.eps_backbone``, ``Stacking.eps_stack``) to avoid collisions.

``BaseConfiguration``
  Immutable parameter container. Defines which parameters are required,
  optimizable, and dependent (computed from others via ``init_params()``).


.. _energy-extending-jaxmd:

Writing Custom Energy Functions (jax_md)
----------------------------------------

Custom energy functions for ``jax_md`` are implemented by subclassing
``BaseEnergyFunction`` and ``BaseConfiguration``.

.. note::

   All energy functions and configurations must be annotated with
   ``@chex.dataclass``, from `chex <https://github.com/google-deepmind/chex>`_.
   This decorator makes the class compatible with JAX transformations.

**Step 1: Define the configuration**

The configuration declares which parameters exist, which are optimizable, and
how dependent parameters are derived:

.. code-block:: python

    import chex
    import mythos.energy.base as jdna_energy

    @chex.dataclass
    class TrivialEnergyConfiguration(jdna_energy.BaseConfiguration):
        some_opt_parameter: float | None = None
        some_dep_parameter: float | None = None

        required_params = ("some_opt_parameter",)
        dependent_params = ("some_dep_parameter",)

        def init_params(self) -> "TrivialEnergyConfiguration":
            self.some_dep_parameter = 2 * self.some_opt_parameter
            return self

**Step 2: Implement the energy function**

Subclass ``BaseEnergyFunction`` and implement ``compute_energy``:

.. code-block:: python

    from typing_extensions import override
    import jax.numpy as jnp
    import mythos.utils.types as typ
    import mythos.energy.dna1 as jdna_energy_dna1

    @chex.dataclass
    class TrivialEnergy(jdna_energy.BaseEnergyFunction):

        @override
        def compute_energy(
            self,
            nucleotide: jdna_energy_dna1.Nucleotide,
        ) -> float:
            bonded_i = nucleotide.center[self.bonded_neighbors[0, :]]
            bonded_j = nucleotide.center[self.bonded_neighbors[1, :]]
            return (
                jnp.sum(jnp.linalg.norm(bonded_i - bonded_j))
                + self.params.some_dep_parameter
            )

**Step 3: Compose with other functions**

Energy functions can be combined using the ``+`` operator:

.. code-block:: python

    total_energy = trivial_energy + fene_energy + stacking_energy

More examples can be found in the implemented energies under
:doc:`autoapi/mythos/energy/dna1/index` and in the
`custom energy functions examples <https://github.com/mythos-bio/mythos/tree/master/examples/custom_energy_functions>`_.


.. _energy-extending-external:

Extending Energy Functions for External Simulators
---------------------------------------------------

For external simulators (oxDNA, GROMACS, LAMMPS), the **simulator** does not
call the energy function directly. Instead, during simulation it:

1. Reads the current parameter values from the energy function
2. Writes them into the simulator's input format
3. Runs the external binary
4. Reads back the trajectory

However, during optimization with :ref:`DiffTRe <difftre>`, the energy
function **is** executed in Python — ``DiffTReObjective`` evaluates it on
the returned trajectory to compute reference energies for Boltzmann
reweighting. This is how gradients are estimated without differentiating
through the external binary.

Because the energy function is evaluated independently of the simulator, you
must ensure that the parameter names and energy formulas in ``mythos`` match
the simulator's implementation. There is no automatic validation.

**GROMACS example:** Parameters are written to the ``.top`` topology file using
:func:`~mythos.input.gromacs_input.replace_params_in_topology`.

**LAMMPS example:** Parameters are mapped to LAMMPS ``pair_coeff`` and
``bond_coeff`` directives via a ``REPLACEMENT_MAP`` dictionary in
:mod:`~mythos.simulators.lammps.lammps_oxdna`.

.. warning::

   When extending energy functions for external simulators:

   - There is **no gradient flow** through the external binary. Use
     :ref:`DiffTRe <difftre>` for gradient estimation.
   - You must **manually verify** that the energy formula in the external
     simulator matches your ``mythos`` energy function.
   - Some energy terms may not be implemented by every simulator (e.g.,
     LAMMPS does not implement ``BondedExcludedVolume``).
   - Parameter updates have different costs: oxDNA requires full
     recompilation (~10–60 s), while GROMACS and LAMMPS use text
     substitution (~0.1 s).


.. _energy-models:

Available Energy Models
-----------------------

``mythos`` ships with several energy models for nucleic acid and
coarse-grained lipid simulations.

.. _energy-dna1:

**dna1 — oxDNA1**
  The original oxDNA coarse-grained DNA model. Includes hydrogen bonding,
  stacking, coaxial stacking, cross stacking, FENE backbone bonds, and
  excluded volume interactions.
  See :doc:`autoapi/mythos/energy/dna1/index`.

.. _energy-dna2:

**dna2 — oxDNA2**
  Extends dna1 with improved stacking interactions for better structural
  properties.
  See :doc:`autoapi/mythos/energy/dna2/index`.

.. _energy-rna2:

**rna2 — RNA**
  RNA-specific interaction model with adapted angular dependencies for
  RNA backbone geometry.
  See :doc:`autoapi/mythos/energy/rna2/index`.

.. _energy-na1:

**na1 — Hybrid DNA/RNA**
  Hybrid model for systems containing both DNA and RNA. Delegates to dna2 or
  rna2 interaction functions based on the nucleotide type.
  See :doc:`autoapi/mythos/energy/na1/index`.

.. _energy-martini-m2:

**martini/m2 — Martini 2**
  Coarse-grained lipid model with harmonic bonds, G96 cosine-based angles,
  and Lennard-Jones non-bonded interactions. Used with the
  :doc:`GROMACS simulator <simulators>`.
  See :doc:`autoapi/mythos/energy/martini/index`.

.. _energy-martini-m3:

**martini/m3 — Martini 3**
  Extends Martini 2 with CHARMM-style angle potentials.
  See :doc:`autoapi/mythos/energy/martini/index`.


Martini Energy Functions
------------------------

Martini energy functions use a different base class hierarchy than the
DNA/RNA models, reflecting the different topology structure of coarse-grained
lipid systems.

``MartiniTopology``
  Describes the coarse-grained system: atom types, residue names, bond and
  angle connectivity. Can be loaded from a GROMACS ``.tpr`` file via
  ``MartiniTopology.from_tpr()`` or from an MDAnalysis ``Universe`` via
  ``MartiniTopology.from_universe()``.

``MartiniEnergyFunction``
  Base class for Martini energy terms. Unlike DNA energy functions, Martini
  functions store atom types and residue names, and do not accept
  user-supplied unbonded neighbor lists.

``MartiniEnergyConfiguration``
  Uses a **dictionary-based** parameter scheme (rather than typed dataclass
  fields) to handle the sparse, name-dependent parameter sets typical of
  Martini systems. Supports **parameter coupling** — a single proxy parameter
  can control multiple underlying parameters:

  .. code-block:: python

      couplings = {
          "eps_bead_type": ["eps_C1_C1", "eps_C1_C2", ...],
      }

See :doc:`autoapi/mythos/energy/martini/index` for the full API reference.
