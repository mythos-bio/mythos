Examples
========

The `examples directory <https://github.com/mythos-bio/mythos/tree/master/examples>`_
contains runnable notebooks covering simulations and optimizations.

When experimenting with the notebooks, be sure to read the `examples README
<https://github.com/mythos-bio/mythos/tree/master/examples/README.md>`_,
including the prequisites section to ensure you have the necessary software and
access to example data.


Simulations
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Notebook
     - Backend
     - Description
   * - `simulation.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/jaxmd/simulation.ipynb>`_
     - JAX-MD
     - Run a DNA simulation entirely in JAX with Langevin dynamics.
   * - `simulation.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/oxdna/simulation.ipynb>`_
     - oxDNA
     - Run an oxDNA simulation and read back the trajectory.
   * - `bottom_up_optimization.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/martini/bottom_up_optimization.ipynb>`_
     - GROMACS
     - Run a GROMACS-based MARTINI workflow (includes simulation + analysis).


Optimizations
-------------

Simple (direct differentiation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Notebook
     - Backend
     - Description
   * - `propeller_twist_optimization.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/jaxmd/propeller_twist_optimization.ipynb>`_
     - JAX-MD
     - Optimize stacking parameters against propeller twist via autodiff.

Advanced (DiffTRe / multi-objective)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Notebook
     - Backend
     - Description
   * - `propeller_twist_optimization.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/oxdna/propeller_twist_optimization.ipynb>`_
     - oxDNA
     - Single-trajectory DiffTRe optimization of stacking parameters.
   * - `multi_trajectory_optimization.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/oxdna/multi_trajectory_optimization.ipynb>`_
     - oxDNA
     - Multi-trajectory DiffTRe optimization with Ray parallelism.
   * - `persistence_length_optimization.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/oxdna/persistence_length_optimization.ipynb>`_
     - oxDNA
     - Persistence length optimization using DiffTRe.
   * - `melting_temperature_optimization.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/oxdna/melting_temperature_optimization.ipynb>`_
     - oxDNA
     - Melting temperature optimization with umbrella sampling.
   * - `lammps_propeller_twist_optimization.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/oxdna/lammps_propeller_twist_optimization.ipynb>`_
     - LAMMPS
     - oxDNA model optimization using the LAMMPS backend.
   * - `bottom_up_optimization.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/martini/bottom_up_optimization.ipynb>`_
     - GROMACS
     - Bottom-up Martini M2 fitting via Wasserstein distance matching.
   * - `membrane_thickness_optimization.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/martini/membrane_thickness_optimization.ipynb>`_
     - GROMACS
     - Membrane thickness optimization for Martini M2.
   * - `melting_temperature_optimization.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/martini/melting_temperature_optimization.ipynb>`_
     - GROMACS
     - Melting temperature optimization for Martini M3 with Ray.
