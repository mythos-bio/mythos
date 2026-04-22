Examples
========

The `examples directory <https://github.com/mythos-bio/mythos/tree/master/examples>`_
contains runnable scripts and notebooks covering simulations, optimizations,
and custom energy functions.


Simulations
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Example
     - Backend
     - Description
   * - `jaxmd.py <https://github.com/mythos-bio/mythos/tree/master/examples/simulations/jaxmd/jaxmd.py>`_
     - JAX-MD
     - Run a DNA simulation entirely in JAX with Langevin dynamics.
   * - `oxDNA.py <https://github.com/mythos-bio/mythos/tree/master/examples/simulations/oxdna/oxDNA.py>`_
     - oxDNA
     - Run an oxDNA simulation and read back the trajectory.
   * - `gromacs.py <https://github.com/mythos-bio/mythos/tree/master/examples/simulations/gromacs/gromacs.py>`_
     - GROMACS
     - Run a GROMACS simulation for a MARTINI lipid system.


Optimizations
-------------

Simple (direct differentiation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Example
     - Backend
     - Description
   * - `jaxmd.py <https://github.com/mythos-bio/mythos/tree/master/examples/simple_optimizations/jaxmd/jaxmd.py>`_
     - JAX-MD
     - Optimize stacking parameters against propeller twist via autodiff.

Advanced (DiffTRe / multi-objective)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Example
     - Backend
     - Description
   * - `oxDNA.py <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/oxDNA/oxDNA.py>`_
     - oxDNA
     - Single-trajectory DiffTRe optimization of stacking parameters.
   * - `multi_trajectory.py <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/oxDNA/multi_trajectory.py>`_
     - oxDNA
     - Multi-trajectory DiffTRe optimization with Ray parallelism.
   * - `lp_optimization.py <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/oxDNA/lp_optimization.py>`_
     - oxDNA
     - Persistence length optimization using DiffTRe.
   * - `tm_optimization.py <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/oxDNA/tm_optimization.py>`_
     - oxDNA
     - Melting temperature optimization with umbrella sampling.
   * - `lammps.py <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/oxDNA/lammps.py>`_
     - LAMMPS
     - oxDNA model optimization using the LAMMPS backend.
   * - `m2_bottom_up_opt.py <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/martini/m2_bottom_up_opt.py>`_
     - GROMACS
     - Bottom-up Martini M2 fitting via Wasserstein distance matching.
   * - `m2_thickness_opt.py <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/martini/m2_thickness_opt.py>`_
     - GROMACS
     - Membrane thickness optimization for Martini M2.
   * - `m3_melting_temp.py <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/martini/m3_melting_temp.py>`_
     - GROMACS
     - Melting temperature optimization for Martini M3 with Ray.


Tutorials (Notebooks)
---------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Notebook
     - Description
   * - `jaxmd.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/simulations/jaxmd/jaxmd.ipynb>`_
     - Interactive JAX-MD simulation walkthrough.
   * - `oxDNA.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/simulations/oxdna/oxDNA.ipynb>`_
     - Interactive oxDNA simulation walkthrough.
   * - `Optimization tutorial <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/Optimization_with_JaxDNA_tutorial.ipynb>`_
     - End-to-end optimization tutorial.
   * - `FNANO 2025 tutorial <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/jaxDNA_tutorial_FNANO_2025.ipynb>`_
     - Tutorial from the FNANO 2025 workshop.
   * - `multi_trajectory.ipynb <https://github.com/mythos-bio/mythos/tree/master/examples/advanced_optimizations/oxDNA/multi_trajectory.ipynb>`_
     - Multi-trajectory DiffTRe notebook.

