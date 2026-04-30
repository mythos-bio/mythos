Welcome to mythos's documentation!
==========================================

**mythos** is a Python library for differentiable simulation and optimization
of coarse-grained molecular models. It enables fitting force field parameters
to macroscopic experimental data using gradient-based optimization.

Core Features
-------------

- **Multiple simulation backends** — run simulations with JAX-MD, oxDNA,
  GROMACS, or LAMMPS from a unified interface. Simple API for simulators make
  extensions easy. See :doc:`simulators`.
- **Differentiable optimization** — compute gradients via JAX automatic
  differentiation (JAX-MD; https://github.com/jax-md/jax-md) or Boltzmann
  reweighting with DiffTRe (for non-differentiable simulators, such as oxDNA,
  GROMACS, and LAMMPS; see https://www.nature.com/articles/s41467-021-27241-4).
  See :doc:`optimization`.
- **DNA, RNA, and lipid energy models** — built-in support for oxDNA1,
  oxDNA2, oxRNA, oxNA (hybrid DNA/RNA), and MARTINI coarse-grained models, with a
  clear extension API. See :doc:`energy_functions`.
- **Parallel optimization with Ray** — run multiple simulators and objectives in
  parallel across heterogeneous hardware, enabling hyperscale optimization
  campaigns. See :doc:`optimization`.
- **Rich observable library** — The extensible Observable API :doc:`observables`
  turns simulation trajectories into values for loss computations. The library
  includes a wide variety of observable implementations for different types of
  simulations, including structural observables (helical rise, pitch, diameter,
  persistence length, melting temperature), membrane properties (thickness, area
  per lipid), and distance metrics (Wasserstein).

Optimization Lifecycle
----------------------

Optimizations in ``mythos`` are organized around four abstractions:

.. image:: ../_static/mythos_opt_diagram.svg
    :align: center
    :width: 70%

**Simulators** run molecular dynamics and expose **Observables** (trajectories,
scalars). **Objectives** consume observables and compute gradients of a loss
function. The **Optimizer** orchestrates the loop — running simulations,
collecting observables, computing gradients, and updating parameters.

For a full description, see :doc:`optimization`.

Contents
--------

.. toctree::
   :maxdepth: 1

   getting_started
   simulators
   energy_functions
   observables
   optimization
   ray_optimizer
   examples
   autoapi/index
