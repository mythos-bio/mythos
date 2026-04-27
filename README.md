<img src="https://raw.githubusercontent.com/mythos-bio/mythos/transition-docs-1/logo.svg" alt="mythos logo">

[![CI](https://github.com/mythos-bio/mythos/actions/workflows/ci.yml/badge.svg)](https://github.com/mythos-bio/mythos/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/mythos/badge/?version=latest)](https://mythos.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/mythos-bio/mythos/branch/master/graph/badge.svg?token=0KPNKHRC2V)](https://codecov.io/gh/mythos-bio/mythos)
[![Security](https://github.com/mythos-bio/mythos/actions/workflows/security.yml/badge.svg?branch=master)](https://github.com/mythos-bio/mythos/actions/workflows/security.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2411.09216-b31b1b.svg)](https://arxiv.org/abs/2411.09216)


Mythos is a Python package for simulating and fitting coarse-grained molecular
models to macroscopic experimental data.

Currently, Mythos can run simulations using
[JAX-MD](https://github.com/jax-md/jax-md), [oxDNA](https://oxdna.org/),
[GROMACS](https://www.gromacs.org/), and
[LAMMPS](https://www.lammps.org/).
(oxDNA, GROMACS, and LAMMPS must be installed separately.)

Further, Mythos supports fitting models using JAX-MD (Direct Differentiation,
and [DiffTRe](https://www.nature.com/articles/s41467-021-27241-4)) and oxDNA /
GROMACS / LAMMPS (DiffTRe only). Built-in energy models include oxDNA1, oxDNA2,
RNA, hybrid DNA/RNA, and MARTINI 2/3 coarse-grained lipid models.


## Quick Start

We recommend using a fresh conda environment with Python 3.11. You can create a
new environment with the following command:

```bash
conda create -y -n mythos python=3.11
conda activate mythos
```

Depending on your hardware, you may want to install the GPU accelerated version
of JAX, see the [JAX
documentation](https://docs.jax.dev/en/latest/installation.html#installation)
for more details on how to do this. If you aren't interested in GPU support, you
can skip straight to installing mythos which will install the CPU version of
JAX.


First install mythos using pip:

```bash
pip install git+https://github.com/mythos-bio/mythos.git
```

### Simulations

Information on how to run a simulation can be found in the
[documentation](https://mythos.readthedocs.io/en/latest/getting_started.html).

One advantage of mythos is that you can specify a custom energy function for
both simulations and optimizations. Information on how energy functions are
defined and how to define your own energy functions can be found in the
documentation.

### Optimizations

The optimization framework is built around four abstractions:

<p align="center">
  <img src="https://raw.githubusercontent.com/mythos-bio/mythos/master/docs/_static/mythos_opt_diagram.svg" alt="mythos optimization lifecycle" width="80%">
</p>

- **Simulator** — runs a simulation and produces observables.
- **Observable** — a quantity produced by a simulator (e.g. a trajectory,
  structural property, or thermodynamic measurement).
- **Objective** — computes gradients of a loss function with respect to
  the parameters being optimized.
- **Optimizer** — coordinates simulators, collects observables, passes them
  to objectives, aggregates gradients, and applies parameter updates.

Multiple simulators (potentially using different backends) and multiple
objectives can be jointly optimized. For example, you can simultaneously fit
structural and thermodynamic properties across simulations at different
conditions. Parallel execution is supported via
[Ray](https://ray.io), enabling distribution across local cores or a remote
cluster.

For more details, see the
[optimization docs](https://mythos.readthedocs.io/en/latest/optimization.html)
and the
[examples](https://github.com/mythos-bio/mythos/tree/master/examples).


## Development

We welcome contributions! If you are looking for something to work on, check out
the [issues](https://github.com/mythos-bio/mythos/issues).

If you have a feature request or an idea that you would like to contribute,
please open an issue. The project is fast moving, opening an issue will help us
to give you quick feedback and help you to get started.

See the [CONTRIBUTING](https://github.com/mythos-bio/mythos/blob/master/CONTRIBUTING.md)


