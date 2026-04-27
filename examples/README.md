# mythos Examples

**[Mythos](https://github.com/mythos-bio/mythos)** is a differentiable molecular simulation framework for parameterizing coarse-grained force fields. These examples demonstrate the core workflows — from running basic simulations to optimizing force-field parameters with DiffTRe.

## Examples by Domain

### oxDNA

| Notebook | Description |
|----------|-------------|
| [simulation.ipynb](oxdna/simulation.ipynb) | Run a basic oxDNA simulation and read the trajectory |
| [propeller_twist_optimization.ipynb](oxdna/propeller_twist_optimization.ipynb) | Direct gradient optimization of propeller twist using DiffTRe |
| [persistence_length_optimization.ipynb](oxdna/persistence_length_optimization.ipynb) | Optimize parameters to match a target persistence length (multi-sim, Ray) |
| [melting_temperature_optimization.ipynb](oxdna/melting_temperature_optimization.ipynb) | Optimize parameters to match a target melting temperature (umbrella sampling) |
| [multi_trajectory_optimization.ipynb](oxdna/multi_trajectory_optimization.ipynb) | Multi-trajectory propeller twist optimization with RayOptimizer |
| [lammps_propeller_twist_optimization.ipynb](oxdna/lammps_propeller_twist_optimization.ipynb) | Propeller twist optimization using LAMMPS as the simulation engine |

### JAX-MD

| Notebook | Description |
|----------|-------------|
| [simulation.ipynb](jaxmd/simulation.ipynb) | Run a differentiable DNA simulation using JAX-MD |
| [propeller_twist_optimization.ipynb](jaxmd/propeller_twist_optimization.ipynb) | Direct gradient optimization of propeller twist through JAX-MD |

### Martini (GROMACS)

| Notebook | Description |
|----------|-------------|
| [bottom_up_optimization.ipynb](martini/bottom_up_optimization.ipynb) | Bottom-up Wasserstein distance matching against atomistic reference distributions |
| [membrane_thickness_optimization.ipynb](martini/membrane_thickness_optimization.ipynb) | Optimize parameters to match a target membrane thickness |
| [melting_temperature_optimization.ipynb](martini/melting_temperature_optimization.ipynb) | Optimize parameters to match a membrane melting temperature |

### Scripts

| Script | Description |
|--------|-------------|
| [gromacs_prep.py](scripts/gromacs_prep.py) | CLI tool to preprocess a GROMACS topology for use with mythos |

## Running the Examples

### As notebooks

```bash
jupyter lab examples/
```

### Converting to scripts

```bash
jupyter nbconvert --to script examples/oxdna/simulation.ipynb
```

## Prerequisites

All notebooks assume you have a working installation of mythos and the relevant
simulation engines. Please refer to the `README.md` in the root of this
repository for installation instructions.

Further, environment specific notebooks may have additional requirements or
considerations:

- **oxDNA examples**: Require the [oxDNA source](https://github.com/lorenzo-rovigatti/oxDNA) cloned alongside the mythos repository (i.e. `../../../oxDNA` relative to the notebook in `examples/oxdna/`). Alternatively, set `source_path` or `binary_path` in the notebook to point to your oxDNA installation.
- **JAX-MD examples**: Run on CPU or GPU. GPU recommended for longer simulations.
- **Martini examples**: Require GROMACS installed and accessible as `gmx` on PATH (or via `--gromacs-binary`).
- **LAMMPS examples**: Require a LAMMPS build with oxDNA support.
