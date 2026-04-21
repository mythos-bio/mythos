"""Full Martini reparameterization from a directory of YAML config files.

Reads YAML configs describing lipid systems and experimental targets, builds
per-system simulators and DiffTRe objectives, and optimizes shared force-field
parameters using RayOptimizer with gradient averaging.

Usage::

    python full_reparam.py --config-dir configs/ --opt-steps 100
    python full_reparam.py --config-file dopc.yaml --config-file dppc.yaml
    python full_reparam.py --config-dir configs/ --config-file extra.yaml

YAML config format (one file per system)::

    name: dopc_303K                # optional, defaults to filename stem
    system: /path/to/gromacs/input
    martini_version: 2             # 2 or 3
    temperature: 303

    thickness:
      target: 36.8
    apl:
      target: 66.9
    bonds:
      units: angstrom               # optional, angstrom (default) or nm
      DOPC:
        NC3-PO4:
          distribution: ref/DOPC_NC3-PO4_bond_dist.npy
    angles:
      units: radian                 # optional, radian (default) or degree
      DOPC:
        PO4-GL1-GL2:
          distribution: ref/DOPC_PO4-GL1-GL2_angle_dist.npy
    melting_temp:
      target: 314.0
      sim_temps: [291, 292.5, 294, ...]
"""

import argparse
import functools
import operator
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import MDAnalysis
import numpy as np
import optax
import ray
import yaml
from mythos.energy.base import ComposedEnergyFunction
from mythos.energy.martini.base import MartiniTopology
from mythos.energy.martini.m2 import LJ, AngleConfiguration, Bond, BondConfiguration, LJConfiguration
from mythos.energy.martini.m2 import Angle as M2Angle
from mythos.energy.martini.m3 import Angle as M3Angle
from mythos.input.gromacs_input import read_params_from_topology
from mythos.observables.area_per_lipid import AreaPerLipid
from mythos.observables.bond_distances import BondDistancesMapped
from mythos.observables.membrane_melting_temp import MembraneMeltingTemp
from mythos.observables.membrane_thickness import MembraneThickness
from mythos.observables.triplet_angles import TripletAnglesMapped
from mythos.observables.wasserstein import WassersteinDistanceMapped
from mythos.optimization.objective import DiffTReObjective
from mythos.optimization.optimization import RayOptimizer
from mythos.simulators.gromacs.gromacs import KB, GromacsSimulator
from mythos.simulators.gromacs.utils import preprocess_topology
from mythos.ui.loggers.console import ConsoleLogger
from mythos.ui.loggers.disk import FileLogger
from mythos.ui.loggers.multilogger import MultiLogger
from mythos.utils.types import PyTree

jax.config.update("jax_enable_x64", True)


class PermissiveComposedEnergyFunction(ComposedEnergyFunction):
    """ComposedEnergyFunction that silently ignores unknown parameters.

    When multiple systems share an optimizer, the merged parameter dict
    contains keys from all systems. The base class raises on params not
    recognized by any sub-function; this subclass filters them out.
    """

    def with_params(self, *repl_dicts, **repl_kwargs):
        merged = {}
        for d in repl_dicts:
            merged.update(d)
        merged.update(repl_kwargs)
        known = {k: v for k, v in merged.items()
                 if any(self._param_in_fn(k, fn) for fn in self.energy_fns)}
        return super().with_params(known)


def tree_mean(trees: tuple[PyTree]) -> PyTree:
    if len(trees) <= 1:
        return trees[0]
    summed = functools.reduce(lambda a, b: jax.tree.map(operator.add, a, b), trees)
    return jax.tree.map(lambda x: x / len(trees), summed)


def parse_args():
    p = argparse.ArgumentParser(description="Full Martini reparameterization from YAML configs")
    p.add_argument("--config-dir", type=Path, default=None,
                   help="Directory containing YAML config files")
    p.add_argument("--config-file", type=Path, action="append", default=[],
                   help="Individual YAML config file (may be specified multiple times)")
    p.add_argument("--opt-steps", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--num-sims", type=int, default=1,
                   help="Default number of parallel sims per system (overridden by per-config num_sims)")
    p.add_argument("--equilibration-steps", type=int, default=200_000)
    p.add_argument("--simulation-steps", type=int, default=500_000)
    p.add_argument("--snapshot-steps", type=int, default=10_000)
    p.add_argument("--gromacs-binary", type=Path, default=None)
    p.add_argument("--metrics-file", type=Path, default=None)
    return p.parse_args()


def load_configs(config_dir: Path | None, config_files: list[Path]):
    yaml_files = list(config_files)
    if config_dir is not None:
        yaml_files.extend(sorted(config_dir.glob("*.yaml")))
    if not yaml_files:
        raise SystemExit("No config files provided. Use --config-dir and/or --config-file.")

    configs = []
    names = set()
    for f in yaml_files:
        with open(f) as fh:
            cfg = yaml.safe_load(fh)
        cfg["_file"] = f
        name = cfg.get("name", f.stem)
        if name in names:
            raise SystemExit(f"Duplicate config name: {name} (from {f})")
        names.add(name)
        cfg["name"] = name

        if "system" not in cfg:
            raise SystemExit(f"Config {f} missing required field: system")
        if "temperature" not in cfg:
            raise SystemExit(f"Config {f} missing required field: temperature")
        system_dir = Path(cfg["system"])
        if not system_dir.is_dir():
            raise SystemExit(f"Config {f}: system directory does not exist: {system_dir}")

        cfg.setdefault("martini_version", 2)
        if cfg["martini_version"] not in (2, 3):
            raise SystemExit(f"Config {f}: martini_version must be 2 or 3")

        configs.append(cfg)

    return configs


def get_topology(data_dir: Path, gromacs_binary: Path | None = None):
    # use the cached preprocessed topology if it exists, otherwise preprocess with grompp
    if not data_dir.joinpath("preprocessed.top").exists() or not data_dir.joinpath("preprocessed.tpr").exists():
        preprocess_topology(data_dir, gromacs_binary=gromacs_binary)

    u = MDAnalysis.Universe(data_dir / "preprocessed.tpr")
    params = read_params_from_topology(data_dir / "preprocessed.top")
    return u, params


def build_energy_fn(top, params, martini_version):
    AngleCls = M2Angle if martini_version == 2 else M3Angle
    fns = []
    if martini_version == 2 and "nonbond_params" in params:
        fns.append(LJ.from_topology(topology=top, params=LJConfiguration(**params["nonbond_params"])))
    fns.append(Bond.from_topology(topology=top, params=BondConfiguration(**params["bond_params"])))
    fns.append(AngleCls.from_topology(topology=top, params=AngleConfiguration(**params["angle_params"])))
    return PermissiveComposedEnergyFunction(energy_fns=fns)


def parse_bond_angle_targets(cfg):
    """Convert YAML bonds/angles sections into topology-named distribution maps.

    Bond/angle names in YAML use '-' separators (e.g. NC3-PO4) nested under
    residue names. These are converted to the topology format using '_'
    separators (e.g. DOPC_NC3_PO4).

    Units can be specified per-section via a 'units' key:
      bonds.units: 'angstrom' (default) or 'nm'
      angles.units: 'radian' (default) or 'degree'
    """
    bonds_section = cfg.get("bonds") or {}
    bond_units = bonds_section.pop("units", "angstrom") if isinstance(bonds_section, dict) else "angstrom"
    bond_map = {}
    for resname, bonds in bonds_section.items():
        for bond_name, bond_info in bonds.items():
            topo_name = f"{resname}_{bond_name.replace('-', '_')}"
            ref = np.load(bond_info["distribution"])
            if bond_units == "angstrom":
                ref = ref * 0.1
            bond_map[topo_name] = jnp.array(ref)

    angles_section = cfg.get("angles") or {}
    angle_units = angles_section.pop("units", "radian") if isinstance(angles_section, dict) else "radian"
    angle_map = {}
    for resname, angles in angles_section.items():
        for angle_name, angle_info in angles.items():
            topo_name = f"{resname}_{angle_name.replace('-', '_')}"
            ref = np.load(angle_info["distribution"])
            if angle_units == "degree":
                ref = np.deg2rad(ref)
            angle_map[topo_name] = jnp.array(ref)

    return bond_map, angle_map


def merge_opt_params(energy_fns):
    merged = {}
    for efn in energy_fns:
        for k, v in efn.opt_params().items():
            if k in merged and not jnp.allclose(merged[k], v):
                warnings.warn(f"Parameter {k} has different initial values across systems")
            merged[k] = v
    return merged


def main():
    args = parse_args()
    if not args.config_dir and not args.config_file:
        raise SystemExit("Error: at least one of --config-dir or --config-file is required.")
    configs = load_configs(args.config_dir, args.config_file)

    all_simulators = []
    all_objectives = []
    energy_fns = []

    for cfg in configs:
        name = cfg["name"]
        system_dir = Path(cfg["system"])
        temperature = cfg["temperature"]
        num_sims = cfg.get("num_sims", args.num_sims)
        martini_version = cfg["martini_version"]

        print(f"\n--- Setting up {name} ---")
        universe, params = get_topology(system_dir, args.gromacs_binary)
        top = MartiniTopology.from_universe(universe)
        energy_fn = build_energy_fn(top, params, martini_version)
        energy_fns.append(energy_fn)

        has_bonds_or_angles = cfg.get("bonds") or cfg.get("angles")
        has_thickness = "thickness" in cfg
        has_apl = "apl" in cfg
        needs_standard_sims = has_bonds_or_angles or has_thickness or has_apl

        # Standard simulators shared across bond/angle/thickness/APL objectives
        standard_sims = []
        if needs_standard_sims:
            standard_sims = GromacsSimulator.create_n(num_sims,
                name=name,
                input_dir=system_dir,
                energy_fn=energy_fn.with_props(unbonded_neighbors=None),
                equilibration_steps=args.equilibration_steps,
                simulation_steps=args.simulation_steps,
                input_overrides={"nstxout": args.snapshot_steps, "ref-t": temperature},
                binary_path=args.gromacs_binary,
            )
            all_simulators.extend(standard_sims)

        standard_required = tuple(obs for s in standard_sims for obs in s.exposes())

        # Wasserstein objective for bonds + angles
        if has_bonds_or_angles:
            bond_map, angle_map = parse_bond_angle_targets(cfg)
            wasserstein_observables = []
            if bond_map:
                wasserstein_observables.append(WassersteinDistanceMapped(
                    observable=BondDistancesMapped(topology=top, bond_names=tuple(bond_map.keys())),
                    v_distribution_map=bond_map,
                ))
            if angle_map:
                wasserstein_observables.append(WassersteinDistanceMapped(
                    observable=TripletAnglesMapped(topology=top, angle_names=tuple(angle_map.keys())),
                    v_distribution_map=angle_map,
                ))
            n_total = sum(len(obs.v_distribution_map) for obs in wasserstein_observables)

            def make_wasserstein_loss(w_obs, n):
                def wasserstein_loss(traj, weights, *_):
                    total = jnp.float64(0.0)
                    for obs in w_obs:
                        for v in obs(traj, weights).values():
                            total = total + v
                    loss = jnp.sqrt(total / n)
                    return loss, (("wasserstein_mean", loss), ())
                return wasserstein_loss

            all_objectives.append(DiffTReObjective(
                energy_fn=energy_fn,
                grad_or_loss_fn=make_wasserstein_loss(wasserstein_observables, n_total),
                required_observables=standard_required,
                name=f"{name}.wasserstein",
            ))

        # Thickness objective
        if has_thickness:
            thickness_cfg = cfg["thickness"]
            target = thickness_cfg["target"]
            thickness_obs = MembraneThickness(
                topology=universe,
                lipid_sel=thickness_cfg.get("lipid_sel", "name GL1 GL2"),
                thickness_sel=thickness_cfg.get("thickness_sel", "name PO4"),
            )

            def make_thickness_loss(obs, tgt):
                def thickness_loss(traj, weights, *_):
                    all_thickness = obs(traj)
                    expected = jnp.dot(weights, all_thickness)
                    loss = jnp.sqrt((tgt - expected) ** 2)
                    return loss, (("thickness", expected), ())
                return thickness_loss

            all_objectives.append(DiffTReObjective(
                energy_fn=energy_fn,
                grad_or_loss_fn=make_thickness_loss(thickness_obs, target),
                required_observables=standard_required,
                name=f"{name}.thickness",
            ))

        # APL objective
        if has_apl:
            apl_target = cfg["apl"]["target"]
            apl_obs = AreaPerLipid(
                topology=universe,
                lipid_sel=cfg["apl"].get("lipid_sel", "name GL1 GL2"),
            )

            def make_apl_loss(obs, tgt):
                def apl_loss(traj, weights, *_):
                    all_apl = obs(traj)
                    expected = jnp.dot(weights, all_apl)
                    loss = jnp.sqrt((tgt - expected) ** 2)
                    return loss, (("apl", expected), ())
                return apl_loss

            all_objectives.append(DiffTReObjective(
                energy_fn=energy_fn,
                grad_or_loss_fn=make_apl_loss(apl_obs, apl_target),
                required_observables=standard_required,
                name=f"{name}.apl",
            ))

        # Melting temperature objective
        if "melting_temp" in cfg:
            tm_cfg = cfg["melting_temp"]
            target_tm = tm_cfg["target"]
            sim_temps = tm_cfg["sim_temps"]
            lipid_sel = tm_cfg.get("lipid_sel", "name GL1 GL2")

            tm_sims = [
                GromacsSimulator(
                    name=f"{name}.tm.temp_{temp:.1f}K",
                    input_dir=system_dir,
                    energy_fn=energy_fn.with_props(unbonded_neighbors=None),
                    simulation_steps=args.simulation_steps,
                    equilibration_steps=0,
                    input_overrides={"nstxout": args.snapshot_steps, "ref-t": temp},
                    binary_path=args.gromacs_binary,
                ) for temp in sim_temps
            ]
            all_simulators.extend(tm_sims)

            tm_obs = MembraneMeltingTemp(
                topology=universe,
                lipid_sel=lipid_sel,
                temperatures=jnp.array(sim_temps) * KB,
            )

            def make_tm_loss(obs, tgt):
                tgt_kt = tgt * KB
                def tm_loss(traj, weights, *_):
                    tm = obs(traj, weights=weights)
                    loss = (tm - tgt_kt) ** 2
                    return loss, (("tm", tm), ())
                return tm_loss

            tm_required = tuple(obs for s in tm_sims for obs in s.exposes())
            all_objectives.append(DiffTReObjective(
                energy_fn=energy_fn,
                grad_or_loss_fn=make_tm_loss(tm_obs, target_tm),
                required_observables=tm_required,
                name=f"{name}.tm",
                max_valid_opt_steps=5,
            ))

    # Merge parameters and build optimizer
    merged_params = merge_opt_params(energy_fns)

    loggers = [ConsoleLogger()]
    if args.metrics_file is not None:
        loggers.append(FileLogger(args.metrics_file))

    opt = RayOptimizer(
        simulators=all_simulators,
        objectives=all_objectives,
        optimizer=optax.adam(learning_rate=args.learning_rate),
        aggregate_grad_fn=tree_mean,
        logger=MultiLogger(loggers),
    )

    print("\n=== Reparameterization Setup ===")
    for cfg in configs:
        print(f"  {cfg['name']}: system={cfg['system']}, T={cfg['temperature']}K, "
              f"martini_v{cfg['martini_version']}")
    print(f"  Simulators: {len(all_simulators)}")
    print(f"  Objectives: {[o.name for o in all_objectives]}")
    print(f"  Parameters: {len(merged_params)}")

    ray.init(runtime_env={"env_vars": {"JAX_ENABLE_X64": "True"}})

    opt.run(params=merged_params, n_steps=args.opt_steps)


if __name__ == "__main__":
    main()
