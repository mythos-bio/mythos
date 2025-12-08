"""Implementation of the oxDNA2 energy model in mythos."""

import functools
from types import MappingProxyType

from mythos.energy import DEFAULT_DISPLACEMENT
from mythos.energy.base import BaseEnergyFunction, ComposedEnergyFunction, EnergyFunction
from mythos.energy.configuration import BaseConfiguration

# Shared dna1 energy functions
from mythos.energy.dna1.bonded_excluded_volume import (
    BondedExcludedVolume,
    BondedExcludedVolumeConfiguration,
)
from mythos.energy.dna1.cross_stacking import CrossStacking, CrossStackingConfiguration
from mythos.energy.dna1.fene import Fene, FeneConfiguration
from mythos.energy.dna1.hydrogen_bonding import HydrogenBonding, HydrogenBondingConfiguration
from mythos.energy.dna1.stacking import StackingConfiguration
from mythos.energy.dna1.unbonded_excluded_volume import (
    UnbondedExcludedVolume,
    UnbondedExcludedVolumeConfiguration,
)

# dna2 specific energy functions
from mythos.energy.dna2.coaxial_stacking import CoaxialStacking, CoaxialStackingConfiguration
from mythos.energy.dna2.debye import Debye, DebyeConfiguration
from mythos.energy.dna2.nucleotide import Nucleotide
from mythos.energy.dna2.stacking import Stacking
from mythos.energy.utils import default_configs_for
from mythos.input.topology import Topology
from mythos.utils.types import PyTree


def default_configs() -> tuple[PyTree, PyTree]:
    """Return the default simulation and energy configurations for dna2."""
    return default_configs_for("dna2")

def default_energy_configs(
    overrides: dict = MappingProxyType({}), opts: dict = MappingProxyType({})
) -> list[BaseConfiguration]:
    """Return the default configurations for the energy functions for dna2."""
    default_sim_config, default_config = default_configs()

    def get_param(x: str) -> dict:
        return default_config[x] | overrides.get(x, {})

    def get_opts(x: str, defaults: tuple[str] = BaseConfiguration.OPT_ALL) -> tuple[str]:
        return opts.get(x, defaults)

    default_stacking_opts = tuple(set(default_config["stacking"].keys()) - {"kT", "ss_stack_weights"})
    default_debye_opts = tuple(set(default_config["debye"].keys()) - {"kT", "salt_conc"})
    debye_params_overrides = {
        "kt": overrides.get("kT", default_sim_config["kT"]),
        "salt_conc": overrides.get("salt_conc", default_sim_config["salt_conc"]),
        "half_charged_ends": overrides.get("half_charged_ends", bool(default_sim_config["half_charged_ends"])),
    }

    return [
        FeneConfiguration.from_dict(get_param("fene"), get_opts("fene")),
        BondedExcludedVolumeConfiguration.from_dict(
            get_param("bonded_excluded_volume"), get_opts("bonded_excluded_volume")
        ),
        StackingConfiguration.from_dict(
            get_param("stacking") | {"kt": overrides.get("kT", default_sim_config["kT"])},
            get_opts("stacking", default_stacking_opts),
        ),
        UnbondedExcludedVolumeConfiguration.from_dict(
            get_param("unbonded_excluded_volume"), get_opts("unbonded_excluded_volume")
        ),
        HydrogenBondingConfiguration.from_dict(get_param("hydrogen_bonding"), get_opts("hydrogen_bonding")),
        CrossStackingConfiguration.from_dict(get_param("cross_stacking"), get_opts("cross_stacking")),
        CoaxialStackingConfiguration.from_dict(get_param("coaxial_stacking"), get_opts("coaxial_stacking")),
        DebyeConfiguration.from_dict(
            get_param("debye") | debye_params_overrides, get_opts("debye", default_debye_opts)
        ),
    ]


def default_energy_fns() -> list[BaseEnergyFunction]:
    """Return the default energy functions for dna2."""
    return [
        Fene,
        BondedExcludedVolume,
        Stacking,
        UnbondedExcludedVolume,
        HydrogenBonding,
        CrossStacking,
        CoaxialStacking,
        Debye,
    ]


def default_transform_fn() -> callable:
    """Return the default transform function for dna2 simulations."""
    _, default_config = default_configs()
    geometry = default_config["geometry"]
    return functools.partial(
        Nucleotide.from_rigid_body,
        com_to_backbone_x=geometry["com_to_backbone_x"],
        com_to_backbone_y=geometry["com_to_backbone_y"],
        com_to_backbone_dna1=geometry["com_to_backbone_dna1"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )


def create_default_energy_fn(
        topology: Topology,
        displacement_fn: callable = DEFAULT_DISPLACEMENT
    ) -> EnergyFunction:
    """Create the default oxDNA2 energy function.

    This creates the composed energy function from the default set of function
    classes, associated configs, and the default transform function - built for
    the provided topology and displacement function.

    Args:
        topology: The topology of the system.
        displacement_fn: The displacement function to use. defaults to
            DEFAULT_DISPLACEMENT.
    """
    return ComposedEnergyFunction.from_lists(
        energy_fns=default_energy_fns(),
        energy_configs=default_energy_configs(),
        transform_fn=default_transform_fn(),
        displacement_fn=displacement_fn,
        topology=topology,
    )


__all__ = [
    "Debye",
    "DebyeConfiguration",
    "CoaxialStacking",
    "CoaxialStackingConfiguration",
    "CrossStacking",
    "CrossStackingConfiguration",
    "Fene",
    "FeneConfiguration",
    "HydrogenBonding",
    "HydrogenBondingConfiguration",
    "Stacking",
    "StackingConfiguration",
    "BondedExcludedVolume",
    "BondedExcludedVolumeConfiguration",
    "UnbondedExcludedVolume",
    "UnbondedExcludedVolumeConfiguration",
    "Nucleotide",
    "default_configs",
    "default_energy_fns",
    "default_energy_configs",
    "default_transform_fn",
    "create_default_energy_fn",
]
