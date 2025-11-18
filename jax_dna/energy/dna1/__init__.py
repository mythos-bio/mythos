"""oxDNA1 energy implementation in jax_dna."""

import functools
from types import MappingProxyType

from jax_dna.energy import DEFAULT_DISPLACEMENT
from jax_dna.energy.base import BaseEnergyFunction, ComposedEnergyFunction, EnergyFunction
from jax_dna.energy.configuration import BaseConfiguration
from jax_dna.energy.dna1.bonded_excluded_volume import BondedExcludedVolume, BondedExcludedVolumeConfiguration
from jax_dna.energy.dna1.coaxial_stacking import CoaxialStacking, CoaxialStackingConfiguration
from jax_dna.energy.dna1.cross_stacking import CrossStacking, CrossStackingConfiguration
from jax_dna.energy.dna1.expected_hydrogen_bonding import ExpectedHydrogenBonding, ExpectedHydrogenBondingConfiguration
from jax_dna.energy.dna1.expected_stacking import ExpectedStacking, ExpectedStackingConfiguration
from jax_dna.energy.dna1.fene import Fene, FeneConfiguration
from jax_dna.energy.dna1.hydrogen_bonding import HydrogenBonding, HydrogenBondingConfiguration
from jax_dna.energy.dna1.nucleotide import Nucleotide
from jax_dna.energy.dna1.stacking import Stacking, StackingConfiguration
from jax_dna.energy.dna1.unbonded_excluded_volume import UnbondedExcludedVolume, UnbondedExcludedVolumeConfiguration
from jax_dna.energy.utils import default_configs_for
from jax_dna.input.topology import Topology
from jax_dna.utils.types import PyTree


def default_configs() -> tuple[PyTree, PyTree]:
    """Return the default simulation and energy configurations for dna1."""
    return default_configs_for("dna1")


def default_energy_configs(
    overrides: dict = MappingProxyType({}), opts: dict = MappingProxyType({})
) -> list[BaseConfiguration]:
    """Return the default configurations for the energy functions."""
    default_sim_config, default_config = default_configs()

    def get_param(x: str) -> dict:
        return default_config[x] | overrides.get(x, {})

    def get_opts(x: str, defaults: tuple[str] = BaseConfiguration.OPT_ALL) -> tuple[str]:
        return opts.get(x, defaults)

    default_stacking_opts = tuple(set(default_config["stacking"].keys()) - {"kT", "ss_stack_weights"})

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
    ]


def default_energy_fns() -> list[BaseEnergyFunction]:
    """Return the default energy functions."""
    return [
        Fene,
        BondedExcludedVolume,
        Stacking,
        UnbondedExcludedVolume,
        HydrogenBonding,
        CrossStacking,
        CoaxialStacking,
    ]


def default_transform_fn() -> callable:
    """Return the default transform function for dna1 simulations."""
    _, default_config = default_configs()
    geometry = default_config["geometry"]
    return functools.partial(
        Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )


def create_default_energy_fn(
        topology: Topology,
        displacement_fn: callable = DEFAULT_DISPLACEMENT
    ) -> EnergyFunction:
    """Create the default oxDNA1 energy function.

    This creates the composed energy function from the default set of function
    classes, associated configs, and the default transform function -  built for
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
    "Nucleotide",
    "Fene",
    "FeneConfiguration",
    "HydrogenBonding",
    "HydrogenBondingConfiguration",
    "Stacking",
    "StackingConfiguration",
    "CoaxialStacking",
    "CoaxialStackingConfiguration",
    "CrossStacking",
    "CrossStackingConfiguration",
    "BondedExcludedVolume",
    "BondedExcludedVolumeConfiguration",
    "UnbondedExcludedVolume",
    "UnbondedExcludedVolumeConfiguration",
    "ExpectedHydrogenBondingConfiguration",
    "ExpectedHydrogenBonding",
    "ExpectedStackingConfiguration",
    "ExpectedStacking",
    "default_configs",
    "default_energy_fns",
]
