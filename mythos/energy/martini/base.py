"""Common Martini Energy Utilities."""

import itertools
from pathlib import Path

import chex
import jax.numpy as jnp
import MDAnalysis
from jax_md import space
from typing_extensions import override

from mythos.energy.base import BaseEnergyFunction
from mythos.utils.types import Arr_N, Vector3D


def get_periodic(box_size: Vector3D) -> callable:
    """Return displacement function given box_size."""
    return space.periodic(box_size)[0]


@chex.dataclass(frozen=True, kw_only=True)
class MartiniTopology:
    """Class representing the topology of a Martini system.

    This class contains information about the atom types, bonded interactions,
    and angles in the system. It can be used to construct energy functions and
    to interpret simulation results.

    Attributes:
        atom_types: A tuple of atom type names.
        atom_names: A tuple of atom names.
        angles: An array of shape (n_angles, 3) containing the indices of the
            atoms involved in each angle.
        bonded_neighbors: An array of shape (n_bonds, 2) containing the indices
            of the bonded pairs of atoms.
        unbonded_neighbors: An array of shape (n_unbonded, 2) containing the indices
            of the unbonded pairs of atoms. If not supplied, it will be computed
            as all pairs of atoms that are not bonded.
    """
    atom_types: tuple[str, ...]
    atom_names: tuple[str, ...]
    residue_names: tuple[str, ...]
    angles: Arr_N
    bonded_neighbors: Arr_N
    unbonded_neighbors: Arr_N | None = None

    @override
    def __post_init__(self) -> None:
        if self.unbonded_neighbors is None:
            n_atoms = len(self.atom_types)
            all_pairs = set(itertools.combinations(range(n_atoms), 2))
            bonded_pairs = {tuple(sorted(pair)) for pair in self.bonded_neighbors.tolist()}
            unbonded_pairs = all_pairs - bonded_pairs
            object.__setattr__(self, "unbonded_neighbors", jnp.array(list(unbonded_pairs)))

    @classmethod
    def from_universe(cls, universe: MDAnalysis.Universe) -> "MartiniTopology":
        """Create a MartiniTopology from a Universe object."""
        return cls(
            atom_types = tuple(universe.atoms.types),
            atom_names = tuple(universe.atoms.names),
            residue_names = tuple(universe.atoms.resnames),
            angles = jnp.array(universe.angles.indices),
            bonded_neighbors = jnp.array(universe.bonds.indices),
        )

    @classmethod
    def from_tpr(cls, tpr_file: Path) -> "MartiniTopology":
        """Create a MartiniTopology from a TPR format topology file."""
        universe = MDAnalysis.Universe(tpr_file)
        return cls.from_universe(universe)


@chex.dataclass(frozen=True, kw_only=True)
class MartiniEnergyFunction(BaseEnergyFunction):
    """Base class for Martini energy functions."""

    atom_types: tuple[str, ...]
    atom_names: tuple[str, ...]
    residue_names: tuple[str, ...]
    angles: Arr_N
    displacement_fn: callable = get_periodic

    @classmethod
    def from_topology(cls, topology: MartiniTopology, **kwargs) -> "MartiniEnergyFunction":
        """Create an energy function from a MartiniTopology."""
        return cls(
            atom_types=topology.atom_types,
            atom_names=topology.atom_names,
            residue_names=topology.residue_names,
            angles=topology.angles,
            bonded_neighbors=topology.bonded_neighbors,
            unbonded_neighbors=topology.unbonded_neighbors,
            **kwargs
        )

    @property
    def bond_names(self) -> tuple[str, ...]:
        """Return bond names based on atom names and bonded neighbors."""
        return tuple(
            f"{self.residue_names[b[0]]}_{self.atom_names[b[0]]}_{self.atom_names[b[1]]}"
            for b in self.bonded_neighbors
        )

    @property
    def angle_names(self) -> tuple[str, ...]:
        """Return angle names based on atom names and angles."""
        return tuple(
            f"{self.atom_names[a[0]]}_{self.atom_names[a[1]]}_{self.atom_names[a[2]]}"
            for a in self.angles
        )


class MartiniEnergyConfiguration:
    """Base class for Martini energy function configurations.

    Given the large size and sparse inclusion of parameters in Martini models,
    this class implements parameters as a dictionary while supporting operations
    of configuration classes used in EnergyFunction.

    This class also supports parameter coupling, where a single proxy parameter
    controls multiple underlying parameters. Couplings should be provided as a
    dictionary of lists, where each key is a proxy parameter name and the value
    is a list of target parameter names that it controls. The `params` field of
    this will be populated with the expanded parameters.

    Subclasses can override `__post_init__` for additional initialization logic.
    Parameters will be available in `self.params` after initialization.
    """
    def __init__(self, couplings: dict[str, list[str]]|None = None, **kwargs):
        self.couplings = couplings or {}
        # ensure that targets for coupling are unique
        all_targets = [v for vals in self.couplings.values() for v in vals]
        if len(all_targets) != len(set(all_targets)):
            raise ValueError("Parameters cannot appear in more than one coupling")
        self.reversed_couplings = {v: k for k, vals in self.couplings.items() for v in vals}

        self.params = {}
        for key, value in kwargs.items():
            if key in self.couplings:
                for subkey in self.couplings[key]:
                    self.params[subkey] = value
            elif key not in self.reversed_couplings:
                self.params[key] = value

        self.__post_init__()

    def __post_init__(self) -> None:
        """Hook for additional initialization in subclasses."""

    @property
    def opt_params(self) -> dict[str, any]:
        """Returns the parameters to optimize."""
        opt_params = {}
        for key, value in self.params.items():
            if key in self.reversed_couplings:
                opt_params[self.reversed_couplings[key]] = value
            else:
                opt_params[key] = value
        return opt_params

    @override
    def __getitem__(self, key: str) -> any:
        if key in self.params:
            return self.params[key]
        if key in self.couplings:
            return self.params[self.couplings[key][0]]  # All have same value
        raise KeyError(f"Parameter '{key}' not found in configuration.")

    @override
    def __contains__(self, key: str) -> bool:
        return key in self.params or key in self.couplings
