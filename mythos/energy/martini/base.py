"""Common Martini Energy Utilities."""

import chex
from jax_md import space
from typing_extensions import override

from mythos.energy.base import BaseEnergyFunction
from mythos.utils.types import Arr_N, Vector3D


def get_periodic(box_size: Vector3D) -> callable:
    """Return displacement function given box_size."""
    return space.periodic(box_size)[0]


@chex.dataclass(frozen=True, kw_only=True)
class MartiniEnergyFunction(BaseEnergyFunction):
    """Base class for Martini energy functions."""

    atom_types: tuple[str, ...]
    bond_names: tuple[str, ...]
    angle_names: tuple[str, ...]
    angles: Arr_N  # Shape: (n_angles, 3) - triplets of atom indices
    displacement_fn: callable = get_periodic


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
