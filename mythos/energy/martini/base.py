"""Common Martini Energy Utilities."""

import chex
from jax_md import space
from typing_extensions import override

from mythos.energy.base import BaseEnergyFunction
from mythos.utils.types import Vector3D


def get_periodic(box_size: Vector3D) -> callable:
    """Return displacement function given box_size."""
    return space.periodic(box_size)[0]


@chex.dataclass(frozen=True, kw_only=True)
class MartiniEnergyFunction(BaseEnergyFunction):
    """Base class for Martini energy functions."""

    atom_types: tuple[str, ...]
    bond_names: tuple[str, ...]
    angle_names: tuple[str, ...]
    displacement_fn: callable = get_periodic

    @override
    def __post_init__(self) -> None:
        pass  # We don't support the instantiation from topology here
