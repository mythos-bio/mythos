"""Base classes for energy functions."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import InitVar
from typing import Any, Union

import chex
import jax
import jax.numpy as jnp
import jax_md
from typing_extensions import override

import mythos.utils.types as typ
from mythos.energy.configuration import BaseConfiguration
from mythos.input.topology import Topology

ERR_PARAM_NOT_FOUND = "Parameter '{key}' not found in {class_name}"
ERR_CALL_NOT_IMPLEMENTED = "Subclasses must implement this method"
ERR_COMPOSED_ENERGY_FN_LEN_MISMATCH = "Weights must have the same length as energy functions"
ERR_COMPOSED_ENERGY_FN_TYPE_ENERGY_FNS = "energy_fns must be a list of energy functions"


class EnergyFunction(ABC):
    """Abstract base class for energy functions.

    These are a class of callable-classes that take in a RigidBody and return
    the energy of the system as a scalar float.
    """

    @abstractmethod
    def __call__(self, body: jax_md.rigid_body.RigidBody) -> float:
        """Calculate the energy of the system."""

    @abstractmethod
    def with_params(self, *repl_dicts: dict, **repl_kwargs: Any) -> "EnergyFunction":
        """Return a new energy function with updated parameters.

        Args:
            *repl_dicts (dict): dictionaries of parameters to update. These
                must come first in the argument list and will be applied in
                order.
            **repl_kwargs: keyword arguments of parameters to update. These are
                applied after any parameter dictionaries supplied as positional
                arguments.
        """

    @abstractmethod
    def with_props(self, **kwargs) -> "EnergyFunction":
        """Create a new energy function from this with updated properties.

        Properties are those that are defined at the energy function class level
        and not the parameters that are defined therein. For example, the
        `displacement_fn` can be modified using this method.
        """

    @abstractmethod
    def with_noopt(self, *params: str) -> "EnergyFunction":
        """Create a new energy function from this with specified parameters non-optimizable."""

    @abstractmethod
    def params_dict(self, *, include_dependent: bool = True, exclude_non_optimizable: bool = False) -> dict:
        """Get the parameters as a dictionary.

        Args:
            include_dependent (bool): whether to include dependent parameters
            exclude_non_optimizable (bool): whether to exclude non-optimizable parameters
        """

    @abstractmethod
    def opt_params(self) -> dict[str, typ.Scalar]:
        """Get the configured optimizable parameters."""

    def map(self, body_sequence: jnp.ndarray) -> jnp.ndarray:
        """Map the energy function over a sequence of rigid bodies."""
        return jax.vmap(self.__call__)(body_sequence)


@chex.dataclass(frozen=True)
class BaseNucleotide(jax_md.rigid_body.RigidBody, ABC):
    """Base nucleotide class."""

    center: typ.Arr_Nucleotide_3
    orientation: typ.Arr_Nucleotide_3 | jax_md.rigid_body.Quaternion
    stack_sites: typ.Arr_Nucleotide_3
    back_sites: typ.Arr_Nucleotide_3
    base_sites: typ.Arr_Nucleotide_3
    back_base_vectors: typ.Arr_Nucleotide_3
    base_normals: typ.Arr_Nucleotide_3
    cross_prods: typ.Arr_Nucleotide_3

    @staticmethod
    @abstractmethod
    def from_rigid_body(rigid_body: jax_md.rigid_body.RigidBody, **kwargs) -> "BaseNucleotide":
        """Create an instance of the subclass from a RigidBody.."""


@chex.dataclass(frozen=True, kw_only=True)
class BaseEnergyFunction(EnergyFunction):
    """Base class for energy functions.

    This class should not be used directly. Subclasses should implement the __call__ method.

    Parameters:
        displacement_fn (Callable): an instance of a displacement function from jax_md.space
    """

    params: BaseConfiguration
    displacement_fn: Callable
    seq: typ.Sequence | None = None
    bonded_neighbors: typ.Arr_Bonded_Neighbors_2 | None = None
    unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2 | None = None
    topology: InitVar[Topology | None] = None
    transform_fn: Callable | None = None

    @override
    def __post_init__(self, topology: Topology | None) -> None:
        if topology:
            object.__setattr__(self, "seq", topology.seq)
            object.__setattr__(self, "bonded_neighbors", topology.bonded_neighbors)
            object.__setattr__(self, "unbonded_neighbors", topology.unbonded_neighbors.T)
        elif any([self.seq is None, self.bonded_neighbors is None, self.unbonded_neighbors is None]):
            raise ValueError("Missing topology information")

    @classmethod
    def create_from(cls, other: "EnergyFunction", **kwargs) -> "EnergyFunction":
        """Create a new energy function from another with updated properties.

        Args:
            other: the energy function to copy properties from
            **kwargs: properties to update, overriding those from other
        """
        props = dict(other) | kwargs
        return cls(**props)

    @property
    def displacement_mapped(self) -> Callable:
        """Returns the displacement function mapped to the space."""
        return jax_md.space.map_bond(self.displacement_fn)

    def __add__(self, other: "BaseEnergyFunction") -> "ComposedEnergyFunction":
        """Add two energy functions together to create a ComposedEnergyFunction."""
        if not isinstance(other, BaseEnergyFunction):
            return NotImplemented

        return ComposedEnergyFunction(energy_fns=[self, other])

    def __mul__(self, other: float) -> "ComposedEnergyFunction":
        """Multiply an energy function by a scalar to create a ComposedEnergyFunction."""
        if not isinstance(other, float | int):
            return NotImplemented

        return ComposedEnergyFunction(
            energy_fns=[self],
            weights=jnp.array([other], dtype=float),
        )

    @override
    def with_props(self, **kwargs: Any) -> EnergyFunction:
        return self.replace(**kwargs)

    @override
    def with_noopt(self, *params: str) -> EnergyFunction:
        updated = set(self.params.non_optimizable_required_params) | set(params)
        new_params = self.params.replace(non_optimizable_required_params=list(updated))
        return self.replace(params=new_params)

    @override
    def opt_params(self) -> dict[str, typ.Scalar]:
        return self.params.opt_params

    @override
    def with_params(self, *repl_dicts: dict, **repl_kwargs: Any) -> EnergyFunction:
        new_params = self.params
        for replacements in repl_dicts:
            new_params = new_params | replacements
        new_params = new_params | repl_kwargs
        return self.replace(params=new_params.init_params())

    @override
    def params_dict(self, include_dependent: bool = True, exclude_non_optimizable: bool = False) -> dict:
        return self.params.to_dictionary(
            include_dependent=include_dependent,
            exclude_non_optimizable=exclude_non_optimizable,
        )

    @override
    def __call__(self, body: jax_md.rigid_body.RigidBody) -> float:
        if self.transform_fn:
            body = self.transform_fn(body)
        return self.compute_energy(body)

    @abstractmethod
    def compute_energy(self, nucleotide: BaseNucleotide) -> float:
        """Compute the energy of the system given the nucleotide."""


@chex.dataclass(frozen=True)
class ComposedEnergyFunction(EnergyFunction):
    """Represents a linear combination of energy functions.

    The parameters of all composite energy functions are treated as sharing a
    global namespace in all setting and retrieval methods. For example, calling
    `with_params(kt=0.1)` will set the parameter `kt` in all those energy
    functions that contain a parameter name `kt`.

    Parameters:
        energy_fns (list[BaseEnergyFunction]): a list of energy functions
        weights (jnp.ndarray): optional, the weights of the energy functions
    """

    energy_fns: list[BaseEnergyFunction]
    weights: jnp.ndarray | None = None

    def __post_init__(self) -> None:
        """Check that the input is valid."""
        if not isinstance(self.energy_fns, list) or not all(
            isinstance(fn, BaseEnergyFunction) for fn in self.energy_fns
        ):
            raise TypeError(ERR_COMPOSED_ENERGY_FN_TYPE_ENERGY_FNS)

        if self.weights is not None and len(self.weights) != len(self.energy_fns):
            raise ValueError(ERR_COMPOSED_ENERGY_FN_LEN_MISMATCH)

    @override
    def with_props(self, **kwargs: Any) -> "ComposedEnergyFunction":
        energy_fns = [fn.with_props(**kwargs) for fn in self.energy_fns]
        return self.replace(energy_fns=energy_fns)

    def _param_in_fn(self, param: str, fn: BaseEnergyFunction) -> bool:
        """Helper for with_params to check if a param is in a given energy function."""
        return param in fn.params

    def _rename_param_for_fn(self, param: str, _fn: BaseEnergyFunction) -> str:
        """Helper to rename a param for input to a given energy function."""
        return param

    def _rename_param_from_fn(self, param: str, _fn: BaseEnergyFunction) -> str:
        """Helper to rename a param for output from a given energy function."""
        return param

    @override
    def with_noopt(self, *params: str) -> "ComposedEnergyFunction":
        energy_fns = []
        for fn in self.energy_fns:
            fn_params = [self._rename_param_for_fn(p, fn) for p in params if self._param_in_fn(p, fn)]
            energy_fns.append(fn.with_noopt(*fn_params))
        return self.replace(energy_fns=energy_fns)

    @override
    def opt_params(self, from_fns: list[type] | None = None) -> dict[str, typ.Scalar]:
        energy_fns = self.energy_fns if from_fns is None else [fn for fn in self.energy_fns if type(fn) in from_fns]
        return {self._rename_param_from_fn(k, fn): v for fn in energy_fns for k, v in fn.opt_params().items()}

    @override
    def with_params(self, *repl_dicts: dict, **repl_kwargs: Any) -> "ComposedEnergyFunction":
        # track replacements which are actually applied to functions in order to
        # error on unused replacements (assume this is unintended)
        all_replacements = set(repl_kwargs) | {k for arg in repl_dicts for k in arg}
        used_replacements = set()
        energy_fns = []
        for fn in self.energy_fns:
            # Flatten all the dict-type arguments. prefer the keyword arguments
            # over the dicts for replacements (they appear last in order).
            new_params = {k: v for arg in repl_dicts for k, v in arg.items() if self._param_in_fn(k, fn)}
            new_params.update({k: v for k, v in repl_kwargs.items() if self._param_in_fn(k, fn)})
            used_replacements.update(new_params.keys())

            # Rename replacement keys if necessary (e.g. for qualified overload)
            new_params = {self._rename_param_for_fn(k, fn): v for k, v in new_params.items()}

            energy_fns.append(fn.with_params(**new_params))

        if unused := all_replacements - used_replacements:
            raise ValueError(f"Some parameters were not used in any energy function: {unused}.")
        return self.replace(energy_fns=energy_fns)

    @override
    def params_dict(self, *, include_dependent: bool = True, exclude_non_optimizable: bool = False) -> dict:
        params = {}
        for fn in self.energy_fns:
            fn_params = fn.params_dict(
                include_dependent=include_dependent, exclude_non_optimizable=exclude_non_optimizable,
            )
            params.update({self._rename_param_from_fn(k, fn): v for k, v in fn_params.items()})
        return params

    def compute_terms(self, body: jax_md.rigid_body.RigidBody) -> jnp.ndarray:
        """Compute each of the energy terms in the energy function."""
        return jnp.array([fn(body) for fn in self.energy_fns])

    @override
    def __call__(self, body: jax_md.rigid_body.RigidBody) -> float:
        energy_vals = self.compute_terms(body)
        return jnp.sum(energy_vals) if self.weights is None else jnp.dot(self.weights, energy_vals)

    def without_terms(self, *terms: list[str|type]) -> "ComposedEnergyFunction":
        """Create a new ComposedEnergyFunction without the specified terms.

        Args:
            *terms: all positional arguments should be either a type or a string
                which is the name of the type to exclude.

        Returns:
            ComposedEnergyFunction: a new ComposedEnergyFunction without the
                specified terms
        """
        new_energy_fns = []
        new_weights = []
        for i, fn in enumerate(self.energy_fns):
            if type(fn) in terms or fn.__class__.__name__ in terms:
                continue
            new_energy_fns.append(fn)
            if self.weights is not None:
                new_weights.append(self.weights[i])

        new_weights = None if self.weights is None else jnp.array(new_weights)
        return self.replace(energy_fns=new_energy_fns, weights=new_weights)

    def add_energy_fn(self, energy_fn: BaseEnergyFunction, weight: float = 1.0) -> "ComposedEnergyFunction":
        """Add an energy function to the list of energy functions.

        Args:
            energy_fn (BaseEnergyFunction): the energy function to add
            weight (float): the weight of the energy function

        Returns:
            ComposedEnergyFunction: a new ComposedEnergyFunction with the added energy function
        """
        if self.weights is None:
            weights = None if weight == 1.0 else jnp.array([1.0] * len(self.energy_fns) + [weight])
        else:
            weights = jnp.concatenate([self.weights, jnp.array([weight])])

        return ComposedEnergyFunction(
            energy_fns=[*self.energy_fns, energy_fn],
            weights=weights,
        )


    def add_composable_energy_fn(self, energy_fn: "ComposedEnergyFunction") -> "ComposedEnergyFunction":
        """Add a ComposedEnergyFunction to the list of energy functions.

        Args:
            energy_fn (ComposedEnergyFunction): the ComposedEnergyFunction to add

        Returns:
            ComposedEnergyFunction: a new ComposedEnergyFunction with the added energy function
        """
        other_weights = energy_fn.weights
        w_none = self.weights is None
        ow_none = other_weights is None
        if w_none and ow_none:
            weights = None
        elif not w_none and not ow_none:
            weights = jnp.concatenate([self.weights, other_weights])
        else:
            this_weights = self.weights if not w_none else jnp.ones(len(energy_fn.energy_fns))
            other_weights = other_weights if not ow_none else jnp.ones(len(self.energy_fns))
            weights = jnp.concatenate([this_weights, other_weights])

        return ComposedEnergyFunction(
            energy_fns=self.energy_fns + energy_fn.energy_fns,
            weights=weights,
        )

    def __add__(self, other: Union[BaseEnergyFunction, "ComposedEnergyFunction"]) -> "ComposedEnergyFunction":
        """Create a new ComposedEnergyFunction by adding another energy function.

        This is a convenience method for the add_energy_fn and add_composable_energy_fn methods.
        """
        if isinstance(other, BaseEnergyFunction):
            energy_fn = self.add_energy_fn
        elif isinstance(other, ComposedEnergyFunction):
            energy_fn = self.add_composable_energy_fn
        else:
            return NotImplemented

        return energy_fn(other)

    def __radd__(self, other: Union[BaseEnergyFunction, "ComposedEnergyFunction"]) -> "ComposedEnergyFunction":
        """Create a new ComposedEnergyFunction by adding another energy function.

        This is a convenience method for the add_energy_fn and add_composable_energy_fn methods.
        """
        return self.__add__(other)

    @classmethod
    def from_lists(
        cls,
        energy_fns: list[BaseEnergyFunction],
        energy_configs: list[BaseConfiguration],
        weights: list[float] | None = None,
        **kwargs,
    ) -> "ComposedEnergyFunction":
        """Create a ComposedEnergyFunction from lists of energy functions and weights.

        Args:
            energy_fns (list[BaseEnergyFunction]): a list of energy functions
            energy_configs (list[BaseConfiguration]): a list of energy configurations
            weights (list[float] | None): optional, a list of weights for the
              energy functions
            **kwargs: keyword arguments to pass to each energy function

        Returns:
            ComposedEnergyFunction: a new ComposedEnergyFunction
        """
        weights = weights if weights is not None else jnp.ones(len(energy_fns))
        functions_configs = zip(energy_fns, energy_configs, strict=True)
        energy_fns = [ef(**kwargs, params=ec.init_params()) for ef, ec in functions_configs]
        return cls(energy_fns=energy_fns, weights=weights)


class QualifiedComposedEnergyFunction(ComposedEnergyFunction):
    """A ComposedEnergyFunction that qualifies parameters by their function.

    Parameters for composite functions do not share a global namespace, but
    instead are qualified by the function they belong to in all setting and
    retrieval methods. For example, parameter `eps_backbone` in Fene energy
    function would be referred to as `Fene.eps_backbone` in the this energy
    function. This is useful for isolating parameters from a specific energy
    function for optimization, however note that not all simulations will
    support this functionality - for example oxDNA simulations write only one
    value per parameter.
    """

    @override
    def _param_in_fn(self, param: str, fn: BaseEnergyFunction) -> bool:
        """Helper for with_params to check if a param is in a given energy function."""
        cls, param = param.split(".", 1)
        return param in fn.params and fn.__class__.__qualname__ == cls

    @override
    def _rename_param_for_fn(self, param: str, fn: BaseEnergyFunction) -> str:
        return param.split(".", 1)[1]

    @override
    def _rename_param_from_fn(self, param: str, fn: BaseEnergyFunction) -> str:
        return f"{fn.__class__.__qualname__}.{param}"

