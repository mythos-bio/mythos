"""Debye-huckel function for DNA2 model."""

import chex
import jax.numpy as jnp
from typing_extensions import override

import mythos.energy.base as je_base
import mythos.energy.configuration as config
import mythos.energy.dna2.interactions as dna2_interactions
import mythos.utils.types as typ
from mythos.input.topology import Topology


@chex.dataclass(frozen=True)
class DebyeConfiguration(config.BaseConfiguration):
    """Configuration for the debye-huckel energy function."""

    # independent parameters
    q_eff: float | None = None
    lambda_factor: float | None = None
    prefactor_coeff: float | None = None

    kt: float | None = None
    salt_conc: float | None = None

    ## not optimizable
    half_charged_ends: bool | None = None

    # dependent parameters
    lambda_: float | None = None
    kappa: float | None = None
    r_high: float | None = None
    prefactor: float | None = None
    smoothing_coeff: float | None = None
    r_cut: float | None = None

    required_params: tuple[str] = (
        "q_eff",
        "lambda_factor",
        "prefactor_coeff",
        "kt",
        "salt_conc",
        "half_charged_ends",
    )

    @override
    def init_params(self) -> "DebyeConfiguration":
        lambda_ = self.lambda_factor * jnp.sqrt(self.kt / 0.1) / jnp.sqrt(self.salt_conc)
        kappa = 1.0 / lambda_
        r_high = 3 * lambda_
        prefactor = self.prefactor_coeff * (self.q_eff**2)
        smoothing_coeff = -(
            jnp.exp(-r_high / lambda_) * prefactor * prefactor * (r_high + lambda_) * (r_high + lambda_)
        ) / (-4.0 * r_high * r_high * r_high * lambda_ * lambda_ * prefactor)
        r_cut = r_high * (prefactor * r_high + 3.0 * prefactor * lambda_) / (prefactor * (r_high + lambda_))

        return self.replace(
            lambda_=lambda_,
            kappa=kappa,
            r_high=r_high,
            prefactor=prefactor,
            smoothing_coeff=smoothing_coeff,
            r_cut=r_cut,
        )


@chex.dataclass(frozen=True)
class Debye(je_base.BaseEnergyFunction):
    """Debye-huckel energy function for DNA2 model."""

    params: DebyeConfiguration
    is_end: typ.Arr_Nucleotide_Int | None = None

    @override
    def __post_init__(self, topology: Topology | None) -> None:
        je_base.BaseEnergyFunction.__post_init__(self, topology)
        if topology is not None:
            object.__setattr__(self, "is_end", topology.is_end)
        if self.is_end is None:
            raise ValueError("is_end must be provided either through topology or directly.")

    def pairwise_energies(
        self,
        body_i: je_base.BaseNucleotide,
        body_j: je_base.BaseNucleotide,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors_2,
    ) -> typ.Arr_Bonded_Neighbors:
        """Computes the debye-huckel energy for each unbonded pair."""
        op_i = unbonded_neighbors[0]
        op_j = unbonded_neighbors[1]
        mask = jnp.array(op_i < body_i.center.shape[0], dtype=jnp.float32)

        dr_backbone_op = self.displacement_mapped(body_j.back_sites[op_j], body_i.back_sites[op_i])
        r_back_op = jnp.linalg.norm(dr_backbone_op, axis=1)

        db_dgs = dna2_interactions.debye(
            r_back_op,
            self.params.kappa,
            self.params.prefactor,
            self.params.smoothing_coeff,
            self.params.r_cut,
            self.params.r_high,
        )
        db_dgs = jnp.where(mask, db_dgs, 0.0)

        dh_mults_op_i = jnp.where(self.is_end[op_i], 0.5, 1.0)
        dh_mults_op_j = jnp.where(self.is_end[op_j], 0.5, 1.0)
        dh_mults = jnp.multiply(dh_mults_op_i, dh_mults_op_j)
        dh_mults = jnp.where(self.params.half_charged_ends, dh_mults, jnp.ones(op_i.shape[0]))
        return jnp.multiply(db_dgs, dh_mults)

    @override
    def compute_energy(self, nucleotide: je_base.BaseNucleotide) -> typ.Scalar:
        dgs = self.pairwise_energies(nucleotide, nucleotide, self.unbonded_neighbors)
        return dgs.sum()
