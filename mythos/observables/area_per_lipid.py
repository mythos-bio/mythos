"""Membrane thickness observable."""
import chex
import jax.numpy as jnp
import lipyphilic as lpp
import MDAnalysis
from typing_extensions import override

from mythos.observables.martini_utils import universe_from_trajectory
from mythos.simulators.io import SimulatorTrajectory


@chex.dataclass(frozen=True, kw_only=True)
class AreaPerLipid:
    """Observable for calculating area per lipid from a trajectory.

    This observable uses the LiPyphilic package to assign leaflets and compute
    area per lipid, so see its documentation for more details on the
    lipid_sel arguments: https://lipyphilic.readthedocs.io/en/latest/.

    Args:
        topology: MDAnalysis Universe topology for the system, should not
            contain the trajectory data.
        lipid_sel: selection string for lipid tail atoms using MDAnalysis query
            (e.g. "name GL1 GL2").
    """
    topology: MDAnalysis.Universe
    lipid_sel: str

    @override
    def __call__(self, trajectory: SimulatorTrajectory) -> float:
        universe = universe_from_trajectory(self.topology, trajectory)

        leaflets = lpp.AssignLeaflets(universe=universe, lipid_sel=self.lipid_sel)
        leaflets.run()
        area_per_lipid = lpp.analysis.AreaPerLipid(
            universe=universe, lipid_sel=self.lipid_sel, leaflets=leaflets.leaflets
        )
        area_per_lipid.run()
        return jnp.mean(area_per_lipid.areas, axis=0)
