"""Membrane thickness observable."""
import chex
import lipyphilic as lpp
import MDAnalysis
from typing_extensions import override

from mythos.observables.martini_utils import universe_from_trajectory
from mythos.simulators.io import SimulatorTrajectory


@chex.dataclass(frozen=True, kw_only=True)
class MembraneThickness:
    """Observable for calculating membrane thickness from a trajectory.

    This observable uses the LiPyphilic package to assign leaflets and compute
    membrane thickness, so see its documentation for more details on the
    lipid_sel and thickness_sel arguments: https://lipyphilic.readthedocs.io/en/latest/.

    Args:
        topology: MDAnalysis Universe topology for the system, should not
            contain the trajectory data.
        lipid_sel: selection string for lipid tail atoms using MDAnalysis query
            (e.g. "name GL1 GL2").
        thickness_sel: selection string for atoms to use in thickness
            calculation (e.g. "name PO4").
    """
    topology: MDAnalysis.Universe
    lipid_sel: str
    thickness_sel: str

    @override
    def __call__(self, trajectory: SimulatorTrajectory) -> float:
        universe = universe_from_trajectory(self.topology, trajectory)

        leaflets = lpp.AssignLeaflets(universe=universe, lipid_sel=self.lipid_sel)
        leaflets.run()
        thicknesses = lpp.analysis.MembThickness(
            universe=universe, lipid_sel=self.thickness_sel, leaflets=leaflets.leaflets
        )
        thicknesses.run()
        return thicknesses.memb_thickness
