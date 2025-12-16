from typing import ClassVar

from mythos.simulators.base import BaseSimulation, MultiSimulation


class DummySimulation(BaseSimulation):
    def run(self, _params):
        return 1


class MultiObsDummySimulation(BaseSimulation):
    exposed_observables: ClassVar[list[str]] = ["obs1", "obs2"]

    def run(self, _params):
        return 1, 2


def test_simulation_base():
    sim = DummySimulation()
    assert sim.exposes() == ["trajectory.DummySimulation." + sim.name]
    assert sim.run({}) == 1


def test_simulation_base_named():
    sim = DummySimulation(name="my_sim")
    assert sim.exposes() == ["trajectory.DummySimulation.my_sim"]
    assert sim.run({}) == 1


def test_multi_simulation():
    sims = [DummySimulation() for _ in range(3)]
    multi_sim = MultiSimulation(simulations=sims)
    assert len(multi_sim.exposes()) == 3
    assert multi_sim.run({}) == [1, 1, 1]


def test_multi_simulation_create():
    multi_sim = MultiSimulation.create(2, DummySimulation)
    exposes = multi_sim.exposes()
    assert len(exposes) == len(set(exposes))  # all unique
    assert len(exposes) == 2
    assert multi_sim.run({}) == [1, 1]


def test_multi_simulation_named():
    multi_sim = MultiSimulation.create(2, DummySimulation, name="test_sim")
    exposes = multi_sim.exposes()
    assert exposes == [
        "trajectory.DummySimulation.test_sim.0",
        "trajectory.DummySimulation.test_sim.1",
    ]
    assert multi_sim.run({}) == [1, 1]


def test_multi_simulation_multiple_exposures():
    multi_sim = MultiSimulation.create(2, MultiObsDummySimulation, name="test_sim")
    exposes = multi_sim.exposes()
    assert len(exposes) == 4
    assert exposes == [
        "obs1.MultiObsDummySimulation.test_sim.0",
        "obs2.MultiObsDummySimulation.test_sim.0",
        "obs1.MultiObsDummySimulation.test_sim.1",
        "obs2.MultiObsDummySimulation.test_sim.1",
    ]

    assert multi_sim.run({}) == [1, 2, 1, 2]  # should be flat
