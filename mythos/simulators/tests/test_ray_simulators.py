import os

import pytest
import ray

from mythos.simulators.base import BaseSimulation
from mythos.simulators.ray import RayMultiSimulation, RaySimulation


@pytest.fixture(autouse=True, scope="module")
def _ray_init() -> None:
    ray.init("local", num_cpus=2, log_to_driver=False)


@pytest.fixture(params=[1, 3])
def simulator_class(request) -> type[BaseSimulation]:
    num_exposes = request.param
    class DummySimulator(BaseSimulation):
        def exposes(self) -> list[str]:
            return [f"obs-{i}.{self.name}" for i in range(num_exposes)]

        def run(self, params):
            res = {"name": self.name, "params": params, "env": os.environ.get("ENV_VAR")}
            if num_exposes == 1:
                return res
            else:
                return [res] + [f"extra-{i}" for i in range(1, num_exposes)]

    return DummySimulator


def get_first_obs(results, exposes):
    if len(exposes) == 1:
        return results
    else:
        return results[0]


def test_ray_simulator(simulator_class):
    ray_sim = RaySimulation(simulator_class, name="test_sim")
    # sync run
    result = get_first_obs(ray_sim.run("test"), ray_sim.exposes())
    assert result == {"name": "test_sim", "params": "test", "env": None}
    # async run
    result = get_first_obs(ray.get(ray_sim.run_async("test")), ray_sim.exposes())
    assert result == {"name": "test_sim", "params": "test", "env": None}


def test_ray_simulator_pass_ray_options(simulator_class):
    ray_sim = RaySimulation(
        simulator_class,
        ray_options={"runtime_env": {"env_vars": {"ENV_VAR": "42"}}},
        name="test_sim"
    )
    # sync run
    result = get_first_obs(ray_sim.run("test"), ray_sim.exposes())
    assert result == {"name": "test_sim", "params": "test", "env": "42"}
    # async run
    result = get_first_obs(ray.get(ray_sim.run_async("test")), ray_sim.exposes())
    assert result == {"name": "test_sim", "params": "test", "env": "42"}


def test_ray_multi_simulator(simulator_class):
    sims = [RaySimulation(simulator_class, name=f"sim_{i}") for i in range(3)]
    multi_sim = RayMultiSimulation(simulations=sims)
    # sync run
    num_exposes = len(multi_sim.exposes())
    assert num_exposes == 3 * len(sims[0].exposes())
    for results in [multi_sim.run("test"), ray.get(multi_sim.run_async("test"))]:
        results_dict = dict(zip(multi_sim.exposes(), results, strict=True))
        for i in range(3):
            assert results_dict[f"obs-0.sim_{i}"] == {"name": f"sim_{i}", "params": "test", "env": None}
        if num_exposes > 3:
            assert results_dict[f"obs-1.sim_0"] == "extra-1"


def test_ray_multi_simulator_create_call(simulator_class):
    multi_sim = RayMultiSimulation.create(
        3,
        simulator_class,
        ray_options={"runtime_env": {"env_vars": {"ENV_VAR": "42"}}},
        name="created_sim"
    )
    results = ray.get(multi_sim.run_async("test"))
    results_dict = dict(zip(multi_sim.exposes(), results, strict=True))
    for i in range(3):
        assert results_dict[f"obs-0.created_sim.{i}"] == {"name": f"created_sim.{i}", "params": "test", "env": "42"}

