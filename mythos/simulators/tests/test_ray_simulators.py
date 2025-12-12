import os

import pytest
import ray

from mythos.simulators.base import BaseSimulation
from mythos.simulators.ray import RayMultiGangSimulation, RayMultiSimulation, RaySimulation


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
            return [res] + [f"extra-{i}" for i in range(1, num_exposes)]

        def custom_method(self, x):
            return f"custom-{x}"

    return DummySimulator


def get_first_obs(results, exposes):
    if len(exposes) == 1:
        return results
    return results[0]


def test_ray_simulator(simulator_class):
    ray_sim = RaySimulation.create(simulator_class, name="test_sim")
    # sync run
    result = get_first_obs(ray_sim.run("test"), ray_sim.exposes())
    assert result == {"name": "test_sim", "params": "test", "env": None}
    # async run
    result = get_first_obs(ray.get(ray_sim.run_async("test")), ray_sim.exposes())
    assert result == {"name": "test_sim", "params": "test", "env": None}


def test_ray_simulator_pass_ray_options(simulator_class):
    ray_sim = RaySimulation.create(
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


def test_ray_simulator_call_method(simulator_class):
    ray_sim = RaySimulation.create(simulator_class, name="test_sim")
    result = ray.get(ray_sim.call_async("custom_method", "value"))
    assert result == "custom-value"
    result = ray_sim.call("custom_method", "value2")
    assert result == "custom-value2"


def test_ray_multi_simulator(simulator_class):
    sims = [RaySimulation.create(simulator_class, name=f"sim_{i}") for i in range(3)]
    multi_sim = RayMultiSimulation(simulations=sims)
    # sync run
    num_exposes = len(multi_sim.exposes())
    assert num_exposes == 3 * len(sims[0].exposes())
    for results in [multi_sim.run("test"), ray.get(multi_sim.run_async("test"))]:
        results_dict = dict(zip(multi_sim.exposes(), results, strict=True))
        for i in range(3):
            assert results_dict[f"obs-0.sim_{i}"] == {"name": f"sim_{i}", "params": "test", "env": None}
        if num_exposes > 3:
            assert results_dict["obs-1.sim_0"] == "extra-1"


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


def test_ray_multi_gang_simulator(simulator_class, tmp_path):
    # create a gang multi-simulator implementation
    class MyGang(RayMultiGangSimulation):
        def pre_run(self, *_args, **_kwargs) -> None:
            tmp_path.joinpath("pre_run_called").touch()

        def post_run(self, observables, *_args, **_kwargs):
            tmp_path.joinpath("post_run_called").touch()
            return observables

    multi_sim = MyGang.create(2, simulator_class)
    expected_exposes = 2 * len(simulator_class().exposes())
    assert len(multi_sim.exposes()) == expected_exposes
    results = multi_sim.run_async("test")
    assert len(results) == expected_exposes
    ray.get(results)
    assert tmp_path.joinpath("pre_run_called").exists()
    assert tmp_path.joinpath("post_run_called").exists()


def test_ray_multi_gang_simulator_gang_exposes(simulator_class, tmp_path):
    # create a gang multi-simulator implementation which adds
    class MyGang(RayMultiGangSimulation):
        def exposes(self) -> list[str]:
            return [*super().exposes(), "group_extra_obs"]

        def post_run(self, observables, *_args, **_kwargs):
            return [*observables, "group_extra_obs_value"]

    multi_sim = MyGang.create(2, simulator_class)
    expected_exposes = 2 * len(simulator_class().exposes()) + 1
    assert len(multi_sim.exposes()) == expected_exposes
    results = multi_sim.run("test")
    assert len(results) == expected_exposes


def test_multi_gang_isolate_gang_run(simulator_class):
    # create a gang multi-simulator implementation which tries to modify self
    class MyGang(RayMultiGangSimulation):
        def pre_run(self, *_args, **_kwargs) -> None:
            self._pre_run = True  # do not try this at home - this test explicitly calls gang_run directly

        def post_run(self, observables, *_args, **_kwargs):
            # try to modify self (should not affect the original)
            self._post_run = True  # see above
            return observables

    multi_sim = MyGang.create(2, simulator_class)
    multi_sim.gang_run("test")
    assert multi_sim._pre_run
    assert multi_sim._post_run
