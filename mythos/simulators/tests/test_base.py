from mythos.simulators.base import Simulator
from mythos.utils.scheduler import SchedulerHints


class TestSimulatorBase:
    def test_scheduler_hints_attribute_accepted(self):
        class MySimulator(Simulator):
            def run(self):
                pass

        MySimulator(scheduler_hints=SchedulerHints(num_cpus=2, num_gpus=1))

    def test_accpets_no_scheduler_hints(self):
        class MySimulator(Simulator):
            def run(self):
                pass

        MySimulator()

