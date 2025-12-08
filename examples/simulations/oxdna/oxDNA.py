"""An example of running a simulation using oxDNA.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

``python -m examples.simulations.oxdna.oxDNA``
"""
from pathlib import Path

import mythos.input.trajectory as jdna_traj
import mythos.input.topology as jdna_top
import mythos.simulators.oxdna as jdna_oxdna
import mythos.utils.types as jdna_types


def main():

    input_dir = Path("data/templates/simple-helix")

    simulator = jdna_oxdna.oxDNASimulator(
        input_dir=input_dir,
        sim_type=jdna_types.oxDNASimulatorType.DNA1,
        source_path="../oxDNA",
        energy_configs=[],
    )

    simulator.run()
    simulator.cleanup_build()

    trajectory = jdna_traj.from_file(
        input_dir / "output.dat",
        strand_lengths=jdna_top.from_oxdna_file(input_dir / "sys.top").strand_counts,
    )

    print("Length of trajectory: ", trajectory.state_rigid_body.center.shape[0])


if __name__ == "__main__":
    main()


