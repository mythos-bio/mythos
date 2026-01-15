from mythos.simulators.gromacs.gromacs import GromacsSimulator

sim = GromacsSimulator(
    input_dir='data/templates/martini/m2/DMPC/273K',
    input_overrides={
        "nsteps": 1000,
        "nstxout": 100,
        "nstlog": 100,
        "nstcalcenergy": 100,
        "nstenergy": 100,
        "nstcomm": 100,
    },
    energy_fn=1,
)

output = sim.run()
traj = output.observables[0]
n_atoms = traj.rigid_body.center.shape[1]
n_frames = traj.length()
print(f"Simulation complete. Trajectory with {n_atoms} atoms and {n_frames} frames.")
