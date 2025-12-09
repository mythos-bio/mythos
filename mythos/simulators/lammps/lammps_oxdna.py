"""LAMMPS-based OxDNA simulator for mythos."""

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import field
from pathlib import Path
from typing import Any

import chex
import numpy as np
from typing_extensions import override

from mythos.energy.base import EnergyFunction
from mythos.input.trajectory import NucleotideState, Trajectory, validate_box_size
from mythos.simulators.base import BaseSimulation
from mythos.simulators.io import SimulatorTrajectory
from mythos.utils.types import Params


@chex.dataclass
class LAMMPSoxDNASimulator(BaseSimulation):
    """LAMMPS-based OxDNA simulator.

    Please note that for LAMMPS simulations of oxDNA, BondedExcludedVolume
    should be left out of the energy function, as LAMMPS does not implement it
    (or does not in a compatible way).

    Args:
        input_dir: Path to the directory containing the LAMMPS input files.
        overwrite: Whether to overwrite the input directory or copy to a
            temporary directory.
        input_file_name: Name of the LAMMPS input file (default "input").
        energy_fn: Energy function used in the simulation, for updating parameters.
        variables: Additional variables to set in the LAMMPS input file before
            run. These variables must already be defined in the input file using
            a command of the form "variable name equal value".
    """
    input_dir: Path
    energy_fn: EnergyFunction
    overwrite: bool = False
    input_file_name: str = "input"
    variables: dict[str, Any] = field(default_factory=dict)

    @override
    def __post_init__(self) -> None:
        if not self.overwrite:
            input_input_dir = self.input_dir
            self.input_dir = Path(tempfile.mkdtemp(prefix="jaxdna-lammps-oxdna-sim-"))
            logging.info("Copying LAMMPS OxDNA input files to temporary directory: %s", self.input_dir)
            shutil.copytree(input_input_dir, self.input_dir, dirs_exist_ok=True)
        else:
            self.input_dir = Path(self.input_dir)
        self.input_lines = self.input_dir.joinpath(self.input_file_name).read_text().splitlines()

    @override
    def run(self, params: list[dict[str, float]], seed: int | None = None) -> Path:
        self._replace_parameters(params, seed)
        with self.input_dir.joinpath("lmp.out").open("a") as f:
            subprocess.check_call(
                ["lmp", "-in", self.input_file_name],
                cwd=self.input_dir,
                shell=False,
                stdout=f,
                stderr=f
            )
        traj = _read_lammps_output(self.input_dir.joinpath("trajectory.dat"))

        return SimulatorTrajectory(
            rigid_body=traj.state_rigid_body,
        )

    def _replace_parameters(self, params: Params, seed: int | None) -> None:
        updated_params = self.energy_fn.with_params(params).params_dict(exclude_non_optimizable=True)
        new_lines = _lammps_oxdna_replace_inputs(self.input_lines, updated_params, seed, variables=self.variables)
        self.input_dir.joinpath(self.input_file_name).write_text("\n".join(new_lines))


def _lammps_oxdna_replace_inputs(  # noqa: C901 TODO: refactor perhaps to class
        input_lines: list[str],
        params: list[dict[str, float]],
        seed: int | None,
        variables: dict[str, Any] | None = None,
    ) -> list[str]:
    variable_replacements = {"seed": seed or np.random.default_rng().integers(0, 2**24), **(variables or {})}
    new_lines = []
    seen = set()
    multiline_buffer = ""
    for input_l in input_lines:
        line = re.sub(r"\s+", " ", input_l.strip())
        if line.endswith("&"):
            multiline_buffer += line.removesuffix("&") + " "
            continue
        if multiline_buffer:
            line = multiline_buffer + line
            multiline_buffer = ""
        if line.startswith("variable "):
            var = line.split()[1]
            if var in variable_replacements:
                line = f"variable {var} equal {variable_replacements.pop(var)}"
        elif line.startswith("dump "):
            line_parts = line.split()
            if len(line_parts) > 6:  # noqa: PLR2004
                fname = line_parts[5]
                fields = set(line_parts[6:])
                if LAMMPS_REQUIRED_FIELDS.issubset(fields) and fname == "trajectory.dat":
                    seen.add("dump_line")
        for key, replacements in REPLACEMENT_MAP.items():
            if line.startswith(key):
                new_parts = _replace_parts_in_line(line.removeprefix(key), replacements, params)
                line = f"{key} {new_parts}"
        new_lines.append(line)
    if "dump_line" not in seen:
        raise ValueError(f"Required dump not found. Must dump to trajectory.dat fields {LAMMPS_REQUIRED_FIELDS}.")
    if variable_replacements:
        raise ValueError("Missing variable for replacements: " + ", ".join(variable_replacements.keys()))
    return new_lines


def _replace_parts_in_line(inputs: str, replacements: tuple[str], params: dict[str, float]) -> str:
    parts = inputs.split()
    def repl(part: str, replacement: str | None) -> str:
        if replacement is None or replacement not in params:
            return part
        return f"{_transform_param(replacement, params[replacement]):f}"
    return " ".join([repl(part, r_param) for part, r_param in zip(parts, replacements, strict=True)])


REPLACEMENT_MAP = {
    "bond_coeff *": ("eps_backbone", "delta_backbone", "r0_backbone"),
    "pair_coeff * * oxdna/excv": (
        "eps_exc",
        "sigma_backbone",
        "dr_star_backbone",
        "eps_exc",
        "sigma_back_base",
        "dr_star_back_base",
        "eps_exc",
        "sigma_base",
        "dr_star_base",
    ),
    "pair_coeff * * oxdna/stk": (
        None,
        None,
        "eps_stack_base",
        "eps_stack_kt_coeff",
        "a_stack",
        "dr0_stack",
        "dr_c_stack",
        "dr_low_stack",
        "dr_high_stack",
        "a_stack_4",
        "theta0_stack_4",
        "delta_theta_star_stack_4",
        "a_stack_5",
        "theta0_stack_5",
        "delta_theta_star_stack_5",
        "a_stack_6",
        "theta0_stack_6",
        "delta_theta_star_stack_6",
        "a_stack_1",
        "neg_cos_phi1_star_stack",
        "a_stack_2",
        "neg_cos_phi2_star_stack",
    ),
    "pair_coeff * * oxdna/hbond": (
        None,
        "HYDR_F1",  # this we don't have replacement for
        "a_hb",
        "dr0_hb",
        "dr_c_hb",
        "dr_low_hb",
        "dr_high_hb",
        "a_hb_1",
        "theta0_hb_1",
        "delta_theta_star_hb_1",
        "a_hb_2",
        "theta0_hb_2",
        "delta_theta_star_hb_2",
        "a_hb_3",
        "theta0_hb_3",
        "delta_theta_star_hb_3",
        "a_hb_4",
        "theta0_hb_4",
        "delta_theta_star_hb_4",
        "a_hb_8",  # 8 and 7 swapped in lammps input
        "theta0_hb_8",
        "delta_theta_star_hb_8",
        "a_hb_7",
        "theta0_hb_7",
        "delta_theta_star_hb_7",
    ),
    "pair_coeff 1 4 oxdna/hbond": (
        None,
        "eps_hb",
        "a_hb",
        "dr0_hb",
        "dr_c_hb",
        "dr_low_hb",
        "dr_high_hb",
        "a_hb_1",
        "theta0_hb_1",
        "delta_theta_star_hb_1",
        "a_hb_2",
        "theta0_hb_2",
        "delta_theta_star_hb_2",
        "a_hb_3",
        "theta0_hb_3",
        "delta_theta_star_hb_3",
        "a_hb_4",
        "theta0_hb_4",
        "delta_theta_star_hb_4",
        "a_hb_8",  # 8 and 7 swapped in lammps input
        "theta0_hb_8",
        "delta_theta_star_hb_8",
        "a_hb_7",
        "theta0_hb_7",
        "delta_theta_star_hb_7",
    ),
    "pair_coeff 2 3 oxdna/hbond": (
        None,
        "eps_hb",
        "a_hb",
        "dr0_hb",
        "dr_c_hb",
        "dr_low_hb",
        "dr_high_hb",
        "a_hb_1",
        "theta0_hb_1",
        "delta_theta_star_hb_1",
        "a_hb_2",
        "theta0_hb_2",
        "delta_theta_star_hb_2",
        "a_hb_3",
        "theta0_hb_3",
        "delta_theta_star_hb_3",
        "a_hb_4",
        "theta0_hb_4",
        "delta_theta_star_hb_4",
        "a_hb_7",
        "theta0_hb_7",
        "delta_theta_star_hb_7",
        "a_hb_8",
        "theta0_hb_8",
        "delta_theta_star_hb_8",
    ),
    "pair_coeff * * oxdna/xstk": (
        "k_cross",
        "r0_cross",
        "dr_c_cross",
        "dr_low_cross",
        "dr_high_cross",
        "a_cross_1",
        "theta0_cross_1",
        "delta_theta_star_cross_1",
        "a_cross_3",  # 3 and 2 swapped in lammps input
        "theta0_cross_3",
        "delta_theta_star_cross_3",
        "a_cross_2",
        "theta0_cross_2",
        "delta_theta_star_cross_2",
        "a_cross_4",
        "theta0_cross_4",
        "delta_theta_star_cross_4",
        "a_cross_8",  # 8 and 7 swapped in lammps input
        "theta0_cross_8",
        "delta_theta_star_cross_8",
        "a_cross_7",
        "theta0_cross_7",
        "delta_theta_star_cross_7",
    ),
    "pair_coeff * * oxdna/coaxstk": (
        "k_coax",
        "dr0_coax",
        "dr_c_coax",
        "dr_low_coax",
        "dr_high_coax",
        "a_coax_1",
        "theta0_coax_1",
        "delta_theta_star_coax_1",
        "a_coax_4",
        "theta0_coax_4",
        "delta_theta_star_coax_4",
        "a_coax_5",
        "theta0_coax_5",
        "delta_theta_star_coax_5",
        "a_coax_6",
        "theta0_coax_6",
        "delta_theta_star_coax_6",
        "a_coax_3p",
        "cos_phi3_star_coax",
        "a_coax_4p",
        "cos_phi4_star_coax",
    ),
}
# Copy common oxdna2 parameters providing overrides where needed
REPLACEMENT_MAP = {
    **REPLACEMENT_MAP,
    **{k.replace("oxdna/", "oxdna2/"): v for k, v in REPLACEMENT_MAP.items() if "oxdna/" in k},
    "pair_coeff * * oxdna2/coaxstk": (
        "k_coax",
        "dr0_coax",
        "dr_c_coax",
        "dr_low_coax",
        "dr_high_coax",
        "a_coax_1",
        "theta0_coax_1",
        "delta_theta_star_coax_1",
        "a_coax_4",
        "theta0_coax_4",
        "delta_theta_star_coax_4",
        "a_coax_5",
        "theta0_coax_5",
        "delta_theta_star_coax_5",
        "a_coax_6",
        "theta0_coax_6",
        "delta_theta_star_coax_6",
        "a_coax_1_f6",
        "b_coax_1_f6",
    ),
    "pair_coeff * * oxdna2/dh": (None, "salt_conc", "q_eff"),
}


def _transform_param(param: str, value: float) -> float:
    if param in ["neg_cos_phi1_star_stack", "neg_cos_phi2_star_stack"]:
        return -value
    return value


LAMMPS_REQUIRED_FIELDS = {
    "x", "y", "z", "vx", "vy", "vz", "c_quat[1]", "c_quat[2]", "c_quat[3]", "c_quat[4]",
    "angmomx", "angmomy", "angmomz"
}

def _transform_lammps_state(state: np.ndarray, fields: str) -> np.ndarray:
    def get_idx(*field_names: str) -> list[int]:
        return [fields.index(name) for name in field_names]
    pos = state[get_idx("x", "y", "z")]
    vel = state[get_idx("vx", "vy", "vz")]
    quat = state[get_idx("c_quat[1]", "c_quat[2]", "c_quat[3]", "c_quat[4]")]
    angmom = state[get_idx("angmomx", "angmomy", "angmomz")]
    vel *= np.sqrt(3.1575)
    angmom /= np.sqrt(0.435179)
    return np.concatenate([pos, _transform_lammps_quat(quat), vel, angmom])


def _transform_lammps_quat(quat: np.ndarray) -> np.ndarray:
    q_2 = quat**2
    i = 1 / q_2.sum()
    a0 = (q_2[0] + q_2[1] - q_2[2] - q_2[3]) * i
    a1 = 2 * (quat[1]*quat[2] + quat[0]*quat[3]) * i
    a2 = 2 * (quat[1]*quat[3] - quat[0]*quat[2]) * i
    b0 = 2 * (quat[1]*quat[3] + quat[0]*quat[2]) * i
    b1 = 2 * (quat[2]*quat[3] - quat[0]*quat[1]) * i
    b2 = (q_2[0] + q_2[3] - q_2[1] - q_2[2]) * i
    return np.array([a0, a1, a2, b0, b1, b2])


def _read_lammps_output(output_file: Path) -> Trajectory:
    """Reads LAMMPS trajectory dump file and extracts the final energy values.

    The file must have been created by a dump LAMMPS dump command similar to:

        compute quat all property/atom quatw quati quatj quatk
        dump {name} all custom {freq} trajectory.dat x y z vx vy vz &
            c_quat[1] c_quat[2] c_quat[3] c_quat[4] angmomx angmomy angmomz

    noting that the above fields are required, but other fields may also be
    present in the dump.

    Args:
        output_file: Path to the LAMMPS trajectory dump file.

    Returns:
        A Trajectory object in mythos format.
    """
    ts = []
    bs = []
    states = []
    num_atoms = None
    with output_file.open() as f:
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                t = float(next(f))
                if t == 0:  # skip initial frame
                    continue
                ts.append(t)
            if not ts:
                continue
            if line.startswith("ITEM: NUMBER OF ATOMS") and num_atoms is None:
                num_atoms = int(next(f))
            elif line.startswith("ITEM: BOX BOUNDS"):
                bounds = " ".join([next(f).replace("\n", " ") for _ in range(3)])
                bx1, bx2, by1, by2, bz1, bz2 = np.fromstring(bounds, dtype=np.float64, sep=" ")
                bs.append(np.array([bx2 - bx1, by2 - by1, bz2 - bz1]))
            elif line.startswith("ITEM: ATOMS"):
                state_fields = line[12:].strip().split()
                if LAMMPS_REQUIRED_FIELDS - set(state_fields):
                    raise ValueError("LAMMPS output file missing required fields.")
                states.append(np.array([
                    _transform_lammps_state(np.fromstring(next(f), dtype=np.float64, sep=" "), state_fields)
                    for _ in range(num_atoms)
                ]))

    validate_box_size(bs)

    return Trajectory(
        n_nucleotides=num_atoms,
        strand_lengths=[num_atoms],  # this is not actually correct
        times=np.array(ts, dtype=np.float64),
        energies=np.zeros((len(ts), 3), dtype=np.float64),  # energies are not parsed here
        states=[NucleotideState(array=s) for s in states],
    )
