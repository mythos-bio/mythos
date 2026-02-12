"""GROMACS input file parser.

Handles reading and writing of GROMACS .mdp (molecular dynamics parameter) files.
The format is a simple key = value format, similar to oxDNA input files.
"""

import io
import logging
from pathlib import Path
from typing import TypeAlias

import numpy as np

# Type alias for the parameters dictionary

logger = logging.getLogger(__name__)
ParamsDict: TypeAlias = dict[str, float]
INVALID_LINE = "Invalid line: {}"


def _parse_numeric(value: str) -> tuple[float | int, bool]:
    """Try to parse a value as a numeric type."""
    is_successful = False
    parsed = 0
    for t in (int, float):
        try:
            parsed = t(value)
        except ValueError:
            continue
        else:
            is_successful = True
            break

    return parsed, is_successful


def _parse_boolean(value: str) -> tuple[bool, bool]:
    """Try to parse a value as a boolean."""
    lowered = value.lower()
    return (
        lowered in ("yes", "true", "on"),
        lowered in ("yes", "true", "on", "no", "false", "off"),
    )


def _parse_value(value: str) -> str | float | int | bool:
    """Parse a value string, handling comments and type inference."""
    # Remove potential comment from end of line (GROMACS uses ; for comments)
    value = value.split(";")[0].strip()

    if not value:
        return ""

    parsed, is_numeric = _parse_numeric(value)
    if not is_numeric:
        parsed, is_boolean = _parse_boolean(value)
        if not is_boolean:
            parsed = value

    return parsed


def read_mdp(input_file: Path) -> dict[str, str | float | int | bool]:
    """Read a GROMACS .mdp input file.

    Args:
        input_file: Path to the .mdp file.

    Returns:
        Dictionary of key-value pairs from the input file.
    """
    parsed = {}
    with Path(input_file).open("r") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Skip empty lines and comment lines
            if not line or line.startswith(";"):
                continue

            # Handle key = value format
            if "=" in line:
                key, str_value = (v.strip() for v in line.split("=", 1))
                parsed[key] = _parse_value(str_value)

    return parsed


def write_mdp_to(input_config: dict, f: io.TextIOWrapper) -> None:
    """Write a GROMACS .mdp input configuration to a file handle.

    Args:
        input_config: Dictionary of configuration key-value pairs.
        f: File handle to write to.
    """
    for key, value in input_config.items():
        # GROMACS uses yes/no for booleans
        parsed_value = ("yes" if value else "no") if isinstance(value, bool) else str(value)
        f.write(f"{key} = {parsed_value}\n")


def write_mdp(input_config: dict, input_file: Path) -> None:
    """Write a GROMACS .mdp input file.

    Args:
        input_config: Dictionary of configuration key-value pairs.
        input_file: Path to write the .mdp file.
    """
    with Path(input_file).open("w") as f:
        write_mdp_to(input_config, f)


def update_mdp_params(mdp_file: Path, params: dict, out_file: Path|None = None) -> None:
    """Update parameters in a GROMACS .mdp file.

    Args:
        mdp_file: Path to the .mdp file to update.
        params: Dictionary of parameters to update.
        out_file: Optional path to write the updated .mdp file. By default
            overwrites the original file.
    """
    config = read_mdp(mdp_file)
    config.update(params)
    out_file = out_file or mdp_file
    write_mdp(config, out_file)


class GromacsParamsParser:
    """Parser and parameter replacer for params in GROMACS topology files.

    Reads in a preprocessed topology file, extracts parameters into structured
    dictionaries. When writing, it replaces parameters in the original file with
    values from a provided dictionary, preserving other content.

    In both cases, it is important the topology file is preprocessed to in order
    that macros have been expanded.
    """

    def __init__(self, filename: str | Path) -> None:
        self.file = Path(filename)

    def _parser_init(self) -> None:
        self._bead_types: list[str] = []
        # Current molecule state
        self._current_molname: str | None = None
        self._current_atom_types: dict[int, str] = {}
        self._current_atom_names: dict[int, str] = {}
        # Parameters dictionary
        self._bond_params: dict[str, float] = {}
        self._angle_params: dict[str, float] = {}
        self._nonbond_params: dict[str, float] = {}
        # Current section being parsed
        self._current_section: str | None = None
        # Write mode state
        self._write_mode = False
        self._replacement_params: ParamsDict = {}
        self._output_lines: list[str] = []

    def parse(self) -> dict[str, ParamsDict]:
        """Parse topology content and return structured data.

        Returns:
            Dictionary with keys 'nonbond_params', 'bond_params', 'angle_params',
            each mapping parameter names to values.

        Raises:
            ValueError: If nonbond_params references unknown atom types.
        """
        self._parser_init()
        self._write_mode = False

        for line in self.file.open():
            self._process_line(line)

        logger.debug("Found %d bead types: %s", len(self._bead_types), self._bead_types)
        logger.debug(
            "Parsed %d parameters",
            len(self._nonbond_params) + len(self._bond_params) + len(self._angle_params),
        )

        return {
            "nonbond_params": self._nonbond_params,
            "bond_params": self._bond_params,
            "angle_params": self._angle_params,
        }

    def replace(self, params: ParamsDict, output_file: str | Path) -> None:
        """Write topology with replaced parameters to a new file.

        Reads the original topology file and writes a new file with
        parameters replaced from the provided dictionary.

        Args:
            params: Dictionary mapping parameter names to new values.
                Keys should match the format from parse():
                - "bond_k_MOLNAME_ATOMI_ATOMJ", "bond_r0_MOLNAME_ATOMI_ATOMJ"
                - "angle_k_MOLNAME_ATOMI_ATOMJ_ATOMK", "angle_theta0_MOLNAME_ATOMI_ATOMJ_ATOMK"
                - "lj_sigma_TYPE1_TYPE2", "lj_epsilon_TYPE1_TYPE2"
            output_file: Path to write the modified topology.
        """
        self._parser_init()
        self._write_mode = True
        self._replacement_params = params

        with self.file.open("r") as f:
            for line in f:
                self._process_line(line)

        Path(output_file).write_text("".join(self._output_lines))
        logger.debug("Wrote modified topology to %s", output_file)

    def _process_line(self, line: str) -> None:
        stripped = line.strip()

        # Preserve empty lines and comments as-is
        if not stripped or stripped.startswith(";"):
            if self._write_mode:
                self._output_lines.append(line)
            return

        if stripped.startswith("["):
            self._handle_section_header(stripped)
            if self._write_mode:
                self._output_lines.append(line)
            return

        self._handle_section_data(stripped, line)


    def _handle_section_header(self, stripped: str) -> None:
        section_name = stripped.replace(" ", "").strip("[]").lower()

        if section_name == "moleculetype":
            # Reset molecule state for new molecule
            self._current_molname = None
            self._current_atom_types = {}
            self._current_atom_names = {}

        self._current_section = section_name

    def _handle_section_data(self, stripped: str, original_line: str) -> None:
        parts = stripped.split()
        if not parts:
            return

        section = self._current_section
        output_line = original_line  # Default: keep original line

        if section == "atomtypes":
            self._bead_types.append(parts[0])
        elif section == "nonbond_params":
            output_line = self._handle_nonbond_params(parts, original_line)
        elif section == "moleculetype":
            self._current_molname = parts[0]
            self._current_section = None
        elif self._current_molname is not None:
            output_line = self._handle_molecule_section_data(section, parts, original_line)

        if self._write_mode:
            self._output_lines.append(output_line)

    def _handle_molecule_section_data(
        self, section: str | None, parts: list[str], original_line: str
    ) -> str:
        if section == "atoms":
            self._current_atom_types[int(parts[0])] = parts[1]
            self._current_atom_names[int(parts[0])] = parts[4]
            return original_line

        if section == "bonds":
            # Bonds are defined by atom names from the index list read by atoms
            # above. Format is:
            #     atom_i atom_j funct length k
            atom_i = self._current_atom_names[int(parts[0])]
            atom_j = self._current_atom_names[int(parts[1])]
            k_key = f"bond_k_{self._current_molname}_{atom_i}_{atom_j}"
            r0_key = f"bond_r0_{self._current_molname}_{atom_i}_{atom_j}"

            if self._write_mode:
                k = self._replacement_params.get(k_key, float(parts[4]))
                r0 = self._replacement_params.get(r0_key, float(parts[3]))
                return f"    {parts[0]} {parts[1]} {parts[2]} {r0} {k}\n"

            self._bond_params[k_key] = float(parts[4])
            self._bond_params[r0_key] = float(parts[3])
            return original_line

        if section == "angles":
            # Angles are defined by atom names from the index list read by atoms
            # above. Format is:
            #     atom_i atom_j atom_k funct theta0 k
            atom_i = self._current_atom_names[int(parts[0])]
            atom_j = self._current_atom_names[int(parts[1])]
            atom_k = self._current_atom_names[int(parts[2])]
            theta0_key = f"angle_theta0_{self._current_molname}_{atom_i}_{atom_j}_{atom_k}"
            k_key = f"angle_k_{self._current_molname}_{atom_i}_{atom_j}_{atom_k}"

            # Convert theta0 from degrees to radians for internal storage, and
            # back to degrees when writing since GROMACS uses degrees.
            theta0_rad = np.deg2rad(float(parts[4]))
            if self._write_mode:
                theta0 = np.rad2deg(self._replacement_params.get(theta0_key, theta0_rad))
                k = self._replacement_params.get(k_key, float(parts[5]))
                return f"    {parts[0]} {parts[1]} {parts[2]} {parts[3]} {theta0} {k}\n"

            self._angle_params[theta0_key] = theta0_rad
            self._angle_params[k_key] = float(parts[5])
            return original_line

        return original_line

    def _handle_nonbond_params(self, parts: list[str], original_line: str) -> str:
        # format is:
        #     type_i type_j func sigma epsilon
        # check against pre-defined bead types to ensure known types
        type_set = set(self._bead_types)
        type_i = parts[0]
        type_j = parts[1]
        if type_i not in type_set or type_j not in type_set:
            msg = f"Unknown atom types in nonbond_params: {type_i}, {type_j}"
            raise ValueError(msg)

        sigma_key = f"lj_sigma_{type_i}_{type_j}"
        epsilon_key = f"lj_epsilon_{type_i}_{type_j}"

        if self._write_mode:
            sigma = self._replacement_params.get(sigma_key, float(parts[3]))
            epsilon = self._replacement_params.get(epsilon_key, float(parts[4]))
            return f"    {type_i} {type_j} {parts[2]} {sigma} {epsilon}\n"

        # we only store one half pair
        self._nonbond_params[sigma_key] = float(parts[3])
        self._nonbond_params[epsilon_key] = float(parts[4])
        return original_line


def read_params_from_topology(topology_file: Path) -> dict[str, ParamsDict]:
    """Read a preprocessed GROMACS topology file.

    This parses the [atomtypes] section to get bead types, the
    [nonbond_params] section for nonbonded parameters, and all [moleculetype]
    sections for bonds and angles.

    Parameters are stored with descriptive keys:
    - Bonds: "bond_k_MOLNAME_ATOMI_ATOMJ" and "bond_r0_MOLNAME_ATOMI_ATOMJ"
    - Angles: "angle_k_MOLNAME_ATOMI_ATOMJ_ATOMK" and "angle_theta0_MOLNAME_ATOMI_ATOMJ_ATOMK"
    - Nonbonded: "lj_sigma_TYPE1_TYPE2" and "lj_epsilon_TYPE1_TYPE2"

    Args:
        topology_file: Path to the preprocessed topology file.

    Returns:
        Dictionary with keys 'nonbond_params', 'bond_params', 'angle_params'.
    """
    logger.debug("Reading preprocessed topology from %s", topology_file)
    return GromacsParamsParser(topology_file).parse()


def replace_params_in_topology(topology_file: Path, params: ParamsDict, output_file: Path) -> None:
    """Write a modified GROMACS topology file with replaced parameters.

    Reads an existing topology file and writes a new file with
    parameters replaced from the provided dictionary.

    Args:
        topology_file: Path to the input preprocessed topology file.
        params: Dictionary mapping parameter names to new values.
        output_file: Path to write the modified topology.
    """
    logger.debug("Writing modified topology from %s to %s", topology_file, output_file)
    GromacsParamsParser(topology_file).replace(params, output_file)
