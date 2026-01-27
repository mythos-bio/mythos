"""GROMACS input file parser.

Handles reading and writing of GROMACS .mdp (molecular dynamics parameter) files.
The format is a simple key = value format, similar to oxDNA input files.
"""

import io
from pathlib import Path

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


def update_mdp_params(mdp_file: Path, params: dict) -> None:
    """Update parameters in a GROMACS .mdp file.

    Args:
        mdp_file: Path to the .mdp file to update.
        params: Dictionary of parameters to update.
    """
    config = read_mdp(mdp_file)
    config.update(params)
    write_mdp(config, mdp_file)
