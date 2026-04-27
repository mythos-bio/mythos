"""oxDNA input file parser."""

from __future__ import annotations

import io
import typing
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

from mythos.input import topology as _topology
from mythos.utils.units import get_kt_from_string

INVALID_DICT_LINE = "Invalid dictionary line: {}"


def _parse_numeric(value: str) -> tuple[float | int, bool]:
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
    lowered = value.lower()
    return (
        lowered == "true",
        lowered in ("true", "false"),
    )


def _parse_value(value: str) -> str | float | int | bool:
    # remove potential comment from end of line
    value = value.split("#", maxsplit=1)[0].strip()
    parsed, is_numeric = _parse_numeric(value)
    if not is_numeric:
        parsed, is_boolean = _parse_boolean(value)
        if not is_boolean:
            parsed = value

    return parsed


def _parse_dict(
    in_line: str, lines: typing.Iterable[str]
) -> tuple[dict[str, str | float | int | bool], typing.Iterable[str]]:
    if "=" not in in_line and "{" not in in_line:
        raise ValueError(INVALID_DICT_LINE.format(in_line))

    var_name = in_line.split("=", maxsplit=1)[0].strip()
    parsed = {}
    for line in lines:
        if "{" not in line and "}" not in line:
            key, value = (v.strip() for v in line.split("="))
            parsed[key] = _parse_value(value)
        elif "{" in line:
            (key, value), lines = _parse_dict(line, lines)
            parsed[key] = value
        elif "}":
            break

    return (var_name, parsed), lines


def read(input_file: Path) -> dict[str, str | float | int | bool]:
    """Read an oxDNA input file."""
    with input_file.open("r") as f:
        lines = filter(lambda line: (len(line.strip()) > 0) and (not line.strip().startswith("#")), f.readlines())

    parsed = {}
    for line in lines:
        if "{" in line:
            (key, value), lines = _parse_dict(line, lines)
        else:
            key, str_value = (v.strip() for v in line.split("="))
            value = _parse_value(str_value)

        parsed[key] = value

    return parsed


def write_to(input_config: dict, f: io.TextIOWrapper) -> None:
    """Write an oxDNA input file."""
    for key, value in input_config.items():
        if isinstance(value, dict):
            f.write(f"{key} = {{\n")
            write_to(value, f)
            f.write("}\n")
        else:
            if key == "T" and isinstance(value, float):
                parsed_value = str(value) + "K"
            elif isinstance(value, bool):
                parsed_value = str(value).lower()
            else:
                parsed_value = str(value)

            f.write(f"{key} = {parsed_value}\n")


def read_box_size(conf_file: Path) -> jnp.ndarray:
    """Read the box size from an oxDNA configuration file.

    Parses the ``b = ...`` line from the configuration file header.

    Args:
        conf_file: Path to the oxDNA configuration (``.conf`` / ``.dat``) file.

    Returns:
        A JAX array of shape ``(3,)`` with the box dimensions.

    Raises:
        ValueError: If no ``b = ...`` line is found in the file.
    """
    with conf_file.open("r") as f:
        for line in f:
            if line.startswith("b ="):
                return jnp.array([float(v) for v in line.split("=")[1].strip().split()])
    raise ValueError(f"No 'b = ...' line found in {conf_file}")


@dataclass
class oxDNAInputData:
    """Data loaded from an oxDNA input directory.

    Attributes:
        topology: The parsed topology.
        kT: The simulation temperature in oxDNA energy units.
        box_size: Box dimensions as a JAX array of shape ``(3,)``.
        config: The full parsed input-file dictionary.
    """

    topology: _topology.Topology
    kT: float
    box_size: jnp.ndarray
    config: dict[str, typing.Any]


def read_input_dir(
    input_dir: Path,
    input_file: str = "input",
) -> oxDNAInputData:
    """Load topology, temperature and box size from an oxDNA input directory.

    Reads the oxDNA ``input`` file (or the name given by *input_file*),
    extracts the topology, simulation temperature (``kT``), and box
    dimensions from the configuration files referenced therein.

    Args:
        input_dir: Directory containing the oxDNA input files.
        input_file: Name of the input file inside *input_dir*.

    Returns:
        An :class:`oxDNAInputData` with the parsed values.
    """
    input_dir = Path(input_dir)
    config = read(input_dir / input_file)
    top = _topology.from_oxdna_file(input_dir / config.get("topology", "sys.top"))
    kT = get_kt_from_string(str(config["T"]))
    box_size = read_box_size(input_dir / config["conf_file"])
    return oxDNAInputData(topology=top, kT=kT, box_size=box_size, config=config)


def write(input_config: dict, input_file: Path) -> None:
    """Write an oxDNA input file."""
    with input_file.open("w") as f:
        write_to(input_config, f)
