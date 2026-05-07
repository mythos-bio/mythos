#!/usr/bin/env python
"""Preprocess a GROMACS topology for use with mythos.

This script wraps `preprocess_topology` so users can prepare their
GROMACS input files from the command line without embedding the logic
in their own scripts.  It optionally copies the input directory to
an archive location before preprocessing.

Example usage::

    python examples/gromacs_prep.py /path/to/gromacs/input
    python examples/gromacs_prep.py /path/to/input --copy-to /path/to/archive
    python examples/gromacs_prep.py /path/to/input --gromacs-binary /usr/local/bin/gmx
    python examples/gromacs_prep.py /path/to/input --params nsteps=5000 dt=0.002
"""

import argparse
import logging

from mythos.simulators.gromacs.utils import preprocess_topology

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_params(param_strings: list[str] | None) -> dict[str, str]:
    """Parse ``key=value`` strings into a dictionary."""
    if not param_strings:
        return {}
    params: dict[str, str] = {}
    for item in param_strings:
        if "=" not in item:
            raise SystemExit(f"Invalid parameter format '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess a GROMACS topology for use with mythos.",
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing the GROMACS input files.",
    )
    parser.add_argument(
        "--copy-to",
        default=None,
        help="Copy input files to this directory before preprocessing (useful for archiving).",
    )
    parser.add_argument(
        "--output-prefix",
        default="preprocessed",
        help="Prefix for the preprocessed .top and .tpr files (default: preprocessed).",
    )
    parser.add_argument(
        "--output-mdp-name",
        default="preprocessed.mdp",
        help="Name of the output .mdp file (default: preprocessed.mdp).",
    )
    parser.add_argument(
        "--gromacs-binary",
        default=None,
        help="Path to the GROMACS binary (default: gmx on PATH).",
    )
    parser.add_argument(
        "--mdp-name",
        default="md.mdp",
        help="Name of the .mdp file in the input directory (default: md.mdp).",
    )
    parser.add_argument(
        "--topology-name",
        default="topol.top",
        help="Name of the topology file (default: topol.top).",
    )
    parser.add_argument(
        "--structure-name",
        default="membrane.gro",
        help="Name of the structure file (default: membrane.gro).",
    )
    parser.add_argument(
        "--index-name",
        default="index.ndx",
        help="Name of the index file (default: index.ndx).",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        metavar="KEY=VALUE",
        help="MDP parameter overrides, e.g. --params nsteps=5000 dt=0.002",
    )
    parser.add_argument(
        "--log-prefix",
        default="topology_preprocess",
        help="Prefix for log messages (default: topology_preprocess).",
    )

    args = parser.parse_args(argv)
    params = parse_params(args.params)

    logger.info("Preprocessing topology in %s", args.input_dir)
    if args.copy_to:
        logger.info("Archiving input to %s", args.copy_to)
    if params:
        logger.info("MDP overrides: %s", params)

    preprocess_topology(
        input_dir=args.input_dir,
        params=params or None,
        copy_to=args.copy_to,
        output_prefix=args.output_prefix,
        output_mdp_name=args.output_mdp_name,
        gromacs_binary=args.gromacs_binary,
        mdp_name=args.mdp_name,
        topology_name=args.topology_name,
        structure_name=args.structure_name,
        index_name=args.index_name,
        log_prefix=args.log_prefix,
    )

    logger.info("Done. Preprocessed files written to %s", args.copy_to or args.input_dir)


if __name__ == "__main__":
    main()
