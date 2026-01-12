"""Helper functions for the mythos package."""

import itertools
import subprocess
import sys
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping as jaxtyp

ERR_BATCHED_N = "n must be at least one"


def batched(iterable: Iterable[Any], n: int) -> Iterable[Any]:
    """Batch an iterable into chunks of size n.

    Args:
        iterable (iter[Any]): iterable to batch
        n (int): batch size

    Returns:
        iter[Any]: batched iterable
    """
    if sys.version_info >= (3, 12):
        batch_f = itertools.batched
    else:
        # taken from https://docs.python.org/3/library/itertools.html#itertools.batched
        def batch_f(iterable: Iterable[Any], n: int) -> Iterable[Any]:
            # batched('ABCDEFG', 3) → ABC DEF G
            if n < 1:
                raise ValueError(ERR_BATCHED_N)
            it = iter(iterable)
            while batch := tuple(itertools.islice(it, n)):
                yield batch

    return batch_f(iterable, n)


def tree_stack(trees: list[jaxtyp.PyTree]) -> jaxtyp.PyTree:
    """Stacks corresponding leaves of PyTrees into arrays along a new axis."""
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def tree_concatenate(trees: list[jaxtyp.PyTree]) -> jaxtyp.PyTree:
    """Concatenates corresponding leaves of PyTrees along the first axis."""
    return jax.tree.map(lambda *v: jnp.concatenate(v), *trees)


def tail_file(path: Path, n: int = 10) -> str:
    """Return the last n lines of a file as a string."""
    buffer = deque(maxlen=n)
    with path.open("r") as f:
        for line in f:
            buffer.append(line.rstrip("\n"))
    return "\n".join(buffer)


def run_command(cmd: list[str], cwd: Path, log_prefix: str = "command-output", err_tail_lines: int = 20) -> None:
    """Run a command in a subprocess, raising RuntimeError on failure.

    Stderr and stdout are captured to files in the `cwd` directory, named with
    provided prefix. If the process fails with a CalledProcessError, the last
    `tail_lines` of each log file are included in the raised RuntimeError.

    Args:
        cmd (list[str]): command and arguments to run
        cwd (Path): working directory to run the command in
        log_prefix (str): prefix for the output log files, within the `cwd` directory
        err_tail_lines (int): number of lines from the end of each log file to
            include in the error message on failure
    """
    out_file = Path(cwd) / f"{log_prefix}.out.log"
    err_file = Path(cwd) / f"{log_prefix}.err.log"
    try:
        with out_file.open("w") as f_out, err_file.open("w") as f_err:
            subprocess.check_call(cmd, cwd=cwd, shell=False, stdout=f_out, stderr=f_err)
    except subprocess.CalledProcessError as e:
        err_lines = tail_file(err_file, n=err_tail_lines)
        out_lines = tail_file(out_file, n=err_tail_lines)
        raise RuntimeError(
            f"Command {' '.join(cmd)} failed with exit code {e.returncode}.\n"
            f"  Last {err_tail_lines} lines of stdout:\n{out_lines}\n"
            f"  Last {err_tail_lines} lines of stderr:\n{err_lines}\n"
        ) from e
