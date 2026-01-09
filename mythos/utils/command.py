import subprocess
from collections import deque
from pathlib import Path


def tail_file(path: Path, n: int = 10) -> list[str]:
    buffer = deque(maxlen=n)
    with path.open("r") as f:
        for line in f:
            buffer.append(line.rstrip("\n"))
    return list(buffer)


def run_command(cmd: list[str], cwd: Path):
    out_file = cwd / "command.out.log"
    err_file = cwd / "command.err.log"
    try:
        with out_file.open("w") as f_out, err_file.open("w") as f_err:
            subprocess.check_call(cmd, cwd=cwd, shell=False, stdout=f_out, stderr=f_err)
    except subprocess.CalledProcessError as e:
        err_lines = tail_file(err_file, n=20)
        out_lines = tail_file(out_file, n=20)
        raise RuntimeError(
            f"Command {' '.join(cmd)} failed with exit code {e.returncode}.\n"
            f"Last 20 lines of stdout:\n" + "\n".join(out_lines) + "\n"
            "Last 20 lines of stderr:\n" + "\n".join(err_lines)
        ) from e
