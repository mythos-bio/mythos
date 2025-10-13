"""Tests for the ui.logger module."""

import datetime
import itertools

import pytest

from jax_dna.ui.loggers.console import ConsoleLogger
from jax_dna.ui.loggers.disk import FileLogger, PerMetricFileLogger, convert_to_fname
from jax_dna.ui.loggers.logger import Logger, NullLogger, Status, StatusKind
from jax_dna.ui.loggers.multilogger import MultiLogger


@pytest.mark.parametrize(
    ("fname", "expected"),
    [
        ("test", "test.csv"),
        ("test/first", "test_first.csv"),
        ("test/first/second", "test_first_second.csv"),
        ("test first", "test_first.csv"),
        ("test first second", "test_first_second.csv"),
        ("test first/second", "test_first_second.csv"),
    ],
)
def test_convert_to_fname(fname: str, expected: str) -> None:
    """Test the convert_to_fname function."""
    assert convert_to_fname(fname) == expected


@pytest.mark.parametrize(
    ("name", "value", "step"),
    [
        ("test", 1.0, 0),
        ("test", 1.0, 1),
    ],
)
def test_log_metric(name: str, value: float, step: int, tmp_path) -> None:
    """Test the log_metric function."""
    log = PerMetricFileLogger(log_dir=tmp_path)

    log.log_metric(name, value, step)
    fname = tmp_path / convert_to_fname(name)
    assert fname.exists()
    fields = fname.read_text().strip().split(",")
    assert fields[0] == str(step)
    assert fields[2] == str(value)
    datetime.datetime.fromisoformat(fields[1])  # raises if not valid isoformat


@pytest.mark.parametrize(
        ("method"), [
            ("update_objective_status"),
            ("update_observable_status"),
            ("update_simulator_status"),
        ]
    )
def test_objective_status_updates(tmp_path, method: str) -> None:  # -- for testing
    """Test the objective status update functions."""

    name = f"test_{method}"
    log = PerMetricFileLogger(log_dir=tmp_path)

    for status in Status:
        getattr(log, method)(name, status)
        fname = tmp_path / convert_to_fname(name)
        assert fname.exists()
        fields = fname.read_text().strip().split()[-1].split(",")  # get last line
        assert fname.stem == name
        datetime.datetime.fromisoformat(fields[0])  # raises if not valid isoformat
        assert fields[1] == str(status)


def test_single_file_logger(tmp_path) -> None:
    """Test the FileLogger class."""
    fname = tmp_path / "all_logs.csv"
    log = FileLogger(log_file=fname, mode="w")

    log.log_metric("A", 1.0, 0)
    log.log_metric("B", 2.0, 1)

    assert fname.exists()

    lines = fname.read_text().strip().split("\n")
    assert len(lines) == 2
    fields = lines[0].split(",")
    assert fields[0] == str(0)
    datetime.datetime.fromisoformat(fields[1])  # raises if not valid isoformat
    assert fields[2] == "A"
    assert fields[3] == str(1.0)

    fields = lines[1].split(",")
    assert fields[0] == str(1)
    datetime.datetime.fromisoformat(fields[1])  # raises if not valid isoformat
    assert fields[2] == "B"
    assert fields[3] == str(2.0)

    # test append mode here, since we already have an existing file
    log2 = FileLogger(log_file=fname, mode="a")
    log2.log_metric("C", 3.0, 2)
    lines = fname.read_text().strip().split("\n")
    assert len(lines) == 3

def test_null_logger() -> None:
    """Test the NullLogger class."""
    log = NullLogger()

    # just test that these methods do not raise
    log.log_metric("A", 1.0, 0)
    log.update_status("A", StatusKind.SIMULATOR, Status.STARTED)
    log.set_simulator_started("A")
    log.set_simulator_running("A")
    log.set_simulator_complete("A")
    log.set_simulator_error("A")
    log.update_simulator_status("A", Status.ERROR)
    log.update_objective_status("A", Status.ERROR)
    log.set_objective_started("A")
    log.set_objective_running("A")
    log.set_objective_complete("A")
    log.set_objective_error("A")

def test_multilogger() -> None:
    """Test the MultiLogger class."""
    class DummyLogger(Logger):
        def __init__(self):
            self.calls = []

        def log_metric(self, name: str, value: float, step: int) -> None:
            self.calls.append(("log_metric", name, value, step))

        def update_status(self, name: str, kind: StatusKind, status: Status) -> None:
            self.calls.append(("update_status", name, kind, status))

    log1 = DummyLogger()
    log2 = DummyLogger()
    log = MultiLogger([log1, log2])

    log.log_metric("A", 1.0, 0)
    assert len(log1.calls) == 1
    assert len(log2.calls) == 1
    assert log1.calls[-1] == ("log_metric", "A", 1.0, 0)
    assert log2.calls[-1] == ("log_metric", "A", 1.0, 0)

    log.update_status("A", StatusKind.SIMULATOR, Status.STARTED)
    assert len(log1.calls) == 2
    assert len(log2.calls) == 2
    assert log1.calls[-1] == ("update_status", "A", StatusKind.SIMULATOR, Status.STARTED)
    assert log2.calls[-1] == ("update_status", "A", StatusKind.SIMULATOR, Status.STARTED)

    # ensure all dispatch methods send logs as expected. Since we do not
    # override individual methods, they all should dispatch to update_status
    # with the appropriate kind.
    kind_map = {
        "simulator": StatusKind.SIMULATOR,
        "objective": StatusKind.OBJECTIVE,
        "observable": StatusKind.OBSERVABLE,
    }
    status_map = {
        "started": Status.STARTED,
        "running": Status.RUNNING,
        "complete": Status.COMPLETE,
        "error": Status.ERROR,
    }
    for kind, status in itertools.product(kind_map.keys(), status_map.keys()):
        getattr(log, f"set_{kind}_{status}")("A")
        assert log1.calls[-1] == ("update_status", "A", kind_map[kind], status_map[status])
        assert log2.calls[-1] == ("update_status", "A", kind_map[kind], status_map[status])



def test_console_logger(capsys) -> None:
    """Test the ConsoleLogger class."""
    log = ConsoleLogger()

    log.log_metric("A", 1.0, 0)
    captured = capsys.readouterr()
    assert "Step: 0, A: 1.0" in captured.out

    log.update_simulator_status("A", Status.STARTED)
    captured = capsys.readouterr()
    assert "A Status.STARTED" in captured.out
