"""Test for oxDNA input file reader."""

import io
from pathlib import Path

import jax.numpy as jnp
import pytest

import mythos.input.oxdna_input as oi

TEST_FILES_DIR = Path(__file__).parent / "test_files"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", (1, True)),
        ("-1.5", (-1.5, True)),
        ("1.5.5", (0, False)),
    ],
)
def test_parse_numeric(value: str, expected: tuple[float | int, bool]) -> None:
    """Test _parse_numeric function."""
    assert oi._parse_numeric(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", (True, True)),
        ("false", (False, True)),
        ("TRUE", (True, True)),
        ("fALSe", (False, True)),
        ("Truee", (False, False)),
    ],
)
def test_parse_boolean(value: str, expected: tuple[bool, bool]) -> None:
    """Test _parse_boolean function."""
    assert oi._parse_boolean(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", 1),
        ("-1.5", -1.5),
        ("true", True),
        ("false", False),
        ("string", "string"),
        ("1-numString", "1-numString"),
    ],
)
def test_parse_value(value: str, expected: str | float | bool) -> None:  # noqa: FBT001 args can be non bool
    """Test _parse_value function."""
    assert oi._parse_value(value) == expected


def test_read() -> None:
    expected = {
        "T": "296.15K",
        "steps": 10000,
        "conf_file": "init.conf",
        "topology": "sys.top",
        "trajectory_file": "output.dat",
        "time_scale": "linear",
        "print_conf_interval": 100,
        "print_energy_every": 100,
        "interaction_type": "DNA_nomesh",
        "seed": 0,
        "lastconf_file": "last_conf.dat",
        "list_type": "no",
        "restart_step_counter": True,
        "energy_file": "energy.dat",
        "equilibration_steps": 0,
    }

    assert oi.read(TEST_FILES_DIR / "test_oxdna_simple_helix_input_trunc.txt") == expected


@pytest.mark.parametrize(
    ("input_dict", "expected"),
    [
        ({"test": 100}, "test = 100\n"),
        ({"test": "string"}, "test = string\n"),
        ({"test": 1.5}, "test = 1.5\n"),
        ({"test": True}, "test = true\n"),
        ({"test": {"inside": False}}, "test = {\ninside = false\n}\n"),
    ],
)
def test_write_to(input_dict: dict, expected: str) -> None:
    text_stream = io.StringIO()
    oi.write_to(input_dict, text_stream)
    assert text_stream.getvalue() == expected


def test_write(tmpdir) -> None:
    in_config = {"test": 100, "test2": "string", "test3": 1.5, "test4": True, "test5": {"inside": False}}

    expected = "test = 100\ntest2 = string\ntest3 = 1.5\ntest4 = true\ntest5 = {\ninside = false\n}\n"

    with Path(tmpdir / "test.txt").open("w") as f:
        oi.write_to(in_config, f)

    with Path(tmpdir / "test.txt").open("r") as f:
        assert f.read() == expected


# --- Tests for read_box_size ---


def test_read_box_size(tmp_path: Path) -> None:
    conf = tmp_path / "init.conf"
    conf.write_text("t = 0.0\nb = 10.0 20.0 30.0\nE = 0.0 0.0 0.0\n")
    result = oi.read_box_size(conf)
    expected = jnp.array([10.0, 20.0, 30.0])
    assert jnp.allclose(result, expected)


def test_read_box_size_existing_file() -> None:
    result = oi.read_box_size(TEST_FILES_DIR / "simple-helix-8bp-5steps.conf")
    expected = jnp.array([50.0, 50.0, 50.0])
    assert jnp.allclose(result, expected)


def test_read_box_size_missing_raises(tmp_path: Path) -> None:
    conf = tmp_path / "no_box.conf"
    conf.write_text("t = 0.0\nE = 0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="No 'b = ...' line found"):
        oi.read_box_size(conf)


# --- Tests for read_input_dir ---


@pytest.fixture()
def oxdna_input_dir(tmp_path: Path) -> Path:
    """Create a minimal oxDNA input directory."""
    # topology: 6 nucleotides, 2 strands
    (tmp_path / "sys.top").write_text("6 2\n1 A -1 1\n1 C 0 2\n1 G 1 -1\n2 C -1 4\n2 G 3 5\n2 T 4 -1\n")
    # configuration
    (tmp_path / "init.conf").write_text(
        "t = 0.0\n"
        "b = 25.0 25.0 25.0\n"
        "E = 0.0 0.0 0.0\n"
        "0 0 0 1 0 0 0 1 0 0 0 0 0 0 0\n"
        "1 0 0 1 0 0 0 1 0 0 0 0 0 0 0\n"
        "2 0 0 1 0 0 0 1 0 0 0 0 0 0 0\n"
        "3 0 0 1 0 0 0 1 0 0 0 0 0 0 0\n"
        "4 0 0 1 0 0 0 1 0 0 0 0 0 0 0\n"
        "5 0 0 1 0 0 0 1 0 0 0 0 0 0 0\n"
    )
    # input file
    (tmp_path / "input").write_text("T = 300K\nsteps = 1000\nconf_file = init.conf\ntopology = sys.top\n")
    return tmp_path


def test_read_input_dir(oxdna_input_dir: Path) -> None:
    result = oi.read_input_dir(oxdna_input_dir)

    assert result.topology.n_nucleotides == 6
    assert len(result.topology.strand_counts) == 2
    assert jnp.allclose(result.box_size, jnp.array([25.0, 25.0, 25.0]))
    assert result.kT == pytest.approx(oi.get_kt_from_string("300K"))
    assert result.config["steps"] == 1000


def test_read_input_dir_custom_input_file(oxdna_input_dir: Path) -> None:
    (oxdna_input_dir / "custom_input").write_text("T = 350K\nsteps = 500\nconf_file = init.conf\ntopology = sys.top\n")
    result = oi.read_input_dir(oxdna_input_dir, input_file="custom_input")
    assert result.kT == pytest.approx(oi.get_kt_from_string("350K"))
    assert result.config["steps"] == 500
