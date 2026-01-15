"""Test for GROMACS input file reader."""

import io
from pathlib import Path

import pytest

import mythos.input.gromacs_input as gi

TEST_FILES_DIR = Path(__file__).parent / "test_files"


class TestParseNumeric:
    """Tests for _parse_numeric function."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("1", (1, True)),
            ("0", (0, True)),
            ("-1", (-1, True)),
            ("100", (100, True)),
            ("-1.5", (-1.5, True)),
            ("3.14159", (3.14159, True)),
            ("0.002", (0.002, True)),
            ("-0.5", (-0.5, True)),
            ("1e10", (1e10, True)),
            ("1.5e-3", (1.5e-3, True)),
            ("1.5.5", (0, False)),
            ("abc", (0, False)),
            ("", (0, False)),
            ("one", (0, False)),
        ],
    )
    def test_parse_numeric(self, value: str, expected: tuple[float | int, bool]) -> None:
        """Test _parse_numeric function with various inputs."""
        assert gi._parse_numeric(value) == expected


class TestParseBoolean:
    """Tests for _parse_boolean function."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # True values
            ("yes", (True, True)),
            ("YES", (True, True)),
            ("Yes", (True, True)),
            ("true", (True, True)),
            ("TRUE", (True, True)),
            ("True", (True, True)),
            ("on", (True, True)),
            ("ON", (True, True)),
            ("On", (True, True)),
            # False values
            ("no", (False, True)),
            ("NO", (False, True)),
            ("No", (False, True)),
            ("false", (False, True)),
            ("FALSE", (False, True)),
            ("False", (False, True)),
            ("off", (False, True)),
            ("OFF", (False, True)),
            ("Off", (False, True)),
            # Invalid values
            ("yess", (False, False)),
            ("noo", (False, False)),
            ("truee", (False, False)),
            ("1", (False, False)),
            ("0", (False, False)),
            ("", (False, False)),
        ],
    )
    def test_parse_boolean(self, value: str, expected: tuple[bool, bool]) -> None:
        """Test _parse_boolean function with various inputs."""
        assert gi._parse_boolean(value) == expected


class TestParseValue:
    """Tests for _parse_value function."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # Integers
            ("1", 1),
            ("0", 0),
            ("-10", -10),
            ("50000", 50000),
            # Floats
            ("-1.5", -1.5),
            ("0.002", 0.002),
            ("3.14", 3.14),
            ("1e-5", 1e-5),
            # Booleans
            ("yes", True),
            ("no", False),
            ("true", True),
            ("false", False),
            ("on", True),
            ("off", False),
            # Strings
            ("string", "string"),
            ("md", "md"),
            ("v-rescale", "v-rescale"),
            ("PME", "PME"),
            ("Verlet", "Verlet"),
            ("h-bonds", "h-bonds"),
            # With comments
            ("md ; integrator type", "md"),
            ("100 ; number of steps", 100),
            ("yes ; enable feature", True),
            # Empty
            ("", ""),
            ("   ", ""),
            ("; just a comment", ""),
        ],
    )
    def test_parse_value(self, value: str, expected: str | float | bool) -> None:
        """Test _parse_value function with various inputs."""
        assert gi._parse_value(value) == expected


class TestRead:
    """Tests for read function."""

    def test_read_basic(self) -> None:
        """Test reading a basic GROMACS .mdp file."""
        result = gi.read_mdp(TEST_FILES_DIR / "test_gromacs.mdp")

        # Check some key values
        assert result["integrator"] == "md"
        assert result["dt"] == 0.002
        assert result["nsteps"] == 50000
        assert result["tcoupl"] == "v-rescale"
        assert result["ref-t"] == 300
        assert not result["pcoupl"]
        assert result["gen-vel"]
        assert not result["continuation"]
        assert not result["pcouple"]

    def test_read_all_keys(self) -> None:
        """Test that all expected keys are read from the file."""
        result = gi.read_mdp(TEST_FILES_DIR / "test_gromacs.mdp")

        expected_keys = {
            "integrator",
            "dt",
            "nsteps",
            "tcoupl",
            "tc-grps",
            "tau-t",
            "ref-t",
            "pcoupl",
            "nstxout",
            "nstvout",
            "nstenergy",
            "nstlog",
            "cutoff-scheme",
            "nstlist",
            "ns-type",
            "pbc",
            "rlist",
            "coulombtype",
            "rcoulomb",
            "vdwtype",
            "rvdw",
            "constraints",
            "gen-vel",
            "gen-temp",
            "gen-seed",
            "continuation",
            "pcouple",
        }

        assert expected_keys.issubset(result.keys())

    def test_read_skips_comments(self) -> None:
        """Test that comment lines are skipped."""
        result = gi.read_mdp(TEST_FILES_DIR / "test_gromacs.mdp")

        # Comments should not appear as keys
        assert "GROMACS test input file" not in result
        assert "This is a comment line" not in result
        assert "Temperature coupling" not in result


class TestWriteTo:
    """Tests for write_to function."""

    @pytest.mark.parametrize(
        ("input_dict", "expected"),
        [
            ({"test": 100}, "test = 100\n"),
            ({"test": "string"}, "test = string\n"),
            ({"test": 1.5}, "test = 1.5\n"),
            ({"test": 0.002}, "test = 0.002\n"),
            ({"test": True}, "test = yes\n"),
            ({"test": False}, "test = no\n"),
        ],
    )
    def test_write_to_single_value(self, input_dict: dict, expected: str) -> None:
        """Test write_to function with single values."""
        text_stream = io.StringIO()
        gi.write_mdp_to(input_dict, text_stream)
        assert text_stream.getvalue() == expected

    def test_write_to_multiple_values(self) -> None:
        """Test write_to function with multiple values."""
        input_dict = {
            "integrator": "md",
            "dt": 0.002,
            "nsteps": 50000,
            "gen-vel": True,
            "pcoupl": False,
        }
        expected = "integrator = md\n" "dt = 0.002\n" "nsteps = 50000\n" "gen-vel = yes\n" "pcoupl = no\n"

        text_stream = io.StringIO()
        gi.write_mdp_to(input_dict, text_stream)
        assert text_stream.getvalue() == expected


class TestWrite:
    """Tests for write function."""

    def test_write_basic(self, tmp_path: Path) -> None:
        """Test writing a basic GROMACS .mdp file."""
        in_config = {
            "integrator": "md",
            "dt": 0.002,
            "nsteps": 50000,
            "gen-vel": True,
            "pcoupl": False,
        }

        expected = "integrator = md\n" "dt = 0.002\n" "nsteps = 50000\n" "gen-vel = yes\n" "pcoupl = no\n"

        output_file = tmp_path / "test.mdp"
        gi.write_mdp(in_config, output_file)

        assert output_file.read_text() == expected

    def test_write_read_roundtrip(self, tmp_path: Path) -> None:
        """Test that writing and reading gives consistent results."""
        original_config = {
            "integrator": "md",
            "dt": 0.002,
            "nsteps": 50000,
            "ref-t": 300,
            "gen-vel": True,
            "pcoupl": False,
            "cutoff-scheme": "Verlet",
        }

        output_file = tmp_path / "roundtrip.mdp"
        gi.write_mdp(original_config, output_file)
        read_config = gi.read_mdp(output_file)

        assert read_config == original_config


class TestIntegration:
    """Integration tests for gromacs_input module."""

    def test_read_modify_write(self, tmp_path: Path) -> None:
        """Test reading a file, modifying values, and writing back."""
        # Read original file
        config = gi.read_mdp(TEST_FILES_DIR / "test_gromacs.mdp")

        # Modify some values
        config["nsteps"] = 100000
        config["ref-t"] = 310
        config["gen-vel"] = False

        # Write to new file
        output_file = tmp_path / "modified.mdp"
        gi.write_mdp(config, output_file)

        # Read back and verify modifications
        modified_config = gi.read_mdp(output_file)
        assert modified_config["nsteps"] == 100000
        assert modified_config["ref-t"] == 310
        assert not modified_config["gen-vel"]
