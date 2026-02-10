"""Test for GROMACS input file reader."""

import importlib
import io
from pathlib import Path

import numpy as np
import pytest

import mythos.input.gromacs_input as gi

TEST_FILES_DIR = Path(__file__).parent / "test_files"
GROMACS_TEST_DATA = importlib.resources.files("mythos").parent / "data" / "test-data" / "gromacs"


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


class TestGromacsParamsParser:
    @pytest.fixture
    def parser(self):
        return gi.GromacsParamsParser(GROMACS_TEST_DATA / "preprocessed_topology.top")

    def test_parse_returns_expected_keys(self, parser):
        result = parser.parse()
        assert set(result.keys()) == {"nonbond_params", "bond_params", "angle_params"}

    def test_parse_nonbond_params(self, parser):
        result = parser.parse()
        nonbond = result["nonbond_params"]
        assert nonbond["lj_sigma_Qda_Qda"] == 0.6
        assert nonbond["lj_epsilon_Qda_Qda"] == 2.7
        assert nonbond["lj_sigma_Qda_P5"] == 0.47
        assert nonbond["lj_epsilon_Qda_P5"] == 0.5

    def test_parse_bond_params(self, parser):
        result = parser.parse()
        bonds = result["bond_params"]
        assert bonds["bond_r0_DMPC_NC3_PO4"] == 0.45
        assert bonds["bond_k_DMPC_NC3_PO4"] == 1250.0
        assert bonds["bond_r0_DMPC_GL1_GL2"] == 0.37
        assert bonds["bond_k_DMPC_GL1_GL2"] == 1250.0

    def test_parse_angle_params(self, parser):
        result = parser.parse()
        angles = result["angle_params"]
        assert angles["angle_theta0_DMPC_PO4_GL1_GL2"] == np.deg2rad(120.0)
        assert angles["angle_k_DMPC_PO4_GL1_GL2"] == 25.0
        assert angles["angle_theta0_DMPC_C1A_C2A_C3A"] == np.deg2rad(180.0)
        assert angles["angle_k_DMPC_C1A_C2A_C3A"] == 35.0

    def test_parse_bead_types(self, parser):
        parser.parse()
        assert set(parser._bead_types) == {"Qda", "Qd", "Qa", "Q0", "P5"}

    def test_replace_bond_params(self, parser, tmp_path):
        output_file = tmp_path / "modified.top"
        new_params = {"bond_r0_DMPC_NC3_PO4": 0.50, "bond_k_DMPC_NC3_PO4": 1500.0}
        parser.replace(new_params, output_file)
        modified = gi.GromacsParamsParser(output_file).parse()
        assert modified["bond_params"]["bond_r0_DMPC_NC3_PO4"] == 0.50
        assert modified["bond_params"]["bond_k_DMPC_NC3_PO4"] == 1500.0

    def test_replace_angle_params(self, parser, tmp_path):
        output_file = tmp_path / "modified.top"
        new_params = {"angle_theta0_DMPC_PO4_GL1_GL2": 110.0, "angle_k_DMPC_PO4_GL1_GL2": 30.0}
        parser.replace(new_params, output_file)
        modified = gi.GromacsParamsParser(output_file).parse()
        assert modified["angle_params"]["angle_theta0_DMPC_PO4_GL1_GL2"] == 110.0
        assert modified["angle_params"]["angle_k_DMPC_PO4_GL1_GL2"] == 30.0

    def test_replace_nonbond_params(self, parser, tmp_path):
        output_file = tmp_path / "modified.top"
        new_params = {"lj_sigma_Qda_Qda": 0.65, "lj_epsilon_Qda_Qda": 3.0}
        parser.replace(new_params, output_file)
        modified = gi.GromacsParamsParser(output_file).parse()
        assert modified["nonbond_params"]["lj_sigma_Qda_Qda"] == 0.65
        assert modified["nonbond_params"]["lj_epsilon_Qda_Qda"] == 3.0

    def test_replace_preserves_unmodified_params(self, parser, tmp_path):
        output_file = tmp_path / "modified.top"
        new_params = {"bond_r0_DMPC_NC3_PO4": 0.55}
        parser.replace(new_params, output_file)
        modified = gi.GromacsParamsParser(output_file).parse()
        # Modified param changed
        assert modified["bond_params"]["bond_r0_DMPC_NC3_PO4"] == 0.55
        # Unmodified params preserved
        assert modified["bond_params"]["bond_k_DMPC_NC3_PO4"] == 1250.0
        assert modified["nonbond_params"]["lj_sigma_Qda_Qda"] == 0.6

    def test_replace_multiple_param_types(self, parser, tmp_path):
        output_file = tmp_path / "modified.top"
        new_params = {
            "bond_r0_DMPC_NC3_PO4": 0.48,
            "angle_k_DMPC_PO4_GL1_GL2": 28.0,
            "lj_epsilon_Qda_P5": 0.6,
        }
        parser.replace(new_params, output_file)
        modified = gi.GromacsParamsParser(output_file).parse()
        assert modified["bond_params"]["bond_r0_DMPC_NC3_PO4"] == 0.48
        assert modified["angle_params"]["angle_k_DMPC_PO4_GL1_GL2"] == 28.0
        assert modified["nonbond_params"]["lj_epsilon_Qda_P5"] == 0.6

    def test_replace_preserves_non_parameter_lines(self, parser, tmp_path):
        output_file = tmp_path / "modified.top"
        parser.replace({"bond_r0_DMPC_NC3_PO4": 0.50}, output_file)
        content = output_file.read_text()
        assert "; DRY MARTINI v2.1" in content
        assert "[ defaults ]" in content
        assert "[ atomtypes ]" in content
        assert "[moleculetype]" in content
        assert "[ system ]" in content
        assert "[ molecules ]" in content
        assert "DMPC            64" in content

    def test_replace_in_place(self, tmp_path):
        import shutil
        topology_copy = tmp_path / "topology.top"
        shutil.copy(GROMACS_TEST_DATA / "preprocessed_topology.top", topology_copy)
        parser = gi.GromacsParamsParser(topology_copy)
        # get the original and ensure it's different from the new value we will
        # set as a sanity check
        bond_params = parser.parse()["bond_params"]
        assert bond_params["bond_r0_DMPC_NC3_PO4"] != 0.52, "unexpected test data"
        parser.replace({"bond_r0_DMPC_NC3_PO4": 0.52}, topology_copy)
        modified = gi.GromacsParamsParser(topology_copy).parse()
        assert modified["bond_params"]["bond_r0_DMPC_NC3_PO4"] == 0.52

    def test_nonbond_params_raises_for_unknown_bead_types(self, tmp_path):
        # Create a minimal topology with nonbond_params referencing unknown types
        topology_content = """; Test topology with unknown bead types
        [ defaults ]
        1      2         no        1.0     1.0

        [ atomtypes ]
        ; only define TypeA, not TypeB
        TypeA     72.0 0.0  A     0.0 0.0

        [ nonbond_params ]
        ; TypeB is not defined in atomtypes
            TypeA TypeB    1    0.6000 2.7000
        """
        topology_file = tmp_path / "bad_topology.top"
        topology_file.write_text(topology_content)

        parser = gi.GromacsParamsParser(topology_file)
        with pytest.raises(ValueError, match="Unknown atom types in nonbond_params"):
            parser.parse()


class TestReadParamsFromTopology:
    """Tests for read_params_from_topology function."""

    def test_read_params_from_topology(self):
        result = gi.read_params_from_topology(GROMACS_TEST_DATA / "preprocessed_topology.top")

        assert set(result.keys()) == {"nonbond_params", "bond_params", "angle_params"}
        assert result["nonbond_params"]["lj_sigma_Qda_Qda"] == 0.6
        assert result["bond_params"]["bond_r0_DMPC_NC3_PO4"] == 0.45
        assert result["angle_params"]["angle_theta0_DMPC_PO4_GL1_GL2"] == np.deg2rad(120.0)


class TestReplaceParamsInTopology:
    """Tests for replace_params_in_topology function."""

    def test_replace_params_in_topology(self, tmp_path):
        output_file = tmp_path / "modified.top"
        new_params = {
            "bond_r0_DMPC_NC3_PO4": 0.50,
            "lj_sigma_Qda_Qda": 0.65,
            "angle_k_DMPC_PO4_GL1_GL2": 30.0,
        }

        gi.replace_params_in_topology(
            GROMACS_TEST_DATA / "preprocessed_topology.top",
            new_params,
            output_file,
        )

        # Verify the modifications
        modified = gi.read_params_from_topology(output_file)
        assert modified["bond_params"]["bond_r0_DMPC_NC3_PO4"] == 0.50
        assert modified["nonbond_params"]["lj_sigma_Qda_Qda"] == 0.65
        assert modified["angle_params"]["angle_k_DMPC_PO4_GL1_GL2"] == 30.0
        # Unmodified params preserved
        assert modified["bond_params"]["bond_k_DMPC_NC3_PO4"] == 1250.0
