"""Tests for the oxdna_utils module."""

import importlib
from pathlib import Path

import mythos.simulators.oxdna.utils as oxdna_utils
import pytest
from mythos.input.trajectory import Trajectory


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", 1),
        ("1.0f", 1.0),
        ("TeSt", "TeSt"),
    ],
)
def test_parse_value_in(value: str, expected: int | float | str) -> None:  # noqa: PYI041 -- this is for test documentation
    assert oxdna_utils._parse_value_in(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "1"),
        (1.0, "1.0f"),
        ("TeSt", "TeSt"),
    ],
)
def test_parse_value_out(value: int | float | str, expected: str) -> None:  # noqa: PYI041 -- this is for test documentation
    assert oxdna_utils._parse_value_out(value) == expected


def test_parse_value_out_raises() -> None:
    with pytest.raises(TypeError, match=oxdna_utils.ERR_INVALID_HEADER_TYPE[:10]):
        assert oxdna_utils._parse_value_out([1, 2, 3])


def test_read_src_h() -> None:
    expected = {
        "HYDR_THETA8_T0": 1.5707963267948966,
        "HYDR_T3_MESH_POINTS": "HYDR_T2_MESH_POINTS",
        "CXST_T5_MESH_POINTS": 6,
        "FENE_DELTA": 2.0,
        "FENE_R0_OXDNA": 0.0,
        "FENE_R0_OXDNA2": 0.0,
        "CXST_THETA1_SA": 0.0,
    }
    path = Path(__file__).parent / "test_data" / "test.model.h"
    assert oxdna_utils.read_src_h(path) == expected


def test_write_src_h() -> None:
    params = {
        "FENE_DELTA": 5.0,
        "FENE_R0_OXDNA": 0.756,
        "FENE_R0_OXDNA2": 0.756,
        "HYDR_THETA8_T0": 1.5707963267948966,
        "HYDR_T3_MESH_POINTS": "HYDR_T2_MESH_POINTS",
        "CXST_T5_MESH_POINTS": 6,
        "CXST_THETA1_SA": 20.0,
    }
    test_dir = Path(__file__).parent / "test_data"
    path = test_dir / "out.model.h"
    oxdna_utils.write_src_h(path, params)

    actual = path.read_text().splitlines()[-11:]
    expected = (test_dir / "expected.model.h").read_text().splitlines()[-11:]
    # remove generated file
    path.unlink()

    assert actual == expected


def test_update_params() -> None:
    params = [
        {
            "delta_backbone": 5.0,
            "theta0_hb_8": 1.5707963267948966,
            "a_coax_1_f6": 40.0,  # oxdna2 specific param
            "r0_backbone": 0.756,  # oxdna2 param which shares name with oxdna1 param
        },
        {},
    ]

    # copy test.model.h to tmp file
    test_dir = Path(__file__).parent / "test_data"
    path = test_dir / "test.model.h"
    tmp_path = test_dir / "tmp.model.h"
    tmp_path.write_text(path.read_text())
    oxdna_utils.update_params(tmp_path, params)
    actual = tmp_path.read_text().splitlines()[-8:]
    expected = (test_dir / "expected.model.h").read_text().splitlines()[-8:]
    # remove generated file
    tmp_path.unlink()
    assert actual == expected


def test_read_output_trajectory() -> None:
    test_input = importlib.resources.files("mythos").parent / "data" / "test-data" / "simple-helix" / "input"
    traj = oxdna_utils.read_output_trajectory(test_input)

    assert isinstance(traj, Trajectory)
    assert traj.state_rigid_body.center.shape == (100, 16, 3)


if __name__ == "__main__":
    test_write_src_h()
