import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxtyping as jaxtyp
import pytest

import mythos.utils.helpers as jdh


def pytree_equal(tree1, tree2):
    """Check if two PyTrees have the same structure and values."""
    # Check if the structures match
    if jax.tree.structure(tree1) != jax.tree.structure(tree2):
        return False

    # Check if the values match
    def values_equal(x, y):
        return jnp.array_equal(x, y)

    return all(jax.tree.flatten(jax.tree.map(values_equal, tree1, tree2))[0])


@pytest.mark.parametrize(
    ("in_iter", "n", "out_iter"),
    [
        (
            "ABCDEFG",
            3,
            [("A", "B", "C"), ("D", "E", "F"), ("G",)],
        ),
    ],
)
def test_batched(in_iter, n, out_iter):
    assert list(jdh.batched(in_iter, n)) == out_iter


def test_batched_raises_value_error():
    with pytest.raises(ValueError, match=jdh.ERR_BATCHED_N):
        list(jdh.batched("ABCDEFG", 0))


@pytest.mark.parametrize(
    ("trees", "expected_pytree"),
    [
        (
            [{"a": jnp.array([1, 2]), "b": jnp.array([3, 4])}, {"a": jnp.array([5, 6]), "b": jnp.array([7, 8])}],
            {"a": jnp.array([[1, 2], [5, 6]]), "b": jnp.array([[3, 4], [7, 8]])},
        ),
    ],
)
def test_tree_stack(trees: list[jaxtyp.PyTree], expected_pytree: jaxtyp.PyTree):
    stacked_pytree = jdh.tree_stack(trees)
    assert pytree_equal(stacked_pytree, expected_pytree)


@pytest.mark.parametrize(
    ("trees", "expected_pytree"),
    [
        (
            [{"a": jnp.array([1, 2]), "b": jnp.array([3, 4])}, {"a": jnp.array([5, 6]), "b": jnp.array([7, 8])}],
            {"a": jnp.array([1, 2, 5, 6]), "b": jnp.array([3, 4, 7, 8])},
        ),
    ],
)
def test_tree_concatenate(trees: list[jaxtyp.PyTree], expected_pytree: jaxtyp.PyTree):
    concatenated_pytree = jdh.tree_concatenate(trees)
    assert pytree_equal(concatenated_pytree, expected_pytree)


@pytest.mark.parametrize(
    ("lines", "n", "expected"),
    [
        (["line1", "line2", "line3", "line4", "line5"], 3, "line3\nline4\nline5"),
        (["line1", "line2"], 5, "line1\nline2"),
        (["only_one"], 1, "only_one"),
        ([], 3, ""),
        (["a", "b", "c"], 0, ""),
    ],
)
def test_tail_file(lines: list[str], n: int, expected: str, tmp_path: Path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("\n".join(lines) + ("\n" if lines else ""))
    result = jdh.tail_file(test_file, n=n)
    assert result == expected


def test_run_command_success(tmp_path: Path):
    """Test that run_command succeeds for a valid command."""
    jdh.run_command(["echo", "hello"], cwd=tmp_path, log_prefix="test")
    out_file = tmp_path / "test.out.log"
    assert out_file.exists()
    assert out_file.read_text().strip() == "hello"


@pytest.mark.parametrize("err_tail_lines", [1, 5, 10, 20])
def test_run_command_failure_includes_tailed_lines(err_tail_lines: int, tmp_path: Path, monkeypatch):
    """Test that run_command raises RuntimeError with tailed output on failure."""
    total_lines = 30

    def mock_check_call(cmd, cwd, shell, stdout, stderr):
        # Write lines to stdout and stderr files
        for i in range(1, total_lines + 1):
            stdout.write(f"stdout line {i}\n")
            stderr.write(f"stderr line {i}\n")
        stdout.flush()
        stderr.flush()
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "check_call", mock_check_call)

    with pytest.raises(RuntimeError) as exc_info:
        jdh.run_command(["fake_command"], cwd=tmp_path, log_prefix="fail-test", err_tail_lines=err_tail_lines)

    error_msg = str(exc_info.value)
    assert f"Last {err_tail_lines} lines of stdout" in error_msg
    assert f"Last {err_tail_lines} lines of stderr" in error_msg

    # Verify the correct number of lines are included
    # The last err_tail_lines lines should be present
    for i in range(total_lines - err_tail_lines + 1, total_lines + 1):
        assert f"stdout line {i}" in error_msg
        assert f"stderr line {i}" in error_msg

    # Lines before the tail should NOT be present
    if err_tail_lines < total_lines:
        assert "stdout line 1\n" not in error_msg
        assert "stderr line 1\n" not in error_msg
