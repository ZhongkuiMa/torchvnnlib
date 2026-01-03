"""Fixtures for unit tests in test_units directory."""

import pytest

from torchvnnlib._backend import get_backend


@pytest.fixture(params=["torch", "numpy"])
def backend_name(request, torch_available):
    """Parametrize tests to run with both torch and numpy backends."""
    if request.param == "torch" and not torch_available:
        pytest.skip("PyTorch not available")
    return request.param


@pytest.fixture
def backend(backend_name):
    """Get backend instance by name."""
    return get_backend(backend_name)


@pytest.fixture
def sample_input_vars():
    """Sample input variable names for testing."""
    return ["X_0", "X_1", "X_2"]


@pytest.fixture
def sample_output_vars():
    """Sample output variable names for testing."""
    return ["Y_0", "Y_1"]


@pytest.fixture
def simple_vnnlib_content():
    """Return simple vnnlib content for basic testing."""
    return """
; Test property
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)

(assert (and (<= X_0 1.0) (>= X_0 0.0)))
(assert (and (<= X_1 1.0) (>= X_1 0.0)))
(assert (<= Y_0 0.5))
"""


# TYPE2 Processor Test Fixtures


@pytest.fixture
def type2_simple_lines():
    """Return simple TYPE2 VNN-LIB test lines.

    TYPE2 structure: (and ...input_constraints... (or ...output_constraints...))
    """
    return [
        "(declare-const X_0 Real)",
        "(declare-const Y_0 Real)",
        "(declare-const Y_1 Real)",
        "(assert (>= X_0 0.5))",
        "(assert (<= X_0 1.0))",
        "(assert (or (>= Y_0 Y_1) (>= Y_1 Y_0)))",
    ]


@pytest.fixture
def type2_multiple_inputs_lines():
    """TYPE2 with multiple input variables."""
    return [
        "(declare-const X_0 Real)",
        "(declare-const X_1 Real)",
        "(declare-const X_2 Real)",
        "(declare-const Y_0 Real)",
        "(declare-const Y_1 Real)",
        "(assert (>= X_0 -1.0))",
        "(assert (<= X_0 2.0))",
        "(assert (>= X_1 0.0))",
        "(assert (<= X_1 1.5))",
        "(assert (>= X_2 -0.5))",
        "(assert (<= X_2 0.5))",
        "(assert (or (>= Y_0 0.5) (<= Y_1 -0.5)))",
    ]


@pytest.fixture
def type2_multiple_or_branches_lines():
    """TYPE2 with multiple OR branches."""
    return [
        "(declare-const X_0 Real)",
        "(declare-const Y_0 Real)",
        "(declare-const Y_1 Real)",
        "(assert (>= X_0 0.0))",
        "(assert (<= X_0 1.0))",
        "(assert (or (and (>= Y_0 0.5) (<= Y_1 0.3)) (and (>= Y_1 Y_0) (<= Y_0 0.2)) (and (= Y_0 Y_1))))",
    ]


@pytest.fixture
def type2_parsed_data():
    """Pre-parsed TYPE2 data structure matching parse_simple_patterns output.

    This represents the output of parse_simple_patterns for a TYPE2 file.
    """
    return {
        "simple_input_bounds": [
            (">=", "X_", 0, 0.5),
            ("<=", "X_", 0, 1.0),
        ],
        "complex_lines": [
            "(assert (or (>= Y_0 Y_1) (>= Y_1 Y_0)))",
        ],
    }


@pytest.fixture
def assert_tensor_close(backend):
    """Provide a callable to assert two tensors are close.

    Returns a callable that compares tensors with small tolerance for floating point.
    """

    def _assert_close(actual, expected, rtol=1e-5, atol=1e-8):
        """Assert tensors are close within tolerance."""
        actual_np = backend.to_numpy(actual) if hasattr(backend, "to_numpy") else actual
        expected_np = backend.to_numpy(expected) if hasattr(backend, "to_numpy") else expected

        import numpy as np

        np.testing.assert_allclose(
            actual_np,
            expected_np,
            rtol=rtol,
            atol=atol,
            err_msg=f"Tensors not close:\nActual: {actual_np}\nExpected: {expected_np}",
        )

    return _assert_close


# TYPE1 Processor Test Fixtures


@pytest.fixture
def type1_simple_lines():
    """Return simple TYPE1 VNN-LIB test lines.

    TYPE1 structure: (and ...input_constraints... ...output_constraints...)
    All constraints are simple (no OR/AND nesting).
    """
    return [
        "(declare-const X_0 Real)",
        "(declare-const Y_0 Real)",
        "(declare-const Y_1 Real)",
        "(assert (>= X_0 0.5))",
        "(assert (<= X_0 1.0))",
        "(assert (>= Y_0 Y_1))",
        "(assert (<= Y_1 0.5))",
    ]


@pytest.fixture
def type1_complex_output_lines():
    """TYPE1 with complex output expressions (nested AND/OR).

    Tests the AST parsing path for complex output constraints.
    """
    return [
        "(declare-const X_0 Real)",
        "(declare-const X_1 Real)",
        "(declare-const Y_0 Real)",
        "(declare-const Y_1 Real)",
        "(assert (>= X_0 -1.0))",
        "(assert (<= X_0 2.0))",
        "(assert (>= X_1 0.0))",
        "(assert (<= X_1 1.5))",
        "(assert (or (and (>= Y_0 0.5) (<= Y_1 0.3)) (and (>= Y_1 Y_0) (<= Y_0 0.2))))",
    ]


@pytest.fixture
def type1_parsed_data_simple():
    """Pre-parsed TYPE1 data for simple case.

    Output from parse_simple_patterns for TYPE1 with simple constraints.
    """
    return {
        "simple_input_bounds": [
            (">=", "X_", 0, 0.5),
            ("<=", "X_", 0, 1.0),
        ],
        "simple_output_constrs": [
            (">=", "Y_", 0, "Y_", 1),
        ],
        "simple_output_bounds": [
            ("<=", "Y_", 1, 0.5),
        ],
        "complex_lines": [],
        "complex_indices": [],
    }


@pytest.fixture
def type1_parsed_data_complex():
    """Pre-parsed TYPE1 data with complex expressions.

    Output from parse_simple_patterns for TYPE1 with complex output constraints.
    """
    return {
        "simple_input_bounds": [
            (">=", "X_", 0, -1.0),
            ("<=", "X_", 0, 2.0),
            (">=", "X_", 1, 0.0),
            ("<=", "X_", 1, 1.5),
        ],
        "simple_output_constrs": [],
        "simple_output_bounds": [],
        "complex_lines": [
            "(assert (or (and (>= Y_0 0.5) (<= Y_1 0.3)) (and (>= Y_1 Y_0) (<= Y_0 0.2))))",
        ],
        "complex_indices": [4],
    }
