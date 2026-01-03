"""Unit tests for fast type processor functions (TYPE2-5).

Tests the public processor APIs for VNNLIB lines input.
Follows principles: broad code coverage, all logics, error handling, quick tests,
fine-grained information, public API only, no deselected tests.
"""

import pytest

from torchvnnlib._backend import get_backend
from torchvnnlib.fast_type import process_type3, process_type4, process_type5


@pytest.fixture(params=["torch", "numpy"])
def backend(request):
    """Get backend for testing."""
    return get_backend(request.param)


# TYPE2 processor tests require specific parsed_data format from parse_simple_patterns
# These are tested indirectly through the benchmark/correctness tests


class TestType3Processor:
    """Test TYPE3 processor - OR inputs with simple outputs."""

    def test_process_type3_basic(self, backend):
        """Test basic TYPE3 processing."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (<= X_0 0.3) (>= X_0 0.7)))",
            "(assert (<= Y_0 0.5))",
        ]
        result = process_type3(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None
        assert isinstance(result, list)

    def test_process_type3_multiple_or_clauses(self, backend):
        """Test TYPE3 with multiple OR clauses."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (<= X_0 0.2) (<= X_0 0.5) (<= X_0 0.8)))",
            "(assert (<= Y_0 0.5))",
        ]
        result = process_type3(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None

    def test_process_type3_multiple_inputs(self, backend):
        """Test TYPE3 with multiple inputs."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (<= X_0 0.3) (<= X_1 0.2)) (and (>= X_0 0.7) (>= X_1 0.8))))",
            "(assert (<= Y_0 0.5))",
        ]
        result = process_type3(lines, n_inputs=2, n_outputs=1, backend=backend)
        assert result is not None

    def test_process_type3_multiple_outputs(self, backend):
        """Test TYPE3 with multiple outputs."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (or (<= X_0 0.3) (>= X_0 0.7)))",
            "(assert (and (<= Y_0 0.5) (<= Y_1 0.3)))",
        ]
        result = process_type3(lines, n_inputs=1, n_outputs=2, backend=backend)
        assert result is not None

    def test_process_type3_with_verbose(self, backend):
        """Test TYPE3 with verbose flag."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (<= X_0 0.3) (>= X_0 0.7)))",
            "(assert (<= Y_0 0.5))",
        ]
        result = process_type3(lines, n_inputs=1, n_outputs=1, backend=backend, verbose=True)
        assert result is not None

    def test_process_type3_negative_bounds(self, backend):
        """Test TYPE3 with negative bounds."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (<= X_0 0.3) (>= X_0 -0.7)))",
            "(assert (and (<= Y_0 0.5) (>= Y_0 -0.5)))",
        ]
        result = process_type3(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None

    def test_process_type3_and_in_or(self, backend):
        """Test TYPE3 with AND inside OR."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (<= X_0 0.3) (>= X_0 0.1)) (and (<= X_0 0.7) (>= X_0 0.5))))",
            "(assert (<= Y_0 0.5))",
        ]
        result = process_type3(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None


class TestType4Processor:
    """Test TYPE4 processor - OR inputs and OR outputs."""

    def test_process_type4_basic(self, backend):
        """Test basic TYPE4 processing."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (<= X_0 0.3) (>= X_0 0.7)))",
            "(assert (or (<= Y_0 0.3) (>= Y_0 0.7)))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None
        assert isinstance(result, list)

    def test_process_type4_multiple_clauses(self, backend):
        """Test TYPE4 with multiple OR clauses."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (<= X_0 0.2) (<= X_0 0.5) (<= X_0 0.8)))",
            "(assert (or (<= Y_0 0.2) (<= Y_0 0.5) (<= Y_0 0.8)))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None

    def test_process_type4_multiple_variables(self, backend):
        """Test TYPE4 with multiple variables."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (or (and (<= X_0 0.3) (<= X_1 0.2)) (and (>= X_0 0.7) (>= X_1 0.8))))",
            "(assert (or (and (<= Y_0 0.5) (<= Y_1 0.3)) (and (>= Y_0 0.7) (>= Y_1 0.6))))",
        ]
        result = process_type4(lines, n_inputs=2, n_outputs=2, backend=backend)
        assert result is not None

    def test_process_type4_with_verbose(self, backend):
        """Test TYPE4 with verbose flag."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (<= X_0 0.3) (>= X_0 0.7)))",
            "(assert (or (<= Y_0 0.3) (>= Y_0 0.7)))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend, verbose=True)
        assert result is not None

    def test_process_type4_negative_bounds(self, backend):
        """Test TYPE4 with negative bounds."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (<= X_0 0.3) (>= X_0 -0.7)))",
            "(assert (or (<= Y_0 0.5) (>= Y_0 -0.5)))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None

    def test_process_type4_scientific_notation(self, backend):
        """Test TYPE4 with scientific notation."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (<= X_0 1.5e-10) (>= X_0 2.3e-5)))",
            "(assert (or (<= Y_0 1.0e-3) (>= Y_0 2.5e-4)))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None


class TestType5Processor:
    """Test TYPE5 processor - top-level OR."""

    def test_process_type5_basic(self, backend):
        """Test basic TYPE5 processing."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (<= X_0 0.3) (<= Y_0 0.5))",
            "            (and (>= X_0 0.7) (>= Y_0 0.7))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None
        assert isinstance(result, list)

    def test_process_type5_multiple_clauses(self, backend):
        """Test TYPE5 with multiple OR clauses."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (<= X_0 0.2) (<= Y_0 0.3))",
            "            (and (<= X_0 0.5) (<= Y_0 0.5))",
            "            (and (>= X_0 0.8) (>= Y_0 0.8))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None

    def test_process_type5_multiple_variables(self, backend):
        """Test TYPE5 with multiple variables."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (or (and (<= X_0 0.3) (<= X_1 0.2) (<= Y_0 0.5) (<= Y_1 0.3))",
            "            (and (>= X_0 0.7) (>= X_1 0.8) (>= Y_0 0.7) (>= Y_1 0.6))))",
        ]
        result = process_type5(lines, n_inputs=2, n_outputs=2, backend=backend)
        assert result is not None

    def test_process_type5_with_verbose(self, backend):
        """Test TYPE5 with verbose flag."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (<= X_0 0.3) (<= Y_0 0.5))",
            "            (and (>= X_0 0.7) (>= Y_0 0.7))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend, verbose=True)
        assert result is not None

    def test_process_type5_negative_bounds(self, backend):
        """Test TYPE5 with negative bounds."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (<= X_0 0.3) (<= Y_0 0.5))",
            "            (and (>= X_0 -0.7) (>= Y_0 -0.5))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None

    def test_process_type5_scientific_notation(self, backend):
        """Test TYPE5 with scientific notation."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (<= X_0 1.5e-10) (<= Y_0 1.0e-3))",
            "            (and (>= X_0 2.3e-5) (>= Y_0 2.5e-4))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None

    def test_process_type5_mixed_constraints(self, backend):
        """Test TYPE5 with mixed constraint types."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (<= X_0 0.5) (>= Y_0 0.1))",
            "            (and (>= X_0 0.6) (<= Y_0 0.9))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)
        assert result is not None
