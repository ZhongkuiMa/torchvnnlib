"""Tests for the TYPE4 VNN-LIB processor (_type4_processor.py).

TYPE4 structure: OR(AND) inputs + OR(AND) outputs.
Structure: (and (or ...input_constraints...) (or ...output_constraints...))

Covers: Cartesian-product structure, output constraint shapes, numeric values,
verbose output substrings, fallback paths, edge cases.
"""

__docformat__ = "restructuredtext"

import math

import pytest

from torchvnnlib.fast_type._type4_processor import process_type4


class TestType4BasicProcessing:
    """Test basic TYPE4 processing and return-value structure."""

    def test_returns_single_and_group(self, backend):
        """Result always has exactly one AND group (outer list length 1)."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0)) (and (<= X_0 1.0) (>= X_0 0.6))))",
            "(assert (or (and (<= Y_0 0.3)) (and (>= Y_0 0.7))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert isinstance(result, list), "result must be a list"
        assert len(result) == 1, "exactly one AND group expected"
        assert isinstance(result[0], list), "AND group must be a list of tuples"

    def test_each_element_is_two_tuple(self, backend):
        """Each element of the OR properties list is a (input_bounds, output_constrs) tuple."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0))))",
            "(assert (or (and (<= Y_0 0.3))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)

        for item in result[0]:
            assert isinstance(item, tuple), "each OR property must be a tuple"
            assert len(item) == 2, "tuple must have exactly (input_bounds, output_constrs)"
            _input_bounds, _output_constrs = item
            assert isinstance(_output_constrs, list), "output_constrs must be a list"

    def test_input_bounds_shape(self, backend):
        """input_bounds has shape (n_inputs, 2) for each OR property."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0) (<= X_1 1.0) (>= X_1 -1.0))))",
            "(assert (or (and (<= Y_0 0.3))))",
        ]
        result = process_type4(lines, n_inputs=2, n_outputs=1, backend=backend)

        for _input_bounds, _output_constrs in result[0]:
            assert tuple(_input_bounds.shape) == (2, 2), (
                f"input_bounds shape must be (2, 2), got {tuple(_input_bounds.shape)}"
            )

    def test_output_constr_width(self, backend):
        """Each output constraint tensor has width n_outputs + 1."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0))))",
            "(assert (or (and (<= Y_0 0.3)) (and (>= Y_1 0.1))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=2, backend=backend)

        for _input_bounds, output_constrs in result[0]:
            for constr in output_constrs:
                assert constr.shape[1] == 3, (
                    f"each output constraint tensor must have width 3 (n_outputs+1), "
                    f"got {constr.shape[1]}"
                )


class TestType4CartesianProductStructure:
    """Test that the Cartesian-product structure is correct for TYPE4."""

    def test_single_input_or_group_single_output_or_group(self, backend):
        """One input OR group and one output OR group => one tuple in result[0]."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0))))",
            "(assert (or (and (<= Y_0 0.3))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) == 1, "one input group x one output group => 1 tuple"

    def test_two_input_or_groups_shares_same_output_constr_list(self, backend):
        """Two input OR groups: result[0] has 2 tuples, each with the SAME output_constrs list."""
        lines = [
            "(assert (or (and (<= X_0 0.3) (>= X_0 0.0)) (and (<= X_0 1.0) (>= X_0 0.7))))",
            "(assert (or (and (<= Y_0 0.5))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)

        # Each input group is paired with the same output_constrs_list
        assert len(result[0]) == 2, "two input groups => 2 tuples in result[0]"
        _ib0, oc0 = result[0][0]
        _ib1, oc1 = result[0][1]
        assert oc0 is oc1, (
            "both tuples must share the identical output_constrs_list object (Cartesian product)"
        )

    def test_numeric_input_bounds_values(self, backend):
        """Verify specific numeric values in input_bounds after parsing."""
        lines = [
            "(assert (or (and (<= X_0 0.3) (>= X_0 0.1))))",
            "(assert (or (and (<= Y_0 0.5))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) >= 1
        input_bounds, _output_constrs = result[0][0]
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert abs(float(ib[0, 0]) - 0.1) < 1e-9, (
            f"X_0 lower bound should be 0.1, got {float(ib[0, 0])}"
        )
        assert abs(float(ib[0, 1]) - 0.3) < 1e-9, (
            f"X_0 upper bound should be 0.3, got {float(ib[0, 1])}"
        )


class TestType4FallbackPath:
    """Test the fallback when fewer than two OR blocks are found in lines."""

    def test_no_or_blocks_returns_nan_input_zeros_output(self, backend):
        """Lines with no OR structure trigger the fallback in parse_dual_or_blocks.

        parse_dual_or_blocks requires len(or_parts) >= 3 (splitting on '(or ').
        If not met, it returns NaN-filled input bounds and a zeros output tensor.
        """
        lines = [
            "(assert (<= X_0 1.0))",
            "(assert (>= X_0 0.0))",
        ]
        result = process_type4(lines, n_inputs=2, n_outputs=3, backend=backend)

        assert len(result) == 1
        assert len(result[0]) == 1, "fallback yields one OR property"
        input_bounds, output_constrs = result[0][0]

        assert tuple(input_bounds.shape) == (2, 2), (
            f"fallback input_bounds shape must be (2, 2), got {tuple(input_bounds.shape)}"
        )
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds

        assert all(math.isnan(float(ib[i, j])) for i in range(2) for j in range(2)), (
            "fallback input_bounds must be all-NaN"
        )

        assert len(output_constrs) == 1, "fallback output_constrs must have one tensor"
        assert tuple(output_constrs[0].shape) == (1, 4), (
            f"fallback output constraint shape must be (1, 4) for n_outputs=3, "
            f"got {tuple(output_constrs[0].shape)}"
        )

    def test_single_or_block_also_falls_back(self, backend):
        """Only one OR block (instead of two) also triggers the fallback."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result) == 1
        assert len(result[0]) == 1
        input_bounds, _output_constrs = result[0][0]

        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert math.isnan(float(ib[0, 0])), "single OR block fallback: input_bounds should be NaN"


class TestType4VerboseMode:
    """Test verbose mode output for process_type4."""

    def test_verbose_true_prints_expected_substrings(self, backend, capsys):
        """verbose=True should print timing and count information to stdout."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0))))",
            "(assert (or (and (<= Y_0 0.3))))",
        ]
        process_type4(lines, n_inputs=1, n_outputs=1, backend=backend, verbose=True)

        captured = capsys.readouterr()
        assert "Type4 detection:" in captured.out, "must print 'Type4 detection:'"
        assert "Input OR groups:" in captured.out, "must print 'Input OR groups:'"
        assert "Output OR options:" in captured.out, "must print 'Output OR options:'"
        assert "Parsing:" in captured.out, "must print 'Parsing:'"
        assert "Type4 total time:" in captured.out, "must print 'Type4 total time:'"

    def test_verbose_false_produces_no_output(self, backend, capsys):
        """verbose=False must not write anything to stdout."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0))))",
            "(assert (or (and (<= Y_0 0.3))))",
        ]
        process_type4(lines, n_inputs=1, n_outputs=1, backend=backend, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == "", f"verbose=False must produce no stdout, got: {captured.out!r}"


class TestType4MultipleInputOutputGroups:
    """Test TYPE4 with multiple OR branches for inputs and outputs."""

    def test_multiple_input_or_branches(self, backend):
        """Multiple AND blocks in the input OR produce multiple tuples in result[0]."""
        lines = [
            "(assert (or "
            "(and (<= X_0 0.25) (>= X_0 0.0)) "
            "(and (<= X_0 0.5) (>= X_0 0.26)) "
            "(and (<= X_0 0.75) (>= X_0 0.51))))",
            "(assert (or (and (<= Y_0 0.5))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result) == 1
        # Three input OR branches => three tuples in result[0]
        assert len(result[0]) == 3, f"three input OR branches => 3 tuples, got {len(result[0])}"

    def test_multiple_inputs_and_outputs(self, backend):
        """Multiple input/output variables with multiple OR branches."""
        lines = [
            "(assert (or "
            "(and (<= X_0 0.5) (>= X_0 0.0) (<= X_1 1.0) (>= X_1 0.0)) "
            "(and (<= X_0 2.0) (>= X_0 1.5) (<= X_1 2.0) (>= X_1 1.5))))",
            "(assert (or (and (<= Y_0 0.3) (<= Y_1 0.3)) (and (>= Y_0 0.7) (>= Y_1 0.7))))",
        ]
        result = process_type4(lines, n_inputs=2, n_outputs=2, backend=backend)

        assert len(result) == 1
        assert len(result[0]) >= 1
        input_bounds, output_constrs = result[0][0]
        assert tuple(input_bounds.shape) == (2, 2), (
            f"input_bounds shape must be (2, 2), got {tuple(input_bounds.shape)}"
        )
        for constr in output_constrs:
            assert constr.shape[1] == 3, (
                f"output constraint width must be 3 (n_outputs+1=3), got {constr.shape[1]}"
            )


class TestType4EdgeCases:
    """Test edge cases including scientific notation, negative, zero, and large bounds."""

    def test_scientific_notation_input_bounds(self, backend):
        """Scientific notation in bounds is parsed correctly."""
        lines = [
            "(assert (or (and (<= X_0 1.5e-10) (>= X_0 2.3e-11))))",
            "(assert (or (and (<= Y_0 1.0e-3))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result) == 1
        assert len(result[0]) >= 1
        input_bounds, _output_constrs = result[0][0]
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert abs(float(ib[0, 1]) - 1.5e-10) < 1e-20, (
            f"X_0 upper bound should be 1.5e-10, got {float(ib[0, 1])}"
        )

    def test_negative_bounds(self, backend):
        """Negative bounds are stored correctly in input_bounds."""
        lines = [
            "(assert (or (and (<= X_0 -0.5) (>= X_0 -2.0))))",
            "(assert (or (and (>= Y_0 -1.0))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result) == 1
        assert len(result[0]) >= 1
        input_bounds, _output_constrs = result[0][0]
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert float(ib[0, 0]) < 0.0, "X_0 lower bound should be negative"
        assert float(ib[0, 1]) < 0.0, "X_0 upper bound should be negative"

    def test_zero_bounds(self, backend):
        """Zero bounds are stored as exact zeros."""
        lines = [
            "(assert (or (and (<= X_0 0.0) (>= X_0 0.0))))",
            "(assert (or (and (<= Y_0 0.0))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result) == 1
        input_bounds, _output_constrs = result[0][0]
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert float(ib[0, 0]) == pytest.approx(0.0), "X_0 lower bound should be 0.0"
        assert float(ib[0, 1]) == pytest.approx(0.0), "X_0 upper bound should be 0.0"

    def test_large_bounds(self, backend):
        """Very large bounds are handled without overflow."""
        lines = [
            "(assert (or (and (<= X_0 1e10) (>= X_0 0.0))))",
            "(assert (or (and (<= Y_0 1e10))))",
        ]
        result = process_type4(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result) == 1
        input_bounds, _output_constrs = result[0][0]
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert float(ib[0, 1]) == pytest.approx(1e10, rel=1e-9), "X_0 upper bound should be 1e10"
