"""Tests for the TYPE5 VNN-LIB processor (_type5_processor.py).

TYPE5 structure: top-level OR wrapping complete AND(input+output) properties.
Structure: (or (and ...input_constraints... ...output_constraints...) ...)

Covers: property count per OR clause, input/output shapes, numeric values,
verbose output substrings, fallback path (no AND blocks), edge cases.
"""

__docformat__ = "restructuredtext"

import math

import pytest

from torchvnnlib.fast_type._type5_processor import process_type5


class TestType5BasicProcessing:
    """Test basic TYPE5 processing and return-value structure."""

    def test_returns_single_and_group(self, backend):
        """Result always has exactly one AND group (outer list length 1)."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0) (<= Y_0 0.3))"
            "            (and (>= X_0 0.7) (<= Y_0 0.9))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert isinstance(result, list), "result must be a list"
        assert len(result) == 1, "exactly one AND group expected"
        assert isinstance(result[0], list), "AND group must be a list of tuples"

    def test_each_element_is_two_tuple(self, backend):
        """Each element of the properties list is a (input_bounds, output_constrs) tuple."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0) (<= Y_0 0.3))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        for item in result[0]:
            assert isinstance(item, tuple), "each property must be a tuple"
            assert len(item) == 2, "tuple must have exactly (input_bounds, output_constrs)"
            _input_bounds, output_constrs = item
            assert isinstance(output_constrs, list), "output_constrs must be a list"

    def test_input_bounds_shape_matches_n_inputs(self, backend):
        """input_bounds has shape (n_inputs, 2)."""
        lines = [
            "(assert (or (and (<= X_0 1.0) (>= X_0 0.0) (<= X_1 2.0) (>= X_1 -1.0) (<= Y_0 0.5))))",
        ]
        result = process_type5(lines, n_inputs=2, n_outputs=1, backend=backend)

        assert len(result[0]) >= 1
        input_bounds, _output_constrs = result[0][0]
        assert tuple(input_bounds.shape) == (2, 2), (
            f"input_bounds shape must be (2, 2), got {tuple(input_bounds.shape)}"
        )

    def test_output_constr_width(self, backend):
        """Each output constraint tensor has width n_outputs + 1."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0) (<= Y_0 0.3) (<= Y_1 0.7))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=2, backend=backend)

        assert len(result[0]) >= 1
        _input_bounds, output_constrs = result[0][0]
        for constr in output_constrs:
            assert constr.shape[1] == 3, (
                f"each output constraint tensor must have width 3 (n_outputs+1), "
                f"got {constr.shape[1]}"
            )


class TestType5PropertyCount:
    """Test that the number of OR clauses equals the number of returned properties."""

    def test_one_and_clause_gives_one_property(self, backend):
        """A single AND clause => exactly one property in result[0]."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0) (<= Y_0 0.3))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) == 1, f"one AND clause => 1 property, got {len(result[0])}"

    def test_two_and_clauses_give_two_properties(self, backend):
        """Two AND clauses => exactly two properties in result[0]."""
        lines = [
            "(assert (or "
            "(and (<= X_0 0.3) (>= X_0 0.0) (<= Y_0 0.5)) "
            "(and (>= X_0 0.7) (<= X_0 1.0) (>= Y_0 0.7))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) == 2, f"two AND clauses => 2 properties, got {len(result[0])}"

    def test_three_and_clauses_give_three_properties(self, backend):
        """Three AND clauses => three properties in result[0]."""
        lines = [
            "(assert (or "
            "(and (<= X_0 0.2) (>= X_0 0.0) (<= Y_0 0.3)) "
            "(and (<= X_0 0.5) (>= X_0 0.3) (<= Y_0 0.6)) "
            "(and (>= X_0 0.8) (<= X_0 1.0) (>= Y_0 0.8))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) == 3, f"three AND clauses => 3 properties, got {len(result[0])}"


class TestType5NumericValues:
    """Test that parsed numeric values are stored correctly."""

    def test_input_bounds_numeric_values(self, backend):
        """Exact input bounds are extracted from an AND clause."""
        lines = [
            "(assert (or (and (<= X_0 0.3) (>= X_0 0.1) (<= Y_0 0.5))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) >= 1
        input_bounds, _output_constrs = result[0][0]
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert abs(float(ib[0, 0]) - 0.1) < 1e-9, (
            f"X_0 lower bound should be 0.1, got {float(ib[0, 0])}"
        )
        assert abs(float(ib[0, 1]) - 0.3) < 1e-9, (
            f"X_0 upper bound should be 0.3, got {float(ib[0, 1])}"
        )

    def test_output_constraint_leq_encoding(self, backend):
        """(<= Y_0 0.5) => constraint row [0.5, -1.0] (first elem is bound, Y_0 col is -1)."""
        lines = [
            "(assert (or (and (<= X_0 1.0) (>= X_0 0.0) (<= Y_0 0.5))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) >= 1
        _input_bounds, output_constrs = result[0][0]
        assert len(output_constrs) >= 1

        # Find the constraint tensor that has a non-zero entry in col 1 (Y_0 col)
        found_leq = False
        for constr in output_constrs:
            oc = backend.to_numpy(constr) if hasattr(backend, "to_numpy") else constr
            for row_idx in range(oc.shape[0]):
                # <= Y_0 0.5 encodes as: [0.5, -1.0]
                if abs(float(oc[row_idx, 1]) - (-1.0)) < 1e-9:
                    assert abs(float(oc[row_idx, 0]) - 0.5) < 1e-9, (
                        f"bound column for (<= Y_0 0.5) should be 0.5, got {float(oc[row_idx, 0])}"
                    )
                    found_leq = True
        assert found_leq, "did not find the expected (<= Y_0 0.5) constraint row"

    def test_output_constraint_geq_encoding(self, backend):
        """(>= Y_0 -1.0) => constraint row [-(-1.0), 1.0] = [1.0, 1.0]."""
        lines = [
            "(assert (or (and (<= X_0 1.0) (>= X_0 0.0) (>= Y_0 -1.0))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) >= 1
        _input_bounds, output_constrs = result[0][0]
        assert len(output_constrs) >= 1

        found_geq = False
        for constr in output_constrs:
            oc = backend.to_numpy(constr) if hasattr(backend, "to_numpy") else constr
            for row_idx in range(oc.shape[0]):
                # >= Y_0 -1.0 encodes as: [-(-1.0), 1.0] = [1.0, 1.0]
                if abs(float(oc[row_idx, 1]) - 1.0) < 1e-9:
                    assert abs(float(oc[row_idx, 0]) - 1.0) < 1e-9, (
                        f"bound col for (>= Y_0 -1.0) should be 1.0, got {float(oc[row_idx, 0])}"
                    )
                    found_geq = True
        assert found_geq, "did not find the expected (>= Y_0 -1.0) constraint row"


class TestType5FallbackPath:
    """Test the fallback when no (and ...) blocks are found in the lines."""

    def test_no_and_blocks_returns_nan_input_zeros_output(self, backend):
        """Lines with no '(and ' substring trigger the fallback in _parse_top_level_or.

        The fallback returns one property with NaN input_bounds and zeros output.
        """
        # No '(and ...' substring, so parts[1:] will be empty
        lines = [
            "(assert (<= X_0 1.0))",
            "(assert (>= X_0 0.0))",
        ]
        result = process_type5(lines, n_inputs=2, n_outputs=3, backend=backend)

        assert len(result) == 1, "always one AND group"
        assert len(result[0]) == 1, "fallback yields exactly one property"
        input_bounds, output_constrs = result[0][0]

        assert tuple(input_bounds.shape) == (2, 2), (
            f"fallback input_bounds shape must be (2, 2), got {tuple(input_bounds.shape)}"
        )
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert all(math.isnan(float(ib[i, j])) for i in range(2) for j in range(2)), (
            "fallback input_bounds must be all-NaN"
        )

        assert len(output_constrs) == 1, "fallback must have one output constraint tensor"
        assert tuple(output_constrs[0].shape) == (1, 4), (
            f"fallback output constraint shape must be (1, n_outputs+1)=(1, 4), "
            f"got {tuple(output_constrs[0].shape)}"
        )
        oc = (
            backend.to_numpy(output_constrs[0])
            if hasattr(backend, "to_numpy")
            else output_constrs[0]
        )
        assert float(oc[0, 0]) == pytest.approx(0.0), "fallback output tensor must be all zeros"

    def test_empty_lines_returns_fallback(self, backend):
        """Empty input lines also trigger the fallback."""
        result = process_type5([], n_inputs=1, n_outputs=1, backend=backend)

        assert len(result) == 1
        assert len(result[0]) == 1
        input_bounds, _output_constrs = result[0][0]
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert math.isnan(float(ib[0, 0])), "empty lines fallback: input_bounds[0,0] must be NaN"


class TestType5VerboseMode:
    """Test verbose mode output for process_type5."""

    def test_verbose_true_prints_expected_substrings(self, backend, capsys):
        """verbose=True should print timing and count information to stdout."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0) (<= Y_0 0.3))"
            "            (and (>= X_0 0.7) (<= Y_0 0.9))))",
        ]
        process_type5(lines, n_inputs=1, n_outputs=1, backend=backend, verbose=True)

        captured = capsys.readouterr()
        assert "Type5 detection:" in captured.err, "must print 'Type5 detection:'"
        assert "OR clauses (properties):" in captured.err, "must print 'OR clauses (properties):'"
        assert "Parsing:" in captured.err, "must print 'Parsing:'"
        assert "Type5 total time:" in captured.err, "must print 'Type5 total time:'"

    def test_verbose_false_produces_no_output(self, backend, capsys):
        """verbose=False must not write anything to stdout."""
        lines = [
            "(assert (or (and (<= X_0 0.5) (>= X_0 0.0) (<= Y_0 0.3))))",
        ]
        process_type5(lines, n_inputs=1, n_outputs=1, backend=backend, verbose=False)

        captured = capsys.readouterr()
        assert captured.err == "", f"verbose=False must produce no stdout, got: {captured.err!r}"


class TestType5EdgeCases:
    """Test edge cases including scientific notation, negative, zero, and large bounds."""

    def test_scientific_notation_bounds(self, backend):
        """Scientific notation in bounds is handled correctly."""
        lines = [
            "(assert (or (and (<= X_0 1.5e-10) (>= X_0 2.3e-11) (<= Y_0 1.0e-3))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) >= 1
        input_bounds, _output_constrs = result[0][0]
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert abs(float(ib[0, 1]) - 1.5e-10) < 1e-20, (
            f"X_0 upper bound should be 1.5e-10, got {float(ib[0, 1])}"
        )

    def test_negative_input_bounds(self, backend):
        """Negative input bounds are stored correctly."""
        lines = [
            "(assert (or (and (<= X_0 -0.5) (>= X_0 -2.0) (>= Y_0 -1.0))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) >= 1
        input_bounds, _output_constrs = result[0][0]
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert float(ib[0, 0]) < 0.0, "X_0 lower bound should be negative"
        assert float(ib[0, 1]) < 0.0, "X_0 upper bound should be negative"

    def test_zero_bounds(self, backend):
        """Zero bounds are stored as exact zeros."""
        lines = [
            "(assert (or (and (<= X_0 0.0) (>= X_0 0.0) (= Y_0 0.0))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) >= 1
        input_bounds, _output_constrs = result[0][0]
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert float(ib[0, 0]) == pytest.approx(0.0), "X_0 lower bound should be 0.0"
        assert float(ib[0, 1]) == pytest.approx(0.0), "X_0 upper bound should be 0.0"

    def test_large_bounds(self, backend):
        """Very large bounds are handled without overflow."""
        lines = [
            "(assert (or (and (<= X_0 1e10) (>= X_0 0.0) (<= Y_0 1e10))))",
        ]
        result = process_type5(lines, n_inputs=1, n_outputs=1, backend=backend)

        assert len(result[0]) >= 1
        input_bounds, _output_constrs = result[0][0]
        ib = backend.to_numpy(input_bounds) if hasattr(backend, "to_numpy") else input_bounds
        assert float(ib[0, 1]) == pytest.approx(1e10, rel=1e-9), "X_0 upper bound should be 1e10"

    def test_multiple_inputs_and_outputs(self, backend):
        """Multiple inputs and outputs with two AND clauses."""
        lines = [
            "(assert (or "
            "(and (<= X_0 0.5) (>= X_0 0.0) (<= X_1 1.0) (>= X_1 0.0) (<= Y_0 0.3) (<= Y_1 0.4)) "
            "(and (>= X_0 0.7) (<= X_0 1.0) (>= X_1 1.5) (<= X_1 2.0) (>= Y_0 0.7) (>= Y_1 0.8))))",
        ]
        result = process_type5(lines, n_inputs=2, n_outputs=2, backend=backend)

        assert len(result) == 1
        assert len(result[0]) == 2, f"two AND clauses => 2 properties, got {len(result[0])}"
        for input_bounds, output_constrs in result[0]:
            assert tuple(input_bounds.shape) == (2, 2), (
                f"input_bounds shape must be (2, 2), got {tuple(input_bounds.shape)}"
            )
            for constr in output_constrs:
                assert constr.shape[1] == 3, (
                    f"output constraint width must be 3, got {constr.shape[1]}"
                )
