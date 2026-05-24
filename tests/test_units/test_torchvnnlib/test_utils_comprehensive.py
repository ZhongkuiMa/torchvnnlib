"""Comprehensive unit tests for fast_type utils - Phase 5 coverage.

This module provides extensive test coverage for fast_type._utils functions:
- parse_input_or_block() with various OR input patterns
- parse_output_or_block() with various OR output patterns
- Constraint parsing for all operator types (<=, >=, =)
- Parsing with multiple variables
- Parsing with mixed constraint types
- Edge cases: empty OR blocks, single constraint, deeply nested
- Error handling for malformed expressions
- convert_simple_input_bounds() tensor construction

Target coverage: All 40 missed statements in _utils.py (79% -> >92%)
"""

__docformat__ = "restructuredtext"

import pytest

from torchvnnlib.fast_type._utils import (
    INPUT_BOUND_INNER_PATTERN,
    OUTPUT_BOUND_INNER_PATTERN,
    OUTPUT_CONSTRAINT_INNER_PATTERN,
    SIMPLE_INPUT_BOUND_PATTERN,
    SIMPLE_OUTPUT_BOUND_PATTERN,
    SIMPLE_OUTPUT_CONSTRAINT_PATTERN,
    convert_simple_input_bounds,
    parse_and_block,
    parse_dual_or_blocks,
    parse_input_bounds_block,
    parse_input_or_block,
    parse_output_and_block,
    parse_output_or_block,
)


class TestInputBoundPattern:
    """Test INPUT_BOUND_INNER_PATTERN regex."""

    @pytest.mark.parametrize(
        ("text", "expected_op"),
        [
            ("(<=  X_0  0.5)", "<="),
            ("(>=  X_5  1.0)", ">="),
            ("(=  X_10  2.5)", "="),
        ],
    )
    def test_operator_patterns(self, text, expected_op):
        """Test operator pattern matching."""
        matches = INPUT_BOUND_INNER_PATTERN.findall(text)
        assert len(matches) == 1
        assert matches[0][0] == expected_op

    # [REVIEW] Deleted: test_leq_pattern, test_geq_pattern, test_eq_pattern (InputBoundPattern).
    # STR1: merged 3 HIGH_DUP into parametrized test_operator_patterns.

    @pytest.mark.parametrize(
        ("text", "expected_count"),
        [
            ("(<=  X_0  -1.5)", 1),
            ("(<=  X_0  1.5e-10)", 1),
            ("(<=  Y_0  0.5)", 0),
        ],
    )
    def test_pattern_edge_cases(self, text, expected_count):
        """Test pattern matching with edge case inputs (STR5: merged 3 tests)."""
        matches = INPUT_BOUND_INNER_PATTERN.findall(text)
        assert len(matches) == expected_count


class TestOutputBoundPattern:
    """Test OUTPUT_BOUND_INNER_PATTERN regex."""

    @pytest.mark.parametrize(
        ("text", "expected_op"),
        [
            ("(<=  Y_0  0.5)", "<="),
            ("(>=  Y_5  1.0)", ">="),
            ("(=  Y_10  2.5)", "="),
        ],
    )
    def test_output_bound_operator_patterns(self, text, expected_op):
        """Test output bound operator pattern matching."""
        matches = OUTPUT_BOUND_INNER_PATTERN.findall(text)
        assert len(matches) == 1
        assert matches[0][0] == expected_op

    # [REVIEW] Deleted: test_leq_pattern, test_geq_pattern, test_eq_pattern (OutputBoundPattern).
    # STR1: merged 3 HIGH_DUP into parametrized test_output_bound_operator_patterns.


class TestOutputConstraintPattern:
    """Test OUTPUT_CONSTRAINT_INNER_PATTERN regex."""

    @pytest.mark.parametrize(
        ("text", "expected_op", "expected_i", "expected_j"),
        [
            ("(<=  Y_0  Y_1)", "<=", "0", "1"),
            ("(>=  Y_5  Y_3)", ">=", "5", "3"),
        ],
    )
    def test_output_constraint_operators(self, text, expected_op, expected_i, expected_j):
        """Test output constraint operator pattern matching."""
        matches = OUTPUT_CONSTRAINT_INNER_PATTERN.findall(text)
        assert len(matches) == 1
        assert matches[0][0] == expected_op
        if expected_i is not None:
            assert matches[0][2] == expected_i
        if expected_j is not None:
            assert matches[0][4] == expected_j

    # [REVIEW] Deleted: test_leq_constraint, test_geq_constraint (OutputConstraintPattern).
    # STR1: merged 2 of 6 group members into parametrized test_output_constraint_operators.


class TestParseInputBoundsBlock:
    """Test parse_input_bounds_block function."""

    def test_single_input_bound_leq(self, backend):
        """Test parsing single <= input bound."""
        block = "(<=  X_0  1.0)"
        result = parse_input_bounds_block(block, n_inputs=1, backend=backend)
        assert result.shape == (1, 2)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 1] == pytest.approx(1.0)

    def test_single_input_bound_geq(self, backend):
        """Test parsing single >= input bound."""
        block = "(>=  X_0  0.5)"
        result = parse_input_bounds_block(block, n_inputs=1, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(0.5)

    def test_equality_bound(self, backend):
        """Test parsing equality bound."""
        block = "(=  X_0  0.75)"
        result = parse_input_bounds_block(block, n_inputs=1, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(0.75)
        assert result_np[0, 1] == pytest.approx(0.75)

    def test_multiple_bounds_same_var(self, backend):
        """Test parsing multiple bounds for same variable."""
        block = "(>=  X_0  0.0) (<=  X_0  1.0)"
        result = parse_input_bounds_block(block, n_inputs=1, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(0.0)
        assert result_np[0, 1] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "block",
        [
            pytest.param(
                "(>=  X_0  0.0) (<=  X_0  1.0) (>=  X_1  -0.5) (<=  X_1  0.5)",
                id="multiple_variables",
            ),
            pytest.param(
                "(>=  X_0  0.123456789) (<=  X_1  1.5e-10)",
                id="float_parsing",
            ),
        ],
    )
    def test_two_variable_bounds_shape(self, block, backend):
        """Test parsing bounds for 2 variables (multiple/float formats; STR11: merged pair)."""
        result = parse_input_bounds_block(block, n_inputs=2, backend=backend)
        assert result.shape == (2, 2)

    # [REVIEW] Deleted: test_multiple_variables, test_float_parsing.
    # STR11: merged into test_two_variable_bounds_shape.

    def test_negative_bounds(self, backend):
        """Test parsing negative bounds."""
        block = "(>=  X_0  -5.0) (<=  X_0  -1.0)"
        result = parse_input_bounds_block(block, n_inputs=1, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(-5.0)
        assert result_np[0, 1] == pytest.approx(-1.0)

    def test_scientific_notation_bounds(self, backend):
        """Test parsing scientific notation in bounds."""
        block = "(>=  X_0  1e-5) (<=  X_0  1e-3)"
        result = parse_input_bounds_block(block, n_inputs=1, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(1e-5)
        assert result_np[0, 1] == pytest.approx(1e-3)

    def test_out_of_range_index_skipped(self, backend):
        """Test that out-of-range indices are skipped."""
        block = "(<=  X_5  1.0) (<=  X_0  2.0)"
        result = parse_input_bounds_block(block, n_inputs=2, backend=backend)
        # Only X_0 should be set
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 1] == pytest.approx(2.0)


class TestParseOutputAndBlock:
    """Test parse_output_and_block function."""

    @pytest.mark.parametrize(
        ("block", "n_outputs", "expected_shape"),
        [
            ("(<=  Y_0  Y_1)", 2, (1, 3)),
            ("(>=  Y_0  Y_1)", 2, (1, 3)),
        ],
    )
    def test_output_constraint_types(self, block, n_outputs, expected_shape, backend):
        """Test parsing output constraint types."""
        result = parse_output_and_block(block, n_outputs=n_outputs, backend=backend)
        assert result.shape == expected_shape

    # [REVIEW] Deleted: test_output_constraint_leq, test_output_constraint_geq.
    # STR2: merged 2 MED_DUP.

    @pytest.mark.parametrize(
        ("block", "n_outputs", "min_rows"),
        [
            ("(<=  Y_0  0.5)", 1, 1),
            ("(>=  Y_0  0.5)", 1, 1),
            ("(=  Y_0  0.5)", 1, 2),
            ("(<=  Y_0  Y_1) (>=  Y_1  Y_2) (<=  Y_0  0.5)", 3, 3),
        ],
    )
    def test_output_bounds_and_constraints(self, block, n_outputs, min_rows, backend):
        """Test parsing output bounds and multiple constraints."""
        result = parse_output_and_block(block, n_outputs=n_outputs, backend=backend)
        assert result.shape[0] >= min_rows

    # [REVIEW] Deleted: test_output_bound_leq, test_output_bound_geq,
    # test_output_bound_eq, test_multiple_constraints.
    # STR2: merged 4 MED_DUP.

    def test_out_of_range_index_skipped(self, backend):
        """Test that out-of-range indices are skipped."""
        block = "(<=  Y_0  0.5)"
        result = parse_output_and_block(block, n_outputs=2, backend=backend)
        # Only Y_0 bound should be processed
        assert result.shape[0] >= 1

    def test_empty_block(self, backend):
        """Test parsing empty block."""
        block = ""
        result = parse_output_and_block(block, n_outputs=2, backend=backend)
        assert result.shape == (1, 3)  # Default zero tensor


class TestParseInputOrBlock:
    """Test parse_input_or_block function."""

    def test_single_and_block(self, backend):
        """Test parsing single AND block in OR."""
        lines = ["(assert (or (and (>=  X_0  0.0) (<=  X_0  1.0))))"]
        result = parse_input_or_block(lines, n_inputs=1, backend=backend)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_multiple_and_blocks(self, backend):
        """Test parsing multiple AND blocks in OR."""
        lines = ["(assert (or (and (<=  X_0  0.5)) (and (>=  X_0  0.75))))"]
        result = parse_input_or_block(lines, n_inputs=1, backend=backend)
        assert len(result) >= 1

    def test_empty_or_block(self, backend):
        """Test parsing empty OR block."""
        lines: list[str] = []
        result = parse_input_or_block(lines, n_inputs=1, backend=backend)
        # Should return NaN bounds
        assert len(result) == 1

    def test_or_block_with_bounds(self, backend):
        """Test OR block with multiple bounds."""
        lines = [
            "(assert (or (and (>=  X_0  0.0) (<=  X_0  1.0)) (and (>=  X_1  -1.0) (<=  X_1  2.0))))"
        ]
        result = parse_input_or_block(lines, n_inputs=2, backend=backend)
        assert len(result) >= 1


class TestParseOutputOrBlock:
    """Test parse_output_or_block function."""

    def test_single_and_block(self, backend):
        """Test parsing single AND block in OR."""
        lines = ["(assert (or (and (<=  Y_0  0.5))))"]
        result = parse_output_or_block(lines, n_inputs=0, n_outputs=1, backend=backend)
        assert isinstance(result, list)
        assert len(result) >= 1

    @pytest.mark.parametrize(
        ("lines", "n_outputs"),
        [
            pytest.param(
                ["(assert (or (and (<=  Y_0  0.3)) (and (>=  Y_0  0.7))))"],
                1,
                id="multiple_and_blocks",
            ),
            pytest.param(
                ["(assert (or (and (<=  Y_0  Y_1)) (and (>=  Y_0  Y_1))))"],
                2,
                id="with_output_constraints",
            ),
        ],
    )
    def test_multi_branch_or_block(self, lines, n_outputs, backend):
        """Test OR block with multiple AND branches (bounds/constraints; STR11: merged pair)."""
        result = parse_output_or_block(lines, n_inputs=0, n_outputs=n_outputs, backend=backend)
        assert len(result) >= 1

    # [REVIEW] Deleted: test_multiple_and_blocks, test_or_block_with_constraints.
    # STR11: merged into test_multi_branch_or_block.

    def test_empty_or_block(self, backend):
        """Test parsing empty OR block."""
        lines: list[str] = []
        result = parse_output_or_block(lines, n_inputs=0, n_outputs=1, backend=backend)
        # Should return zero bounds
        assert len(result) == 1


class TestParseAndBlock:
    """Test parse_and_block function."""

    def test_simple_and_block(self, backend):
        """Test parsing simple AND block."""
        block = "(>=  X_0  0.0) (<=  X_0  1.0) (<=  Y_0  0.5)"
        input_bounds, _output_constrs = parse_and_block(
            block, n_inputs=1, n_outputs=1, backend=backend
        )
        assert input_bounds.shape == (1, 2)
        assert isinstance(_output_constrs, list)

    def test_and_block_with_constraints(self, backend):
        """Test AND block with output constraints."""
        block = "(>=  X_0  0.0) (<=  X_0  1.0) (<=  Y_0  Y_1) (>=  Y_1  0.2)"
        input_bounds, _output_constrs = parse_and_block(
            block, n_inputs=1, n_outputs=2, backend=backend
        )
        assert input_bounds.shape == (1, 2)
        assert len(_output_constrs) >= 1

    def test_and_block_multiple_inputs(self, backend):
        """Test AND block with multiple inputs."""
        block = "(>=  X_0  -1.0) (<=  X_0  1.0) (>=  X_1  0.0) (<=  X_1  2.0) (<=  Y_0  0.5)"
        input_bounds, _ = parse_and_block(block, n_inputs=2, n_outputs=1, backend=backend)
        assert input_bounds.shape == (2, 2)


class TestParseDualOrBlocks:
    """Test parse_dual_or_blocks function."""

    def test_dual_or_blocks(self, backend):
        """Test parsing dual OR blocks."""
        lines = [
            "(assert (or (and (<=  X_0  0.5)) (and (>=  X_0  0.75))))"
            "(assert (or (and (<=  Y_0  0.3)) (and (>=  Y_0  0.7))))"
        ]
        input_bounds_list, output_constrs_list = parse_dual_or_blocks(
            lines, n_inputs=1, n_outputs=1, backend=backend
        )
        assert isinstance(input_bounds_list, list)
        assert isinstance(output_constrs_list, list)

    def test_insufficient_or_blocks(self, backend):
        """Test with insufficient OR blocks."""
        lines = ["(assert (or (and (<=  X_0  0.5))))"]
        input_bounds_list, output_constrs_list = parse_dual_or_blocks(
            lines, n_inputs=1, n_outputs=1, backend=backend
        )
        # Should return NaN for missing blocks
        assert len(input_bounds_list) >= 1
        assert len(output_constrs_list) >= 1


class TestConvertSimpleInputBounds:
    """Test convert_simple_input_bounds function."""

    def test_single_leq_bound(self, backend):
        """Test conversion of single <= bound with both bounds."""
        bounds = [(">=", "X_", 0, 0.0), ("<=", "X_", 0, 1.0)]
        result = convert_simple_input_bounds(bounds, n_inputs=1, backend=backend)
        assert result.shape == (1, 2)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(0.0)
        assert result_np[0, 1] == pytest.approx(1.0)

    def test_single_geq_bound(self, backend):
        """Test conversion of single >= bound with both bounds."""
        bounds = [(">=", "X_", 0, 0.5), ("<=", "X_", 0, 1.0)]
        result = convert_simple_input_bounds(bounds, n_inputs=1, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(0.5)

    def test_equality_bound(self, backend):
        """Test conversion of equality bound."""
        bounds = [("=", "X_", 0, 0.75)]
        result = convert_simple_input_bounds(bounds, n_inputs=1, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(0.75)
        assert result_np[0, 1] == pytest.approx(0.75)

    def test_multiple_bounds(self, backend):
        """Test conversion of multiple bounds."""
        bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
            (">=", "X_", 1, -0.5),
            ("<=", "X_", 1, 0.5),
        ]
        result = convert_simple_input_bounds(bounds, n_inputs=2, backend=backend)
        assert result.shape == (2, 2)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(0.0)
        assert result_np[0, 1] == pytest.approx(1.0)

    def test_non_input_variables_skipped(self, backend):
        """Test that Y_ variables are skipped."""
        bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
            (">=", "Y_", 0, 0.5),
        ]
        result = convert_simple_input_bounds(bounds, n_inputs=1, backend=backend)
        # Y_ should be skipped
        assert result.shape == (1, 2)

    def test_missing_bounds_raises(self, backend):
        """Test that missing bounds raises error."""
        bounds = [(">=", "X_", 0, 0.0)]  # Only lower bound
        with pytest.raises(ValueError, match="Missing input bounds"):
            convert_simple_input_bounds(bounds, n_inputs=1, backend=backend)

    def test_negative_bounds(self, backend):
        """Test conversion with negative bounds."""
        bounds = [
            (">=", "X_", 0, -5.0),
            ("<=", "X_", 0, -1.0),
        ]
        result = convert_simple_input_bounds(bounds, n_inputs=1, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(-5.0)
        assert result_np[0, 1] == pytest.approx(-1.0)


class TestSimplePatterns:
    """Test simple pattern regex matching."""

    @pytest.mark.parametrize(
        ("text", "pattern"),
        [
            ("(assert (<=  X_0  1.0))", "SIMPLE_INPUT_BOUND_PATTERN"),
            ("(assert (<=  Y_0  0.5))", "SIMPLE_OUTPUT_BOUND_PATTERN"),
            ("(assert (>=  Y_0  Y_1))", "SIMPLE_OUTPUT_CONSTRAINT_PATTERN"),
        ],
    )
    def test_simple_pattern_matching(self, text, pattern):
        """Test simple pattern regex matching (STR5: merged 3 tests)."""
        patterns = {
            "SIMPLE_INPUT_BOUND_PATTERN": SIMPLE_INPUT_BOUND_PATTERN,
            "SIMPLE_OUTPUT_BOUND_PATTERN": SIMPLE_OUTPUT_BOUND_PATTERN,
            "SIMPLE_OUTPUT_CONSTRAINT_PATTERN": SIMPLE_OUTPUT_CONSTRAINT_PATTERN,
        }
        match = patterns[pattern].match(text)
        assert match
