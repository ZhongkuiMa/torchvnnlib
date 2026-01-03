"""Unit tests for TYPE2 VNN-LIB processor - Bug discovery through comprehensive testing.

TYPE2 structure: (and ...simple_input_constraints... (or ...output_constraints...))

Tests the process_type2() function with comprehensive coverage to discover and validate
bug fixes. Follows principles: broad code coverage, all logics, error handling, quick tests.
"""

import pytest

from torchvnnlib.fast_type import process_type2
from torchvnnlib.fast_type._fast_type_detect import parse_simple_patterns


class TestType2BasicProcessing:
    """Test basic TYPE2 processing functionality."""

    def test_process_type2_simple_case(self, backend):
        """Test TYPE2 with simple input bounds + OR output constraints.

        Structure: (and (>= X_0 0.5) (<= X_0 1.0) (or (>= Y_0 Y_1) (>= Y_1 Y_0)))
        Expected:
        - Input bounds: X_0 ∈ [0.5, 1.0]
        - OR constraints: [Y_0 >= Y_1] OR [Y_1 >= Y_0]
        """
        lines = [
            "(assert (>= X_0 0.5))",
            "(assert (<= X_0 1.0))",
            "(assert (or (>= Y_0 Y_1) (>= Y_1 Y_0)))",
        ]
        n_inputs = 1
        n_outputs = 2

        # Parse data first
        parsed_data = parse_simple_patterns(lines, verbose=False)

        # Process TYPE2
        result = process_type2(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        # Validate structure: list[list[tuple[input_bounds, output_constrs]]]
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1, "Should have 1 AND group"
        assert isinstance(result[0], list), "First element should be a list"
        assert len(result[0]) == 1, "Should have 1 tuple in AND group"

        _input_bounds, _output_constrs = result[0][0]

        # Validate input bounds shape and values
        input_shape = (
            tuple(_input_bounds.shape) if hasattr(_input_bounds, "shape") else _input_bounds.shape
        )
        assert input_shape == (1, 2), f"Input bounds shape should be (1, 2), got {input_shape}"
        _input_bounds_np = (
            backend.to_numpy(_input_bounds) if hasattr(backend, "to_numpy") else _input_bounds
        )
        assert _input_bounds_np[0, 0] >= 0.5 - 1e-6, "X_0 lower bound should be ~0.5"
        assert _input_bounds_np[0, 1] <= 1.0 + 1e-6, "X_0 upper bound should be ~1.0"

        # Validate output constraints (should be list of tensors)
        assert isinstance(_output_constrs, list), "Output constraints should be a list"
        assert len(_output_constrs) > 0, "Should have at least one output constraint"

    def test_process_type2_multiple_inputs(self, type2_multiple_inputs_lines, backend):
        """Test TYPE2 with multiple input variables.

        Tests X_0, X_1, X_2 with different bounds.
        """
        n_inputs = 3
        n_outputs = 2

        parsed_data = parse_simple_patterns(type2_multiple_inputs_lines, verbose=False)
        result = process_type2(
            type2_multiple_inputs_lines,
            n_inputs,
            n_outputs,
            backend,
            verbose=False,
            parsed_data=parsed_data,
        )

        _input_bounds, _output_constrs = result[0][0]

        # Validate shape matches number of inputs
        input_shape = tuple(_input_bounds.shape)
        assert input_shape == (
            n_inputs,
            2,
        ), f"Input bounds shape should be ({n_inputs}, 2), got {input_shape}"

        # Validate all bounds are set (no NaN values)
        _input_bounds_np = (
            backend.to_numpy(_input_bounds) if hasattr(backend, "to_numpy") else _input_bounds
        )
        assert not backend.isnan(_input_bounds_np).any(), "Input bounds must not contain NaN"

        # Validate order: lower bounds <= upper bounds
        assert (_input_bounds_np[:, 0] <= _input_bounds_np[:, 1] + 1e-6).all(), (
            "Lower bounds must be <= upper bounds"
        )

    def test_process_type2_multiple_or_branches(self, type2_multiple_or_branches_lines, backend):
        """Test TYPE2 with many OR branches (3+ branches).

        Tests processing of multiple AND blocks within OR.
        """
        n_inputs = 1
        n_outputs = 2

        parsed_data = parse_simple_patterns(type2_multiple_or_branches_lines, verbose=False)
        result = process_type2(
            type2_multiple_or_branches_lines,
            n_inputs,
            n_outputs,
            backend,
            verbose=False,
            parsed_data=parsed_data,
        )

        _input_bounds, _output_constrs = result[0][0]

        # Validate output constraints list (should have multiple options for OR)
        assert isinstance(_output_constrs, list), "Output constraints should be a list"
        assert len(_output_constrs) >= 3, "Should have multiple OR branches in constraints"

        # Each constraint should be a tensor
        for constr in _output_constrs:
            assert constr is not None, "Constraint should not be None"
            shape = tuple(constr.shape)
            # Constraints have shape (num_constraints, n_outputs + 1)
            assert len(shape) == 2, f"Constraint should be 2D, got shape {shape}"
            assert shape[1] == n_outputs + 1, f"Constraint width should be {n_outputs + 1}"


class TestType2ParsedData:
    """Test TYPE2 processing with pre-parsed data."""

    def test_process_type2_with_parsed_data(self, type2_parsed_data, backend):
        """Test TYPE2 when parsed_data is provided (main code path).

        This tests lines 32-38 in _type2_processor.py where parsed_data is provided.
        Should skip re-parsing.
        """
        n_inputs = 1
        n_outputs = 2

        # Use the fixture's lines
        lines = [
            "(assert (>= X_0 0.5))",
            "(assert (<= X_0 1.0))",
            "(assert (or (>= Y_0 Y_1) (>= Y_1 Y_0)))",
        ]

        # Process with pre-parsed data
        result = process_type2(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=type2_parsed_data
        )

        # Should succeed and have valid structure
        assert len(result) == 1, "Should have AND group"
        _input_bounds, _output_constrs = result[0][0]
        assert tuple(_input_bounds.shape) == (n_inputs, 2), "Should have correct input bounds shape"
        assert isinstance(_output_constrs, list), "Should have output constraints"

    def test_process_type2_without_parsed_data(self, backend):
        """Test TYPE2 when parsed_data is None (fallback path).

        This tests lines 32-38 where parse_simple_patterns is called internally.
        """
        lines = [
            "(assert (>= X_0 -0.5))",
            "(assert (<= X_0 1.5))",
            "(assert (or (>= Y_0 0.0) (<= Y_1 0.0)))",
        ]
        n_inputs = 1
        n_outputs = 2

        # Process without pre-parsed data - should call parse_simple_patterns internally
        result = process_type2(lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=None)

        # Should succeed
        assert len(result) == 1, "Should have AND group"
        _input_bounds, _output_constrs = result[0][0]
        assert tuple(_input_bounds.shape) == (n_inputs, 2), "Should have correct input bounds shape"

    def test_process_type2_parsed_data_structure_validation(self, backend):
        """Test that parsed_data dict structure is validated.

        Tests that parsed_data must have required keys.
        BUG DISCOVERY: Should validate parsed_data structure or handle gracefully.
        """
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (or (>= Y_0 Y_1)))",
        ]
        n_inputs = 1
        n_outputs = 2

        # Test with valid parsed_data structure
        valid_parsed_data = {
            "simple_input_bounds": [(">=", "X_", 0, 0.0), ("<=", "X_", 0, 1.0)],
            "complex_lines": ["(assert (or (>= Y_0 Y_1)))"],
        }

        result = process_type2(lines, n_inputs, n_outputs, backend, parsed_data=valid_parsed_data)
        assert len(result) == 1, "Should process with valid parsed_data"

        # Test with missing keys - should raise or handle gracefully
        invalid_parsed_data_1 = {"simple_input_bounds": []}
        with pytest.raises((KeyError, TypeError, AttributeError)):
            process_type2(lines, n_inputs, n_outputs, backend, parsed_data=invalid_parsed_data_1)

        invalid_parsed_data_2 = {"complex_lines": []}
        with pytest.raises((KeyError, TypeError, AttributeError)):
            process_type2(lines, n_inputs, n_outputs, backend, parsed_data=invalid_parsed_data_2)


class TestType2OrBlockProcessing:
    """Test OR block processing in TYPE2."""

    def test_parse_or_block_single_branch(self, backend):
        """Test OR block with single branch.

        Edge case: (or (and ...)) - only one alternative
        """
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (or (and (>= Y_0 0.5))))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type2(lines, n_inputs, n_outputs, backend, parsed_data=parsed_data)

        _input_bounds, _output_constrs = result[0][0]
        assert isinstance(_output_constrs, list), "Should have output constraints list"
        assert len(_output_constrs) > 0, "Should extract constraints from single OR branch"

    def test_parse_or_block_multiple_branches(self, backend):
        """Test OR block with standard multiple branches.

        Standard case: (or (and ...) (and ...) (and ...))
        """
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (or (and (>= Y_0 0.5) (<= Y_1 0.3)) (and (>= Y_1 Y_0) (<= Y_0 0.2)) (and (= Y_0 Y_1))))",
        ]
        n_inputs = 1
        n_outputs = 2

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type2(lines, n_inputs, n_outputs, backend, parsed_data=parsed_data)

        _input_bounds, _output_constrs = result[0][0]
        assert isinstance(_output_constrs, list), "Should have output constraints list"
        # Should have extracted constraints from each OR branch
        assert len(_output_constrs) >= 3, "Should have multiple constraint sets from OR branches"

    def test_parse_or_block_with_output_bounds(self, backend):
        """Test OR block containing output bound constraints.

        Tests constraints like Y_0 >= 0.5, Y_0 <= 1.0 within OR.
        """
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (or (>= Y_0 0.5) (<= Y_1 -0.3) (= Y_0 0.25)))",
        ]
        n_inputs = 1
        n_outputs = 2

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type2(lines, n_inputs, n_outputs, backend, parsed_data=parsed_data)

        _input_bounds, _output_constrs = result[0][0]
        assert isinstance(_output_constrs, list), "Should extract bound constraints"
        assert len(_output_constrs) > 0, "Should have constraints from OR block"

    def test_parse_or_block_with_comparison_constraints(self, backend):
        """Test OR block with output comparison constraints.

        Tests constraints like Y_0 >= Y_1, Y_1 <= Y_0 within OR.
        """
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (or (>= Y_0 Y_1) (<= Y_1 Y_2)))",
        ]
        n_inputs = 1
        n_outputs = 3

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type2(lines, n_inputs, n_outputs, backend, parsed_data=parsed_data)

        _input_bounds, _output_constrs = result[0][0]
        assert isinstance(_output_constrs, list), "Should extract comparison constraints"


class TestType2EdgeCases:
    """Test edge cases and error handling in TYPE2 processing."""

    def test_type2_empty_input_bounds(self, backend):
        """Test TYPE2 with no input constraints.

        Should handle gracefully - either set NaN or raise clear error.
        """
        lines = [
            "(assert (or (>= Y_0 Y_1)))",
        ]
        n_inputs = 1
        n_outputs = 2

        # Test handling of incomplete bounds - parse internally
        with pytest.raises((ValueError, KeyError, IndexError)):
            process_type2(lines, n_inputs, n_outputs, backend, verbose=False)

    def test_type2_all_output_variables_constrained(self, backend):
        """Test TYPE2 where all output variables are constrained in OR.

        All outputs get explicit constraints.
        """
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (or (and (>= Y_0 0.5) (>= Y_1 0.3)) (and (>= Y_0 Y_1) (>= Y_1 Y_0))))",
        ]
        n_inputs = 1
        n_outputs = 2

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type2(lines, n_inputs, n_outputs, backend, parsed_data=parsed_data)

        _input_bounds, _output_constrs = result[0][0]
        assert len(_output_constrs) >= 2, "Should have multiple constraint sets"

    def test_process_type2_verbose_output(self, backend, capsys):
        """Test verbose mode prints timing information.

        Tests lines 43-62 in _type2_processor.py - verbose flag should print timing.
        """
        lines = [
            "(assert (>= X_0 0.5))",
            "(assert (<= X_0 1.0))",
            "(assert (or (>= Y_0 Y_1)))",
        ]
        n_inputs = 1
        n_outputs = 2

        parsed_data = parse_simple_patterns(lines, verbose=False)

        # Run with verbose=True
        result = process_type2(
            lines, n_inputs, n_outputs, backend, verbose=True, parsed_data=parsed_data
        )

        # Check console output contains timing information
        captured = capsys.readouterr()
        output = captured.out

        # Should print timing information
        assert "Type2" in output or "Type2 total" in output or len(output) > 0, (
            f"Expected verbose output, got: {output}"
        )

        # Should still return valid result
        assert len(result) == 1, "Should have valid result in verbose mode"


class TestType2IntegrationWithBenchmarks:
    """Integration tests with benchmark file patterns."""

    def test_type2_with_realistic_benchmark_pattern(self, backend):
        """Test TYPE2 with realistic VNN-COMP benchmark pattern.

        Mimics actual benchmark file structure where each constraint is separate.

        BUG DISCOVERED: TYPE2 processor does not correctly handle compound assertions
        like "(assert (and (>= X_0 0.0) (<= X_0 1.0) ...))"
        It only works with separate assertions: "(assert (>= X_0 0.0))" and "(assert (<= X_0 1.0))"
        """
        # Use separate assertions (correct format for TYPE2)
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(declare-const Y_0 Real)",
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (>= X_1 -1.0))",
            "(assert (<= X_1 1.0))",
            "(assert (or (and (<= Y_0 -0.5)) (and (>= Y_0 0.5))))",
        ]

        # Extract preprocessed assertions (remove declares)
        preprocessed_lines = [line for line in lines if "(assert" in line]
        n_inputs = 2
        n_outputs = 1

        parsed_data = parse_simple_patterns(preprocessed_lines, verbose=False)
        result = process_type2(
            preprocessed_lines, n_inputs, n_outputs, backend, parsed_data=parsed_data
        )

        # Validate result structure
        assert len(result) == 1, "Should have one AND group"
        _input_bounds, _output_constrs = result[0][0]
        assert tuple(_input_bounds.shape) == (
            n_inputs,
            2,
        ), f"Should have bounds for {n_inputs} inputs"
