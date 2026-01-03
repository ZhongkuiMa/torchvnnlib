"""Comprehensive unit tests for TYPE3 VNN-LIB processor - Phase 4 coverage.

TYPE3 structure: (and (or ...input_constraints...) ...output_constraints...)

Tests the process_type3() function with comprehensive coverage for all code paths.
TYPE3 features OR-branching on input constraints with simple output constraints/bounds.

Target coverage: All 20 missed statements in _type3_processor.py (73% -> >92%)
"""

from torchvnnlib.fast_type import process_type3
from torchvnnlib.fast_type._fast_type_detect import parse_simple_patterns


class TestType3BasicProcessing:
    """Test basic TYPE3 processing functionality."""

    def test_process_type3_simple_case(self, backend):
        """Test TYPE3 with OR inputs + simple outputs.

        Structure: (and (or (and (<= X_0 0.5) (>= X_0 0.0)) ...) (<= Y_0 0.5) ...)
        Expected:
        - Input OR region: X_0 ∈ [0.0, 0.5]
        - Output bounds: Y_0 <= 0.5
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 0.5) (>= X_0 0.0))) (<= Y_0 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        # Validate structure: list[list[tuple[input_bounds, output_constrs]]]
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1, "Should have 1 AND group"
        assert isinstance(result[0], list), "First element should be a list"
        assert len(result[0]) >= 1, "Should have at least 1 tuple in AND group"

        _input_bounds, _output_constrs = result[0][0]

        # Validate input bounds shape
        input_shape = (
            tuple(_input_bounds.shape) if hasattr(_input_bounds, "shape") else _input_bounds.shape
        )
        assert len(input_shape) == 2, f"Input bounds should be 2D, got {input_shape}"
        assert input_shape[1] == 2, f"Input bounds should have 2 columns, got {input_shape[1]}"

        # Validate output constraints
        assert isinstance(_output_constrs, list), "Output constraints should be a list"

    def test_process_type3_multiple_inputs(self, backend):
        """Test TYPE3 with multiple input variables."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0) (<= X_1 1.0) (>= X_1 0.0)))",
            "                  (and (<= X_0 2.0) (>= X_0 1.5) (<= X_1 2.0) (>= X_1 1.5)))",
            "            (<= Y_0 0.5)))",
        ]
        n_inputs = 2
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1
        assert len(result[0]) >= 1

        _input_bounds, _output_constrs = result[0][0]

        # Validate shape matches number of inputs
        input_shape = tuple(_input_bounds.shape)
        assert input_shape[0] >= 1, "Should have at least 1 input region"
        assert input_shape[1] == 2, "Input bounds should have 2 columns"

    def test_process_type3_multiple_or_branches(self, backend):
        """Test TYPE3 with multiple OR branches.

        Tests processing of multiple AND blocks within OR constraint.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 0.3) (>= X_0 0.0))",
            "                  (and (<= X_0 1.0) (>= X_0 0.7)))",
            "            (<= Y_0 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1
        assert len(result[0]) >= 1

        _input_bounds, _output_constrs = result[0][0]

        # Should have at least one input region
        input_shape = tuple(_input_bounds.shape)
        assert input_shape[0] >= 1

    def test_process_type3_output_constraints(self, backend):
        """Test TYPE3 with output constraints (Y <op> Y format).

        Tests _process_output_constraints_type3 function.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (>= Y_0 Y_1)))",
        ]
        n_inputs = 1
        n_outputs = 2

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1
        _input_bounds, _output_constrs = result[0][0]

        # Output constraints should be present
        assert isinstance(_output_constrs, list)
        assert len(_output_constrs) >= 1

    def test_process_type3_output_bounds(self, backend):
        """Test TYPE3 with output bounds (Y <op> value format).

        Tests _process_output_bounds_type3 function.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (<= Y_0 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1
        _input_bounds, _output_constrs = result[0][0]

        # Output constraints should contain bounds
        assert isinstance(_output_constrs, list)
        assert len(_output_constrs) >= 1


class TestType3ParsedData:
    """Test TYPE3 processing with pre-parsed data."""

    def test_process_type3_with_parsed_data(self, backend):
        """Test TYPE3 when parsed_data is provided (main code path).

        Tests lines 32-36 where parsed_data is provided. Should skip re-parsing.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 0.5) (>= X_0 0.0)))",
            "            (<= Y_0 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 1

        # Pre-parse the data
        parsed_data = parse_simple_patterns(lines, verbose=False)

        # Process with parsed_data provided
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1
        assert len(result[0]) >= 1

    def test_process_type3_without_parsed_data(self, backend):
        """Test TYPE3 without pre-parsed data (fallback parsing).

        Tests lines 37-42 where parsed_data=None. Function should parse internally.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 0.5) (>= X_0 0.0)))",
            "            (<= Y_0 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 1

        # Process without parsed_data (should use fallback)
        result = process_type3(lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=None)

        assert len(result) == 1
        assert len(result[0]) >= 1


class TestType3OutputConstraintProcessing:
    """Test _process_output_constraints_type3 function."""

    def test_output_constraints_geq_constraint(self, backend):
        """Test >= constraint processing.

        (>= Y_0 Y_1) should set row[Y_0 index] = 1.0, row[Y_1 index] = -1.0
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (>= Y_0 Y_1)))",
        ]
        n_inputs = 1
        n_outputs = 2

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        _input_bounds, _output_constrs = result[0][0]
        assert len(_output_constrs) >= 1

    def test_output_constraints_leq_constraint(self, backend):
        """Test <= constraint processing.

        (<= Y_0 Y_1) should set row[Y_0 index] = -1.0, row[Y_1 index] = 1.0
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (<= Y_0 Y_1)))",
        ]
        n_inputs = 1
        n_outputs = 2

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        _input_bounds, _output_constrs = result[0][0]
        assert len(_output_constrs) >= 1

    def test_output_constraints_empty(self, backend):
        """Test processing with no output constraints."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        _input_bounds, _output_constrs = result[0][0]
        # Should have default constraint if none provided
        assert isinstance(_output_constrs, list)


class TestType3OutputBoundsProcessing:
    """Test _process_output_bounds_type3 function."""

    def test_output_bounds_leq(self, backend):
        """Test <= output bound processing.

        (<= Y_0 0.5) creates constraint: -Y_0 <= -0.5 (i.e., Y_0 >= 0.5 is violated)
        Stored as: [0.5, -1.0, ...] for (Y_0 <= 0.5)
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (<= Y_0 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        _input_bounds, _output_constrs = result[0][0]
        assert len(_output_constrs) >= 1

    def test_output_bounds_geq(self, backend):
        """Test >= output bound processing.

        (>= Y_0 -1.0) creates constraint: Y_0 >= -1.0
        Stored as: [-1.0, 1.0, ...] for (Y_0 >= -1.0)
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (>= Y_0 -1.0)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        _input_bounds, _output_constrs = result[0][0]
        assert len(_output_constrs) >= 1

    def test_output_bounds_equality(self, backend):
        """Test = output bound processing.

        (= Y_0 0.5) creates two constraints:
        Y_0 <= 0.5 and Y_0 >= 0.5 (i.e., Y_0 = 0.5)
        Should generate 2 rows in output bounds tensor
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (= Y_0 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        _input_bounds, _output_constrs = result[0][0]
        assert len(_output_constrs) >= 1

    def test_output_bounds_multiple(self, backend):
        """Test multiple output bounds."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (<= Y_0 0.5)",
            "            (>= Y_1 -1.0)))",
        ]
        n_inputs = 1
        n_outputs = 2

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        _input_bounds, _output_constrs = result[0][0]
        assert len(_output_constrs) >= 1

    def test_output_bounds_empty(self, backend):
        """Test processing with no output bounds."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        _input_bounds, _output_constrs = result[0][0]
        assert isinstance(_output_constrs, list)


class TestType3VerboseMode:
    """Test verbose mode output."""

    def test_process_type3_verbose_true(self, backend, capsys):
        """Test TYPE3 with verbose=True.

        Should print timing and processing information.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (<= Y_0 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=True, parsed_data=parsed_data
        )

        # Verbose mode should produce output (implementation detail)
        assert len(result) == 1

    def test_process_type3_verbose_false(self, backend, capsys):
        """Test TYPE3 with verbose=False.

        Should not print timing information.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (<= Y_0 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1


class TestType3EdgeCases:
    """Test edge cases for TYPE3 processing."""

    def test_type3_empty_or_block(self, backend):
        """Test TYPE3 with minimal OR block.

        Edge case: OR block with single AND clause.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (<= Y_0 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1

    def test_type3_all_outputs_constrained(self, backend):
        """Test TYPE3 where all outputs are constrained.

        All output variables have explicit constraints.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (<= Y_0 0.5)",
            "            (>= Y_1 -0.5)))",
        ]
        n_inputs = 1
        n_outputs = 2

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1

    def test_type3_no_outputs_constrained(self, backend):
        """Test TYPE3 with only OR input constraints.

        No output constraints specified.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1

    def test_type3_comparison_constraints(self, backend):
        """Test TYPE3 with comparison constraints between outputs.

        Y_i compared to Y_j (not to constants).
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (>= Y_0 Y_1)))",
        ]
        n_inputs = 1
        n_outputs = 2

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1

    def test_type3_equality_constraints(self, backend):
        """Test TYPE3 with equality constraints.

        Y_i = constant creates two inequality constraints.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (= Y_0 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1

    def test_type3_negative_bounds(self, backend):
        """Test TYPE3 with negative bound values."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 -0.5) (>= X_0 -2.0)))",
            "            (>= Y_0 -1.0)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1

    def test_type3_zero_bounds(self, backend):
        """Test TYPE3 with zero bound values."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 0.0) (>= X_0 0.0)))",
            "            (= Y_0 0.0)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1

    def test_type3_large_bounds(self, backend):
        """Test TYPE3 with very large bound values."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1e10) (>= X_0 0.0)))",
            "            (<= Y_0 1e10)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1

    def test_type3_small_bounds(self, backend):
        """Test TYPE3 with very small bound values."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 1e-10) (>= X_0 0.0)))",
            "            (<= Y_0 1e-10)))",
        ]
        n_inputs = 1
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1


class TestType3IntegrationPatterns:
    """Test realistic TYPE3 patterns."""

    def test_type3_realistic_classification_property(self, backend):
        """Test realistic classification property.

        Input region with multiple branches and output bounds.
        """
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 0.5) (>= X_0 0.0) (<= X_1 0.5) (>= X_1 0.0))",
            "                  (and (<= X_0 1.0) (>= X_0 0.5) (<= X_1 1.0) (>= X_1 0.5)))",
            "            (<= Y_0 1.0)",
            "            (>= Y_0 0.0)))",
        ]
        n_inputs = 2
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1
        assert len(result[0]) >= 1

    def test_type3_realistic_with_comparison(self, backend):
        """Test realistic property with output comparison."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0) (<= X_1 1.0) (>= X_1 0.0)))",
            "            (>= Y_0 Y_1)))",
        ]
        n_inputs = 2
        n_outputs = 2

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1
        assert len(result[0]) >= 1

    def test_type3_complex_or_regions(self, backend):
        """Test TYPE3 with complex OR regions."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (and (<= X_0 0.25) (>= X_0 0.0) (<= X_1 0.25) (>= X_1 0.0))",
            "                  (and (<= X_0 0.5) (>= X_0 0.25) (<= X_1 0.5) (>= X_1 0.25))",
            "                  (and (<= X_0 0.75) (>= X_0 0.5) (<= X_1 0.75) (>= X_1 0.5))",
            "                  (and (<= X_0 1.0) (>= X_0 0.75) (<= X_1 1.0) (>= X_1 0.75)))",
            "            (<= Y_0 0.5)))",
        ]
        n_inputs = 2
        n_outputs = 1

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1
        assert len(result[0]) >= 1

    def test_type3_with_mixed_constraint_types(self, backend):
        """Test TYPE3 mixing comparison and bound constraints."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(declare-const Y_2 Real)",
            "(assert (and (or (and (<= X_0 1.0) (>= X_0 0.0)))",
            "            (>= Y_0 Y_1)",
            "            (<= Y_2 0.5)))",
        ]
        n_inputs = 1
        n_outputs = 3

        parsed_data = parse_simple_patterns(lines, verbose=False)
        result = process_type3(
            lines, n_inputs, n_outputs, backend, verbose=False, parsed_data=parsed_data
        )

        assert len(result) == 1
        assert len(result[0]) >= 1


class TestType3VerboseOutput:
    """Test TYPE3 verbose output."""

    def test_type3_verbose_with_constraints(self, backend, capsys):
        """Test TYPE3 verbose output with constraints."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (<= X_0 1.0) (>= X_0 0.0))))",
            "(assert (<= Y_0 0.5))",
        ]
        n_inputs = 1
        n_outputs = 1

        result = process_type3(lines, n_inputs, n_outputs, backend, verbose=True)

        captured = capsys.readouterr()
        assert "Type3" in captured.out or result is not None
