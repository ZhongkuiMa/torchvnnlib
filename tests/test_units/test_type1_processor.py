"""Unit tests for TYPE1 VNN-LIB processor - Comprehensive coverage.

TYPE1 structure: (and ...simple_input_constraints... ...output_constraints...)

Tests the process_type1() function with comprehensive coverage for all code paths.
TYPE1 is the most common VNN-LIB pattern (~70% of benchmarks).
"""

from torchvnnlib.fast_type import process_type1


class TestType1BasicProcessing:
    """Test basic TYPE1 processing functionality."""

    def test_process_type1_simple_case(self, backend):
        """Test TYPE1 with simple input bounds + simple output constraints.

        Tests basic path with all simple patterns (no complex expressions).
        Output constraints are grouped into tensors by type (comparisons, bounds).
        """
        simple_input_bounds = [
            (">=", "X_", 0, 0.5),
            ("<=", "X_", 0, 1.0),
        ]
        simple_output_constrs = [
            (">=", "Y_", 0, "Y_", 1),
        ]
        simple_output_bounds = [
            ("<=", "Y_", 1, 0.5),
        ]

        result = process_type1(
            simple_input_bounds,
            simple_output_constrs,
            [],
            n_inputs=1,
            n_outputs=2,
            backend=backend,
            simple_output_bounds=simple_output_bounds,
        )

        # Validate structure: list[list[tuple[input_bounds, output_constrs]]]
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1, "Should have 1 AND group"
        assert isinstance(result[0], list), "First element should be a list"
        assert len(result[0]) == 1, "Should have 1 tuple in AND group"

        _input_bounds, _output_constrs = result[0][0]

        # Validate input bounds
        assert tuple(_input_bounds.shape) == (1, 2), "Input bounds shape should be (1, 2)"
        input_bounds_np = (
            backend.to_numpy(_input_bounds) if hasattr(backend, "to_numpy") else _input_bounds
        )
        assert input_bounds_np[0, 0] >= 0.5 - 1e-6, "X_0 lower bound should be ~0.5"
        assert input_bounds_np[0, 1] <= 1.0 + 1e-6, "X_0 upper bound should be ~1.0"

        # Validate output constraints: list of constraint tensors (grouped by type)
        # With simple_output_constrs + simple_output_bounds, should have 2 tensors
        assert isinstance(_output_constrs, list), "Output constraints should be a list"
        assert len(_output_constrs) >= 1, "Should have at least one constraint tensor"

    def test_process_type1_multiple_inputs(self, backend):
        """Test TYPE1 with multiple input variables."""
        simple_input_bounds = [
            (">=", "X_", 0, -1.0),
            ("<=", "X_", 0, 2.0),
            (">=", "X_", 1, 0.0),
            ("<=", "X_", 1, 1.5),
        ]
        simple_output_constrs = []

        result = process_type1(
            simple_input_bounds, simple_output_constrs, [], n_inputs=2, n_outputs=2, backend=backend
        )

        _input_bounds, _output_constrs = result[0][0]

        # Validate shape matches number of inputs
        assert tuple(_input_bounds.shape) == (2, 2), "Input bounds shape should be (2, 2)"

        # Validate all bounds are set
        input_bounds_np = (
            backend.to_numpy(_input_bounds) if hasattr(backend, "to_numpy") else _input_bounds
        )
        assert (input_bounds_np[:, 0] <= input_bounds_np[:, 1] + 1e-6).all(), (
            "Lower bounds should be <= upper bounds"
        )

    def test_process_type1_multiple_output_constraints(self, backend):
        """Test TYPE1 with multiple output constraints and bounds.

        Output constraints are grouped into tensors by type:
        - One tensor for all comparison constraints (Y <op> Y format)
        - One tensor for all bound constraints (Y <op> value format)
        """
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        simple_output_constrs = [
            (">=", "Y_", 0, "Y_", 1),
            ("<=", "Y_", 0, "Y_", 2),
        ]
        simple_output_bounds = [
            (">=", "Y_", 0, 0.5),
            ("<=", "Y_", 1, 0.3),
            ("=", "Y_", 2, 0.0),
        ]

        result = process_type1(
            simple_input_bounds,
            simple_output_constrs,
            [],
            n_inputs=1,
            n_outputs=3,
            backend=backend,
            simple_output_bounds=simple_output_bounds,
        )

        _input_bounds, _output_constrs = result[0][0]
        assert isinstance(_output_constrs, list), "Should have output constraints list"
        # Should have 2 tensors: one for comparisons, one for bounds
        assert len(_output_constrs) == 2, "Should have comparison tensor + bound tensor"

        # Verify first tensor has 2 constraints (the comparisons)
        comparisons_tensor = _output_constrs[0]
        assert comparisons_tensor.shape[0] == 2, "Should have 2 comparison constraints"

        # Verify second tensor has 4 constraints (>= bound + <= bound + 2 for = bound)
        bounds_tensor = _output_constrs[1]
        assert bounds_tensor.shape[0] == 4, "Should have 4 bound constraints (>= + <= + 2 for =)"


class TestType1OutputConstraintProcessing:
    """Test output constraint processing in TYPE1."""

    def test_simple_output_bound_constraints(self, backend):
        """Test processing of simple output bounds.

        Output bounds are grouped into a single tensor with all bound constraints.
        """
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        simple_output_bounds = [
            (">=", "Y_", 0, 0.5),
            ("<=", "Y_", 1, 0.3),
        ]

        result = process_type1(
            simple_input_bounds,
            [],
            [],
            n_inputs=1,
            n_outputs=2,
            backend=backend,
            simple_output_bounds=simple_output_bounds,
        )

        _input_bounds, _output_constrs = result[0][0]
        # Should have one tensor with 2 bound constraints
        assert len(_output_constrs) == 1, "Should have bounds tensor"
        assert _output_constrs[0].shape[0] == 2, "Should have 2 bounds constraints"

    def test_simple_output_comparison_constraints(self, backend):
        """Test processing of simple output comparison constraints (Y0 >= Y1 style).

        Comparison constraints are grouped into a single tensor.
        """
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        simple_output_constrs = [
            (">=", "Y_", 0, "Y_", 1),
            ("<=", "Y_", 1, "Y_", 2),
        ]

        result = process_type1(
            simple_input_bounds, simple_output_constrs, [], n_inputs=1, n_outputs=3, backend=backend
        )

        _input_bounds, _output_constrs = result[0][0]
        # Should have one tensor with 2 comparison constraints
        assert len(_output_constrs) == 1, "Should have comparison constraints tensor"
        assert _output_constrs[0].shape[0] == 2, "Should have 2 comparison constraints"

    def test_equality_output_bounds(self, backend):
        """Test equality constraints in output bounds (Y = value).

        Each equality constraint Y = value creates 2 constraint rows (Y >= value and Y <= value).
        """
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        simple_output_bounds = [
            ("=", "Y_", 0, 0.5),  # Creates 2 constraints (>= and <=)
            ("=", "Y_", 1, -0.3),  # Creates 2 constraints (>= and <=)
        ]

        result = process_type1(
            simple_input_bounds,
            [],
            [],
            n_inputs=1,
            n_outputs=2,
            backend=backend,
            simple_output_bounds=simple_output_bounds,
        )

        _input_bounds, _output_constrs = result[0][0]
        # Should have one bounds tensor with 4 rows (2 equalities x 2 constraints each)
        assert len(_output_constrs) == 1, "Should have bounds tensor"
        assert _output_constrs[0].shape[0] == 4, "Equality bounds should create 2 constraints each"

    def test_mixed_output_constraints(self, backend):
        """Test mixing simple output constraints and bounds.

        Constraints are separated into: comparison tensor + bounds tensor.
        """
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        simple_output_constrs = [(">=", "Y_", 0, "Y_", 1)]
        simple_output_bounds = [("<=", "Y_", 0, 0.5), (">=", "Y_", 1, -0.2)]

        result = process_type1(
            simple_input_bounds,
            simple_output_constrs,
            [],
            n_inputs=1,
            n_outputs=2,
            backend=backend,
            simple_output_bounds=simple_output_bounds,
        )

        _input_bounds, _output_constrs = result[0][0]
        # Should have 2 tensors: one for comparisons, one for bounds
        assert len(_output_constrs) == 2, "Should have comparison tensor + bounds tensor"
        assert _output_constrs[0].shape[0] == 1, "Should have 1 comparison constraint"
        assert _output_constrs[1].shape[0] == 2, "Should have 2 bound constraints"


class TestType1ComplexExpressionsAST:
    """Test TYPE1 with complex output expressions (AST parsing)."""

    def test_complex_or_and_output_constraints(self, backend):
        """Test processing complex OR/AND expressions in output constraints.

        Complex expressions require AST parsing.
        """
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        # Complex output with OR/AND
        complex_lines = [
            "(assert (or (and (>= Y_0 0.5) (<= Y_1 0.3)) (and (>= Y_1 Y_0) (<= Y_0 0.2))))",
        ]

        result = process_type1(
            simple_input_bounds,
            [],
            complex_lines,
            n_inputs=1,
            n_outputs=2,
            backend=backend,
            simple_output_bounds=None,
        )

        _input_bounds, _output_constrs = result[0][0]
        assert isinstance(_output_constrs, list), "Should process complex constraints"
        assert len(_output_constrs) > 0, "Should extract constraints from complex OR/AND"

    def test_nested_and_output_constraints(self, backend):
        """Test nested AND expressions in output."""
        simple_input_bounds = [(">=", "X_", 0, 0.0), ("<=", "X_", 0, 1.0)]
        complex_lines = [
            "(assert (and (>= Y_0 0.5) (>= Y_1 Y_0) (<= Y_0 1.0)))",
        ]

        result = process_type1(
            simple_input_bounds, [], complex_lines, n_inputs=1, n_outputs=2, backend=backend
        )

        _input_bounds, _output_constrs = result[0][0]
        assert len(_output_constrs) > 0, "Should extract constraints from nested AND"

    def test_mixed_simple_and_complex_constraints(self, backend):
        """Test TYPE1 with both simple and complex output constraints.

        Simple constraints are grouped separately from complex expressions.
        """
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        simple_output_constrs = [(">=", "Y_", 0, "Y_", 1)]
        complex_lines = [
            "(assert (or (>= Y_0 0.5) (<= Y_1 0.3)))",
        ]

        result = process_type1(
            simple_input_bounds,
            simple_output_constrs,
            complex_lines,
            n_inputs=1,
            n_outputs=2,
            backend=backend,
        )

        _input_bounds, _output_constrs = result[0][0]
        # Should have at least 2 tensors: simple comparison + complex constraints
        assert len(_output_constrs) >= 2, "Should combine simple and complex constraints"


class TestType1EdgeCases:
    """Test edge cases and error handling."""

    def test_type1_no_output_constraints(self, backend):
        """Test TYPE1 with no output constraints at all.

        Should default to zero constraints.
        """
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]

        result = process_type1(
            simple_input_bounds, [], [], n_inputs=1, n_outputs=2, backend=backend
        )

        _input_bounds, _output_constrs = result[0][0]
        assert isinstance(_output_constrs, list), "Should have constraint list"
        assert len(_output_constrs) > 0, "Should have default zero constraints"

    def test_type1_empty_simple_output_constrs(self, backend):
        """Test with empty simple output constraints but with bounds."""
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        simple_output_bounds = [(">=", "Y_", 0, 0.5)]

        result = process_type1(
            simple_input_bounds,
            [],
            [],
            n_inputs=1,
            n_outputs=1,
            backend=backend,
            simple_output_bounds=simple_output_bounds,
        )

        _input_bounds, _output_constrs = result[0][0]
        assert len(_output_constrs) == 1, "Should have bounds tensor"
        assert _output_constrs[0].shape[0] == 1, "Should have 1 bound constraint"

    def test_type1_verbose_output(self, backend, capsys):
        """Test verbose mode prints timing information."""
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]

        result = process_type1(
            simple_input_bounds, [], [], n_inputs=1, n_outputs=1, backend=backend, verbose=True
        )

        captured = capsys.readouterr()
        output = captured.out

        # Should print timing information
        assert "Type1" in output or len(output) > 0, "Should have verbose output"

        # Should still return valid result
        assert len(result) == 1, "Should have valid result in verbose mode"

    def test_type1_all_output_variables_constrained(self, backend):
        """Test when all output variables have constraints.

        All outputs get explicit constraints (both comparison and bounds).
        """
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        simple_output_constrs = [
            (">=", "Y_", 0, "Y_", 1),
            (">=", "Y_", 1, "Y_", 2),
        ]
        simple_output_bounds = [
            (">=", "Y_", 0, 0.5),
            ("<=", "Y_", 1, 0.3),
            ("=", "Y_", 2, 0.0),
        ]

        result = process_type1(
            simple_input_bounds,
            simple_output_constrs,
            [],
            n_inputs=1,
            n_outputs=3,
            backend=backend,
            simple_output_bounds=simple_output_bounds,
        )

        _input_bounds, _output_constrs = result[0][0]
        # Should have 2 tensors: comparisons + bounds
        assert len(_output_constrs) == 2, "Should have comparison + bounds tensors"
        # Comparisons: 2 constraints, Bounds: 1 >= + 1 <= + 2 for = = 4 constraints
        assert _output_constrs[0].shape[0] == 2, "Should have 2 comparison constraints"
        assert _output_constrs[1].shape[0] == 4, "Should have 4 bound constraints"


class TestType1IntegrationWithBenchmarks:
    """Integration tests with benchmark patterns."""

    def test_type1_with_parsed_data(self, type1_parsed_data_simple, backend):
        """Test TYPE1 with pre-parsed data structure.

        Validates correct format matching parse_simple_patterns output.
        """
        parsed = type1_parsed_data_simple
        result = process_type1(
            parsed["simple_input_bounds"],
            parsed["simple_output_constrs"],
            parsed["complex_lines"],
            n_inputs=1,
            n_outputs=2,
            backend=backend,
            simple_output_bounds=parsed.get("simple_output_bounds"),
        )

        _input_bounds, _output_constrs = result[0][0]
        assert tuple(_input_bounds.shape) == (1, 2), "Should have correct input bounds shape"
        assert len(_output_constrs) > 0, "Should have output constraints"

    def test_type1_realistic_benchmark_pattern(self, backend):
        """Test TYPE1 with realistic VNN-COMP benchmark pattern.

        Tests a common real-world pattern with multiple inputs and bounds.
        """
        # Simulate parsing a real benchmark
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
            (">=", "X_", 1, -1.0),
            ("<=", "X_", 1, 1.0),
        ]
        simple_output_constrs = []
        simple_output_bounds = [
            ("<=", "Y_", 0, 0.5),
            (">=", "Y_", 0, -0.5),
        ]

        result = process_type1(
            simple_input_bounds,
            simple_output_constrs,
            [],
            n_inputs=2,
            n_outputs=1,
            backend=backend,
            simple_output_bounds=simple_output_bounds,
        )

        _input_bounds, _output_constrs = result[0][0]
        assert tuple(_input_bounds.shape) == (2, 2), "Should have 2 inputs"
        # Should have 1 bounds tensor with 2 constraints (<= and >=)
        assert len(_output_constrs) == 1, "Should have bounds tensor"
        assert _output_constrs[0].shape[0] == 2, "Should have 2 bound constraints"

    def test_type1_with_complex_ast_parsing(self, type1_parsed_data_complex, backend):
        """Test TYPE1 that requires AST parsing for complex expressions."""
        parsed = type1_parsed_data_complex
        result = process_type1(
            parsed["simple_input_bounds"],
            parsed["simple_output_constrs"],
            parsed["complex_lines"],
            n_inputs=2,
            n_outputs=2,
            backend=backend,
            simple_output_bounds=parsed.get("simple_output_bounds"),
        )

        _input_bounds, _output_constrs = result[0][0]
        assert tuple(_input_bounds.shape) == (2, 2), "Should process input bounds"
        assert len(_output_constrs) > 0, "Should extract constraints from complex expressions"


class TestType1ConstraintTensorShapes:
    """Test that constraint tensors have correct shapes."""

    def test_output_constraint_tensor_shape(self, backend):
        """Test that output constraint tensors have correct shape (n_constrs, n_outputs + 1)."""
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        simple_output_constrs = [(">=", "Y_", 0, "Y_", 1)]
        simple_output_bounds = [("<=", "Y_", 0, 0.5)]

        result = process_type1(
            simple_input_bounds,
            simple_output_constrs,
            [],
            n_inputs=1,
            n_outputs=2,
            backend=backend,
            simple_output_bounds=simple_output_bounds,
        )

        _input_bounds, _output_constrs = result[0][0]

        # Each constraint tensor should have shape (n_constraints, n_outputs + 1)
        for constr in _output_constrs:
            shape = tuple(constr.shape)
            assert len(shape) == 2, f"Constraint should be 2D, got shape {shape}"
            assert shape[1] == 3, f"Constraint width should be {2 + 1} for 2 outputs"

    def test_input_bounds_tensor_shape(self, backend):
        """Test that input bounds tensor has correct shape (n_inputs, 2)."""
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
            (">=", "X_", 1, -1.0),
            ("<=", "X_", 1, 1.0),
            (">=", "X_", 2, 0.5),
            ("<=", "X_", 2, 0.9),
        ]

        result = process_type1(
            simple_input_bounds, [], [], n_inputs=3, n_outputs=1, backend=backend
        )

        _input_bounds, _output_constrs = result[0][0]
        shape = tuple(_input_bounds.shape)
        assert shape == (3, 2), f"Input bounds should be (3, 2), got {shape}"


class TestType1ComplexExpressions:
    """Test TYPE1 processing with complex output expressions."""

    def test_type1_with_or_complex_output(self, backend):
        """Test TYPE1 with Or complex output expressions."""
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        simple_output_bounds = []
        complex_lines = [
            "(assert (or (>= Y_0 0.5) (<= Y_0 0.3)))",
        ]
        result = process_type1(
            simple_input_bounds,
            simple_output_bounds,
            complex_lines,
            n_inputs=1,
            n_outputs=1,
            backend=backend,
        )
        assert result is not None
        _input_bounds, _output_constrs = result[0][0]
        assert len(_output_constrs) > 0

    def test_type1_with_nested_and_or(self, backend):
        """Test TYPE1 with nested And/Or expressions."""
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        complex_lines = [
            "(assert (and (or (>= Y_0 0.5) (<= Y_0 0.3)) (>= Y_0 0.0)))",
        ]
        result = process_type1(
            simple_input_bounds,
            [],
            complex_lines,
            n_inputs=1,
            n_outputs=1,
            backend=backend,
        )
        assert result is not None

    def test_type1_with_simple_and_complex(self, backend):
        """Test TYPE1 with both simple and complex constraints."""
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        # Use output constraint format (5 elements)
        simple_output_constrs = [
            (">=", "Y_", 0, "X_", 0),
        ]
        complex_lines = [
            "(assert (and (<= Y_0 1.0)))",
        ]
        result = process_type1(
            simple_input_bounds,
            simple_output_constrs,
            complex_lines,
            n_inputs=1,
            n_outputs=1,
            backend=backend,
        )
        assert result is not None


class TestType1VerboseMode:
    """Test TYPE1 processing with verbose output."""

    def test_type1_verbose_with_complex(self, backend, capsys):
        """Test verbose output with complex expressions."""
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        # Complex line with And wrapper to ensure proper parsing
        complex_lines = [
            "(assert (and (>= Y_0 0.5)))",
        ]
        result = process_type1(
            simple_input_bounds,
            [],
            complex_lines,
            n_inputs=1,
            n_outputs=1,
            backend=backend,
            verbose=True,
        )
        assert result is not None

    def test_type1_verbose_mode(self, backend, capsys):
        """Test verbose output in TYPE1 processing."""
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        # Use output constraint format (5 elements) instead of bounds
        simple_output_constrs = [
            (">=", "Y_", 0, "Y_", 1),
        ]
        result = process_type1(
            simple_input_bounds,
            simple_output_constrs,
            [],
            n_inputs=1,
            n_outputs=2,
            backend=backend,
            verbose=True,
        )
        assert result is not None


class TestType1EmptyAndEdgeCases:
    """Test TYPE1 with empty constraints and edge cases."""

    def test_type1_empty_complex_lines(self, backend):
        """Test TYPE1 with empty complex lines."""
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        result = process_type1(
            simple_input_bounds,
            [],
            [],
            n_inputs=1,
            n_outputs=1,
            backend=backend,
        )
        assert result is not None
        _input_bounds, _output_constrs = result[0][0]
        # Should have default zeros
        assert len(_output_constrs) > 0

    def test_type1_constraint_comparison(self, backend):
        """Test TYPE1 with output constraint comparisons."""
        simple_input_bounds = [
            (">=", "X_", 0, 0.0),
            ("<=", "X_", 0, 1.0),
        ]
        # Use Y_0 >= X_0 format (5-element tuple)
        simple_output_constrs = [
            (">=", "Y_", 0, "X_", 0),
            ("<=", "Y_", 0, "X_", 0),
        ]
        result = process_type1(
            simple_input_bounds,
            simple_output_constrs,
            [],
            n_inputs=1,
            n_outputs=1,
            backend=backend,
        )
        assert result is not None
        _input_bounds, _output_constrs = result[0][0]
        # Should have constraints
        assert len(_output_constrs) > 0
