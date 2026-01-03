"""Unit tests for AST preprocessing.

Tests the public preprocess_vnnlib() API for preparing VNNLIB content.
Follows principles: broad code coverage, all logics, error handling, quick tests,
fine-grained information, public API only, no deselected tests.
"""

import pytest

from torchvnnlib.ast._preprocess import preprocess_vnnlib


class TestPreprocessBasic:
    """Test basic preprocessing functionality."""

    def test_preprocess_empty_content(self):
        """Test preprocessing empty content."""
        lines = []
        result = preprocess_vnnlib(lines)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3  # lines, n_inputs, n_outputs

    def test_preprocess_single_declare(self):
        """Test preprocessing single declaration."""
        lines = ["(declare-const X_0 Real)"]
        result = preprocess_vnnlib(lines)
        assert result is not None
        _lines_out, n_inputs, n_outputs = result
        assert n_inputs == 1
        assert n_outputs == 0

    def test_preprocess_with_assert(self):
        """Test preprocessing with assertions."""
        lines = [
            "(declare-const X_0 Real)",
            "(assert (<= X_0 1.0))",
        ]
        result = preprocess_vnnlib(lines)
        assert result is not None
        lines_out, _n_inputs, _n_outputs = result
        assert len(lines_out) == 1  # Only assert line remains
        assert "(assert" in lines_out[0]

    def test_preprocess_with_output_variables(self):
        """Test preprocessing with output variables."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (<= Y_0 0.5))",
        ]
        result = preprocess_vnnlib(lines)
        assert result is not None
        _lines_out, n_inputs, n_outputs = result
        assert n_inputs == 1
        assert n_outputs == 1


class TestPreprocessComments:
    """Test comment removal."""

    def test_preprocess_removes_line_comments(self):
        """Test that comments are removed."""
        lines = [
            "; This is a comment",
            "(declare-const X_0 Real)",
            "; Another comment",
            "(assert (<= X_0 1.0))",
        ]
        result = preprocess_vnnlib(lines)
        lines_out, _n_inputs, _n_outputs = result
        # Should only have assert line (declarations removed)
        assert len(lines_out) == 1
        assert "(assert" in lines_out[0]

    def test_preprocess_preserves_content_without_comments(self):
        """Test that content without comments is preserved."""
        lines = [
            "(declare-const X_0 Real)",
            "(assert (<= X_0 1.0))",
        ]
        result = preprocess_vnnlib(lines)
        lines_out, _n_inputs, _n_outputs = result
        assert len(lines_out) >= 1

    def test_preprocess_handles_multiple_comments(self):
        """Test handling multiple comment lines."""
        lines = [
            "; Comment 1",
            "(declare-const X_0 Real)",
            "; Comment 2",
            "; Comment 3",
            "(declare-const Y_0 Real)",
            "(assert (<= X_0 1.0))",
        ]
        result = preprocess_vnnlib(lines)
        lines_out, _n_inputs, _n_outputs = result
        # Should have 1 assert, no comments or declarations
        assert len(lines_out) == 1
        assert "(assert" in lines_out[0]


class TestPreprocessWhitespace:
    """Test whitespace handling."""

    def test_preprocess_empty_lines(self):
        """Test handling of empty lines."""
        lines = [
            "(declare-const X_0 Real)",
            "",
            "(declare-const X_1 Real)",
            "(assert (<= X_0 1.0))",
        ]
        result = preprocess_vnnlib(lines)
        assert result is not None

    def test_preprocess_simple_declaration(self):
        """Test simple declaration processing."""
        lines = ["(declare-const X_0 Real)"]
        result = preprocess_vnnlib(lines)
        assert result is not None
        _lines_out, n_inputs, _n_outputs = result
        assert n_inputs == 1

    def test_preprocess_multiple_declarations(self):
        """Test processing multiple declarations."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(assert (<= X_0 1.0))",
        ]
        result = preprocess_vnnlib(lines)
        assert result is not None
        _lines_out, n_inputs, _n_outputs = result
        assert n_inputs == 2

    def test_preprocess_mixed_variables(self):
        """Test processing mixed variable types."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (<= Y_0 0.5))",
        ]
        result = preprocess_vnnlib(lines)
        assert result is not None
        _lines_out, n_inputs, n_outputs = result
        assert n_inputs == 1
        assert n_outputs == 1


class TestPreprocessVariableDeclarations:
    """Test variable declaration handling."""

    def test_preprocess_input_variables(self):
        """Test extracting input variable declarations."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(declare-const X_2 Real)",
            "(assert (<= X_0 1.0))",
        ]
        result = preprocess_vnnlib(lines)
        _lines_out, n_inputs, n_outputs = result
        assert n_inputs == 3
        assert n_outputs == 0

    def test_preprocess_output_variables(self):
        """Test extracting output variable declarations."""
        lines = [
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (<= Y_0 0.5))",
        ]
        result = preprocess_vnnlib(lines)
        _lines_out, n_inputs, n_outputs = result
        assert n_inputs == 0
        assert n_outputs == 2

    def test_preprocess_all_variable_types(self):
        """Test with both input and output variables."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(declare-const Y_0 Real)",
            "(declare-const Y_1 Real)",
            "(assert (<= X_0 1.0))",
        ]
        result = preprocess_vnnlib(lines)
        _lines_out, n_inputs, n_outputs = result
        assert n_inputs == 2
        assert n_outputs == 2

    @pytest.mark.parametrize(
        ("var_decls", "expected_inputs", "expected_outputs"),
        [
            (
                [
                    "(declare-const X_0 Real)",
                    "(declare-const X_1 Real)",
                    "(declare-const X_2 Real)",
                ],
                3,
                0,
            ),
            (["(declare-const Y_0 Real)", "(declare-const Y_1 Real)"], 0, 2),
            (
                [
                    "(declare-const X_0 Real)",
                    "(declare-const X_1 Real)",
                    "(declare-const Y_0 Real)",
                    "(declare-const Y_1 Real)",
                ],
                2,
                2,
            ),
        ],
    )
    def test_preprocess_variable_patterns(self, var_decls, expected_inputs, expected_outputs):
        """Test different variable naming patterns."""
        lines = [*var_decls, "(assert (<= X_0 1.0))"]
        result = preprocess_vnnlib(lines)
        _lines_out, n_inputs, n_outputs = result
        assert n_inputs == expected_inputs
        assert n_outputs == expected_outputs


class TestPreprocessConstraints:
    """Test constraint handling."""

    def test_preprocess_simple_constraint(self):
        """Test simple constraint processing."""
        lines = [
            "(declare-const X_0 Real)",
            "(assert (<= X_0 1.0))",
        ]
        result = preprocess_vnnlib(lines)
        lines_out, _n_inputs, _n_outputs = result
        assert len(lines_out) == 1
        assert "(assert" in lines_out[0]

    def test_preprocess_multiple_constraints(self):
        """Test multiple constraint processing."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(assert (<= X_0 1.0))",
            "(assert (>= X_0 0.0))",
            "(assert (<= X_1 1.0))",
            "(assert (>= X_1 0.0))",
        ]
        result = preprocess_vnnlib(lines)
        lines_out, _n_inputs, _n_outputs = result
        assert len(lines_out) == 4

    def test_preprocess_logical_constraints(self):
        """Test logical constraint processing."""
        lines = [
            "(declare-const X_0 Real)",
            "(assert (and (<= X_0 1.0) (>= X_0 0.0)))",
        ]
        result = preprocess_vnnlib(lines)
        lines_out, _n_inputs, _n_outputs = result
        assert len(lines_out) == 1
        assert "and" in lines_out[0]

    def test_preprocess_or_constraints(self):
        """Test OR constraint processing."""
        lines = [
            "(declare-const X_0 Real)",
            "(assert (or (<= X_0 0.3) (>= X_0 0.7)))",
        ]
        result = preprocess_vnnlib(lines)
        lines_out, _n_inputs, _n_outputs = result
        assert len(lines_out) == 1
        assert "or" in lines_out[0]


class TestPreprocessReturnType:
    """Test return type and structure."""

    def test_preprocess_returns_tuple(self):
        """Test that preprocess returns tuple."""
        lines = [
            "(declare-const X_0 Real)",
            "(assert (<= X_0 1.0))",
        ]
        result = preprocess_vnnlib(lines)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_preprocess_tuple_structure(self):
        """Test tuple structure contains correct elements."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (<= X_0 1.0))",
        ]
        result = preprocess_vnnlib(lines)
        assert isinstance(result, tuple)
        assert len(result) == 3
        lines_out, n_inputs, n_outputs = result
        # First element should be lines
        assert isinstance(lines_out, list)
        # Second and third should be integers
        assert isinstance(n_inputs, int)
        assert isinstance(n_outputs, int)


class TestPreprocessComplexVNNLIB:
    """Test preprocessing complex VNNLIB content."""

    def test_preprocess_realistic_vnnlib(self):
        """Test with realistic VNNLIB content."""
        lines = [
            "; Test property",
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(declare-const Y_0 Real)",
            "; Input bounds",
            "(assert (and (<= X_0 1.0) (>= X_0 0.0)))",
            "(assert (and (<= X_1 1.0) (>= X_1 0.0)))",
            "; Output constraint",
            "(assert (<= Y_0 0.5))",
        ]
        result = preprocess_vnnlib(lines)
        lines_out, n_inputs, n_outputs = result
        assert len(lines_out) == 3  # Three assert statements
        assert n_inputs == 2
        assert n_outputs == 1

    def test_preprocess_complex_or_constraints(self):
        """Test complex OR constraints."""
        lines = [
            "(declare-const X_0 Real)",
            "(assert (or (<= X_0 0.3) (>= X_0 0.7)))",
        ]
        result = preprocess_vnnlib(lines)
        lines_out, _n_inputs, _n_outputs = result
        assert len(lines_out) == 1

    def test_preprocess_nested_logical_operators(self):
        """Test nested logical operators."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const X_1 Real)",
            "(assert (and (or (<= X_0 0.5) (>= X_0 0.75)) (<= X_1 1.0)))",
        ]
        result = preprocess_vnnlib(lines)
        lines_out, _n_inputs, _n_outputs = result
        assert len(lines_out) == 1
        assert "and" in lines_out[0]
        assert "or" in lines_out[0]
