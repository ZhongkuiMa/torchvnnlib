"""Unit tests for AST parsing.

Tests the public parse() API for converting token sequences to AST expressions.
Follows principles: broad code coverage, all logics, error handling, quick tests,
fine-grained information, public API only, no deselected tests.
"""

from collections import deque

import pytest

from torchvnnlib.ast._expr import Add, And, Cst, Div, Geq, Leq, Mul, Or, Sub, Var
from torchvnnlib.ast._parse import _parse_tokens, _parse_tokens_list, parse


class TestParseBasic:
    """Test basic parsing functionality."""

    def test_parse_single_constant(self):
        """Test parsing a single constant value."""
        tokens = deque(["5.0"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Cst)
        assert result.value == pytest.approx(5.0)

    def test_parse_single_variable(self):
        """Test parsing a single variable."""
        tokens = deque(["X_0"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Var)
        assert result.name == "X_0"

    def test_parse_simple_addition(self):
        """Test parsing simple addition."""
        tokens = deque(["(", "+", "1", "2", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Add)
        assert len(result.args) == 2

    def test_parse_simple_subtraction(self):
        """Test parsing subtraction."""
        tokens = deque(["(", "-", "X_0", "1.5", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Sub)

    def test_parse_multiplication(self):
        """Test parsing multiplication."""
        tokens = deque(["(", "*", "2.0", "X_0", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Mul)

    def test_parse_division(self):
        """Test parsing division."""
        tokens = deque(["(", "/", "Y_0", "2.0", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Div)


class TestParseComparisons:
    """Test parsing comparison operators."""

    def test_parse_less_equal(self):
        """Test parsing <= constraint."""
        tokens = deque(["(", "<=", "X_0", "1.0", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Leq)

    def test_parse_greater_equal(self):
        """Test parsing >= constraint."""
        tokens = deque(["(", ">=", "X_0", "0.0", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Geq)

    @pytest.mark.parametrize(
        "tokens_list",
        [
            deque(["(", "<=", "X_0", "1.0", ")"]),
            deque(["(", ">=", "X_0", "0.0", ")"]),
        ],
    )
    def test_parse_comparison_operators(self, tokens_list):
        """Test parsing various comparison operators."""
        result = _parse_tokens(tokens_list)
        assert result is not None
        assert hasattr(result, "has_input_vars") or hasattr(result, "has_output_vars")


class TestParseLogicalOperators:
    """Test parsing logical operators."""

    def test_parse_and_two_constraints(self):
        """Test parsing AND with two constraints."""
        tokens = deque(
            ["(", "and", "(", "<=", "X_0", "1.0", ")", "(", ">=", "X_0", "0.0", ")", ")"]
        )
        result = _parse_tokens(tokens)
        assert isinstance(result, And)
        assert len(result.args) == 2

    def test_parse_or_two_constraints(self):
        """Test parsing OR with two constraints."""
        tokens = deque(
            ["(", "or", "(", "<=", "X_0", "0.5", ")", "(", ">=", "X_0", "0.75", ")", ")"]
        )
        result = _parse_tokens(tokens)
        assert isinstance(result, Or)
        assert len(result.args) == 2

    def test_parse_and_multiple_constraints(self):
        """Test parsing AND with multiple constraints."""
        tokens = deque(
            [
                "(",
                "and",
                "(",
                "<=",
                "X_0",
                "1.0",
                ")",
                "(",
                ">=",
                "X_0",
                "0.0",
                ")",
                "(",
                "<=",
                "X_1",
                "1.0",
                ")",
                ")",
            ]
        )
        result = _parse_tokens(tokens)
        assert isinstance(result, And)
        assert len(result.args) == 3


class TestParseComplexExpressions:
    """Test parsing complex nested expressions."""

    def test_parse_nested_arithmetic_multiplication(self):
        """Test parsing multiplication inside addition."""
        # (+ (* 2.0 X_0) X_1)
        tokens = deque(["(", "+", "(", "*", "2.0", "X_0", ")", "X_1", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Add)
        assert len(result.args) == 2

    def test_parse_addition_with_variables(self):
        """Test parsing addition with multiple variables."""
        # (+ X_0 X_1)
        tokens = deque(["(", "+", "X_0", "X_1", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Add)
        assert len(result.args) == 2
        assert result.has_input_vars is True

    def test_parse_subtraction_with_variables(self):
        """Test parsing subtraction with variables."""
        # (- Y_0 Y_1)
        tokens = deque(["(", "-", "Y_0", "Y_1", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Sub)
        assert result.has_output_vars is True

    def test_parse_nested_and_with_constraints(self):
        """Test parsing AND with two constraints."""
        # (and (<= X_0 1.0) (>= X_0 0.0))
        tokens = deque(
            ["(", "and", "(", "<=", "X_0", "1.0", ")", "(", ">=", "X_0", "0.0", ")", ")"]
        )
        result = _parse_tokens(tokens)
        assert isinstance(result, And)
        assert len(result.args) == 2


class TestParseVariables:
    """Test parsing with different variable types."""

    @pytest.mark.parametrize(
        "var_name",
        ["X_0", "X_1", "X_999", "Y_0", "Y_1", "Y_10000"],
    )
    def test_parse_variable_names(self, var_name):
        """Test parsing various variable names."""
        tokens = deque([var_name])
        result = _parse_tokens(tokens)
        assert isinstance(result, Var)
        assert result.name == var_name

    def test_parse_mixed_variable_types(self):
        """Test parsing expression with input and output variables."""
        # (+ X_0 Y_0)
        tokens = deque(["(", "+", "X_0", "Y_0", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Add)
        assert result.has_input_vars is True
        assert result.has_output_vars is True


class TestParseTokensList:
    """Test parse_tokens_list function for multiple expressions."""

    def test_parse_tokens_list_single(self):
        """Test parsing single token list."""
        tokens_list = [deque(["(", "+", "1", "2", ")"])]
        result = _parse_tokens_list(tokens_list)
        assert len(result) == 1
        assert isinstance(result[0], Add)

    def test_parse_tokens_list_multiple(self):
        """Test parsing multiple token lists."""
        tokens_list = [
            deque(["(", "+", "1", "2", ")"]),
            deque(["(", "-", "3", "1", ")"]),
            deque(["(", "*", "X_0", "2.0", ")"]),
        ]
        result = _parse_tokens_list(tokens_list)
        assert len(result) == 3
        assert isinstance(result[0], Add)
        assert isinstance(result[1], Sub)
        assert isinstance(result[2], Mul)


class TestParseNumbers:
    """Test parsing different number formats."""

    @pytest.mark.parametrize(
        "number",
        ["0", "1", "42", "999999"],
    )
    def test_parse_integers(self, number):
        """Test parsing integer values."""
        tokens = deque([number])
        result = _parse_tokens(tokens)
        assert isinstance(result, Cst)
        assert float(result.value) == float(number)

    @pytest.mark.parametrize(
        "number",
        ["0.0", "1.5", "3.14", "2.71828"],
    )
    def test_parse_floats(self, number):
        """Test parsing floating point values."""
        tokens = deque([number])
        result = _parse_tokens(tokens)
        assert isinstance(result, Cst)
        assert result.value == pytest.approx(float(number))

    @pytest.mark.parametrize(
        "number",
        ["1e-10", "1e308", "2.5E+3", "3.2e-5"],
    )
    def test_parse_scientific_notation(self, number):
        """Test parsing scientific notation."""
        tokens = deque([number])
        result = _parse_tokens(tokens)
        assert isinstance(result, Cst)
        assert result.value == pytest.approx(float(number))

    @pytest.mark.parametrize(
        "number",
        ["-1", "-1.5", "-1e-10"],
    )
    def test_parse_negative_numbers(self, number):
        """Test parsing negative numbers."""
        tokens = deque([number])
        result = _parse_tokens(tokens)
        assert isinstance(result, Cst)
        assert result.value == pytest.approx(float(number))


class TestParseProperties:
    """Test expression properties after parsing."""

    def test_parse_input_variable_property(self):
        """Test that parsed input variable has correct property."""
        tokens = deque(["X_5"])
        result = _parse_tokens(tokens)
        assert result.has_input_vars is True
        assert result.has_output_vars is False

    def test_parse_output_variable_property(self):
        """Test that parsed output variable has correct property."""
        tokens = deque(["Y_5"])
        result = _parse_tokens(tokens)
        assert result.has_output_vars is True
        assert result.has_input_vars is False

    def test_parse_constant_has_no_variables(self):
        """Test that constant has no variables."""
        tokens = deque(["5.0"])
        result = _parse_tokens(tokens)
        assert result.has_input_vars is False
        assert result.has_output_vars is False

    def test_parse_expression_propagates_variables(self):
        """Test that expression propagates variable properties."""
        # (+ X_0 1.0)
        tokens = deque(["(", "+", "X_0", "1.0", ")"])
        result = _parse_tokens(tokens)
        assert result.has_input_vars is True
        assert result.has_output_vars is False


class TestParseBinaryOperators:
    """Test parsing binary operators."""

    def test_parse_subtraction_basic(self):
        """Test basic subtraction without parentheses."""
        # (- 5 3)
        tokens = deque(["(", "-", "5", "3", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Sub)

    def test_parse_multiplication_basic(self):
        """Test basic multiplication without parentheses."""
        # (* 2.0 X_0)
        tokens = deque(["(", "*", "2.0", "X_0", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Mul)

    def test_parse_division_basic(self):
        """Test basic division without parentheses."""
        # (/ Y_0 2.0)
        tokens = deque(["(", "/", "Y_0", "2.0", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Div)

    def test_parse_comparison_leq_basic(self):
        """Test basic <= comparison."""
        # (<= X_0 1.0)
        tokens = deque(["(", "<=", "X_0", "1.0", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Leq)

    def test_parse_comparison_geq_basic(self):
        """Test basic >= comparison."""
        # (>= X_0 0.0)
        tokens = deque(["(", ">=", "X_0", "0.0", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Geq)


class TestParseVerboseMode:
    """Test verbose output from parsing functions."""

    def test_parse_tokens_list_verbose_sequential(self, capsys):
        """Test verbose output with sequential parsing."""
        tokens_list = [
            deque(["(", "+", "1", "2", ")"]),
            deque(["(", "-", "3", "1", ")"]),
        ]
        result = _parse_tokens_list(tokens_list, verbose=True, use_parallel=False)
        captured = capsys.readouterr()
        assert "Parse tokens (sequential)" in captured.out
        assert len(result) == 2
        assert isinstance(result[0], Add)
        assert isinstance(result[1], Sub)

    def test_parse_tokens_list_verbose_parallel(self, capsys):
        """Test verbose output with parallel parsing."""
        tokens_list = [
            deque(["(", "+", "1", "2", ")"]),
            deque(["(", "-", "3", "1", ")"]),
            deque(["(", "*", "2", "5", ")"]),
        ]
        result = _parse_tokens_list(tokens_list, verbose=True, use_parallel=True)
        captured = capsys.readouterr()
        assert "Parse tokens (parallel)" in captured.out
        assert len(result) == 3

    def test_parse_verbose_mode(self, capsys):
        """Test verbose output from main parse function."""
        tokens_list = [
            deque(["(", "+", "1", "2", ")"]),
            deque(["(", "<=", "X_0", "1.0", ")"]),
        ]
        result = parse(tokens_list, verbose=True, use_parallel=False)
        captured = capsys.readouterr()
        assert "Parse tokens (sequential)" in captured.out
        assert "Merge exprs" in captured.out
        assert result is not None


class TestParseErrorHandling:
    """Test error handling in parsing."""

    def test_parse_unknown_operator(self):
        """Test error handling for unknown operators."""
        # (unknown_op 1 2)
        tokens = deque(["(", "unknown_op", "1", "2", ")"])
        with pytest.raises(ValueError, match="Unknown operator"):
            _parse_tokens(tokens)

    def test_parse_unknown_operator_with_variables(self):
        """Test error handling for unknown operators with variables."""
        # (invalid X_0 Y_0)
        tokens = deque(["(", "invalid", "X_0", "Y_0", ")"])
        with pytest.raises(ValueError, match="Unknown operator"):
            _parse_tokens(tokens)

    def test_parse_assert_statement(self):
        """Test parsing assert statements."""
        # (assert (<= X_0 1.0))
        tokens = deque(["(", "assert", "(", "<=", "X_0", "1.0", ")", ")"])
        result = _parse_tokens(tokens)
        assert isinstance(result, Leq)
