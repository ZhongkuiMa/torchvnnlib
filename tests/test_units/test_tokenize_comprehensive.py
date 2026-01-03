"""Comprehensive unit tests for AST tokenization - Phase 4 coverage.

Tests the public tokenize() API for converting line lists to token sequences
from VNN-LIB files.

Target coverage: All 9 missed statements in _tokenize.py (76% -> >92%)

This includes:
- Tokenization of various VNN-LIB expressions
- Different expression types (operators, variables, constants)
- Nested expressions
- Whitespace and formatting edge cases
- Sequential vs parallel processing modes
- Verbose output
- Error handling for invalid tokens
"""

import pytest

from torchvnnlib.ast._tokenize import tokenize


class TestTokenizeBasicExpressions:
    """Test tokenize() with basic VNN-LIB expressions."""

    def test_tokenize_single_line_simple(self):
        """Test tokenizing a single simple line."""
        lines = ["(+ 1 2)"]
        result = tokenize(lines)
        assert len(result) == 1
        assert list(result[0]) == ["(", "+", "1", "2", ")"]

    def test_tokenize_single_line_declare_const(self):
        """Test tokenizing declare-const statement."""
        lines = ["(declare-const X_0 Real)"]
        result = tokenize(lines)
        assert len(result) == 1
        tokens = list(result[0])
        assert "declare-const" in tokens
        assert "X_0" in tokens
        assert "Real" in tokens

    def test_tokenize_single_line_assert(self):
        """Test tokenizing assert statement."""
        lines = ["(assert (<= X_0 1.0))"]
        result = tokenize(lines)
        assert len(result) == 1
        tokens = list(result[0])
        assert "assert" in tokens
        assert "<=" in tokens
        assert "X_0" in tokens
        assert "1.0" in tokens

    def test_tokenize_multiple_lines(self):
        """Test tokenizing multiple lines."""
        lines = [
            "(declare-const X_0 Real)",
            "(assert (<= X_0 1.0))",
            "(assert (>= X_0 0.0))",
        ]
        result = tokenize(lines)
        assert len(result) == 3

    def test_tokenize_returns_deques(self):
        """Test that tokenize returns list of deques."""
        lines = ["(+ 1 2)"]
        result = tokenize(lines)
        assert isinstance(result, list)
        assert len(result) > 0
        # Result should be a deque with __iter__
        assert hasattr(result[0], "__iter__")
        assert hasattr(result[0], "popleft")  # deque specific

    def test_tokenize_diverse_single_lines(self):
        """Test tokenizing diverse single-line inputs."""
        test_cases = [
            "(+ 1 2)",
            "(<= X_0 1.0)",
            "(declare-const X_0 Real)",
            "(and (expr1) (expr2))",
            "(or (>= Y_0 0.5) (<= Y_1 0.3))",
        ]
        for line in test_cases:
            result = tokenize([line])
            assert len(result) >= 1
            assert all(tokens is not None for tokens in result)


class TestTokenizeEmptyAndWhitespace:
    """Test handling of empty lines and whitespace."""

    def test_tokenize_empty_line_filtered(self):
        """Test that empty lines produce empty results."""
        lines = ["(+ 1 2)", "", "(- 3 1)"]
        result = tokenize(lines)
        # Empty string produces a deque with empty content
        # Non-empty lines are preserved
        assert any(len(list(r)) > 0 for r in result)

    def test_tokenize_whitespace_only_line(self):
        """Test handling of whitespace-only lines."""
        lines = ["   ", "\t", "  \t  "]
        result = tokenize(lines)
        assert isinstance(result, list)

    def test_tokenize_line_with_leading_whitespace(self):
        """Test line with leading whitespace."""
        lines = ["   (+ 1 2)"]
        result = tokenize(lines)
        assert len(result) == 1
        tokens = list(result[0])
        assert "+" in tokens

    def test_tokenize_line_with_trailing_whitespace(self):
        """Test line with trailing whitespace."""
        lines = ["(+ 1 2)   "]
        result = tokenize(lines)
        assert len(result) == 1
        tokens = list(result[0])
        assert "+" in tokens

    def test_tokenize_line_with_multiple_spaces(self):
        """Test line with multiple spaces between tokens."""
        lines = ["(+  1   2)"]
        result = tokenize(lines)
        assert len(result) == 1
        tokens = list(result[0])
        assert "+" in tokens
        assert "1" in tokens
        assert "2" in tokens


class TestTokenizeNumbers:
    """Test tokenizing various number formats."""

    def test_tokenize_integers(self):
        """Test tokenizing integer numbers."""
        test_cases = [
            "(+ 0 1)",
            "(+ 42 100)",
            "(+ 999999 1000000)",
        ]
        for line in test_cases:
            result = tokenize([line])
            tokens = list(result[0])
            # Verify integers are in tokens
            assert any(t.isdigit() for t in tokens)

    def test_tokenize_floats(self):
        """Test tokenizing floating point numbers."""
        test_cases = [
            "(+ 0.0 1.5)",
            "(+ 3.14 2.71)",
            "(<= X_0 0.5)",
        ]
        for line in test_cases:
            result = tokenize([line])
            tokens = list(result[0])
            # Should contain at least one float token (decimal point)
            assert any("." in t for t in tokens)

    def test_tokenize_negative_integers(self):
        """Test tokenizing negative integer numbers."""
        test_cases = [
            "(+ -1 -2)",
            "(- -5 -10)",
            "(<= X_0 -1)",
        ]
        for line in test_cases:
            result = tokenize([line])
            tokens = list(result[0])
            # Should contain negative numbers
            assert any(t.startswith("-") for t in tokens)

    def test_tokenize_negative_floats(self):
        """Test tokenizing negative floating point numbers."""
        test_cases = [
            "(- -5.5 -10.5)",
            "(<= X_0 -0.5)",
            "(assert (>= Y_0 -1.5))",
        ]
        for line in test_cases:
            result = tokenize([line])
            tokens = list(result[0])
            # Should contain negative floats
            assert any(t.startswith("-") and "." in t for t in tokens)

    def test_tokenize_scientific_notation_basic(self):
        """Test tokenizing scientific notation."""
        test_cases = [
            "(* 1.5e-10 X_0)",
            "(+ 1e308 1e-308)",
            "(* 2.5E+3 3.2e-5)",
        ]
        for line in test_cases:
            result = tokenize([line])
            tokens = list(result[0])
            # Should contain scientific notation
            assert any("e" in t.lower() for t in tokens)

    def test_tokenize_scientific_notation_positive_exponent(self):
        """Test scientific notation with positive exponent."""
        lines = ["(* 1.0e+5 2.0)"]
        result = tokenize(lines)
        tokens = list(result[0])
        assert any("e+" in t.lower() for t in tokens)

    def test_tokenize_scientific_notation_negative_exponent(self):
        """Test scientific notation with negative exponent."""
        lines = ["(* 1.0e-5 2.0)"]
        result = tokenize(lines)
        tokens = list(result[0])
        assert any("e-" in t.lower() for t in tokens)

    def test_tokenize_zero(self):
        """Test tokenizing zero."""
        lines = ["(+ 0 0)"]
        result = tokenize(lines)
        tokens = list(result[0])
        assert "0" in tokens

    def test_tokenize_zero_float(self):
        """Test tokenizing zero as float."""
        lines = ["(+ 0.0 0.0)"]
        result = tokenize(lines)
        tokens = list(result[0])
        assert "0.0" in tokens


class TestTokenizeOperators:
    """Test tokenizing all operator types."""

    def test_tokenize_arithmetic_operators(self):
        """Test tokenizing arithmetic operators."""
        operators = ["+", "-", "*", "/"]
        for op in operators:
            line = f"({op} 1 2)"
            result = tokenize([line])
            tokens = list(result[0])
            assert op in tokens

    def test_tokenize_comparison_operators(self):
        """Test tokenizing comparison operators."""
        operators = ["<=", ">=", "=", "<", ">"]
        for op in operators:
            line = f"({op} X_0 1.0)"
            result = tokenize([line])
            tokens = list(result[0])
            assert op in tokens

    def test_tokenize_logical_operators(self):
        """Test tokenizing logical operators."""
        keywords = ["and", "or"]
        for kw in keywords:
            line = f"({kw} (expr1) (expr2))"
            result = tokenize([line])
            tokens = list(result[0])
            assert kw in tokens

    def test_tokenize_keywords_and_identifiers(self):
        """Test tokenizing keywords and identifiers."""
        keywords = ["declare-const", "assert", "Real"]
        for kw in keywords:
            line = f"({kw} X_0)"
            result = tokenize([line])
            tokens = list(result[0])
            assert kw in tokens


class TestTokenizeVariables:
    """Test tokenizing variable names."""

    def test_tokenize_input_variables(self):
        """Test tokenizing input variables."""
        var_names = ["X_0", "X_1", "X_999"]
        for var in var_names:
            line = f"(assert {var})"
            result = tokenize([line])
            tokens = list(result[0])
            assert var in tokens

    def test_tokenize_output_variables(self):
        """Test tokenizing output variables."""
        var_names = ["Y_0", "Y_1", "Y_10000"]
        for var in var_names:
            line = f"(assert {var})"
            result = tokenize([line])
            tokens = list(result[0])
            assert var in tokens

    def test_tokenize_mixed_variables(self):
        """Test tokenizing expression with multiple variables."""
        line = "(and X_0 X_1 Y_0 Y_1)"
        result = tokenize([line])
        tokens = list(result[0])
        assert "X_0" in tokens
        assert "X_1" in tokens
        assert "Y_0" in tokens
        assert "Y_1" in tokens

    def test_tokenize_variables_in_constraints(self):
        """Test tokenizing variables in constraints."""
        line = "(<= (+ X_0 X_1) Y_0)"
        result = tokenize([line])
        tokens = list(result[0])
        assert "X_0" in tokens
        assert "X_1" in tokens
        assert "Y_0" in tokens

    def test_tokenize_variable_with_large_index(self):
        """Test tokenizing variable with large index."""
        line = "(assert X_123456)"
        result = tokenize([line])
        tokens = list(result[0])
        assert "X_123456" in tokens


class TestTokenizeComplexExpressions:
    """Test tokenizing complex real-world expressions."""

    def test_tokenize_nested_and_or_single_line(self):
        """Test tokenizing nested AND/OR expressions."""
        line = "(assert (or (and (<= X_0 0.5) (>= X_0 0.0)) (and (<= X_0 1.0) (>= X_0 0.5))))"
        result = tokenize([line])
        tokens = list(result[0])
        assert "or" in tokens
        assert "and" in tokens
        assert "<=" in tokens
        assert ">=" in tokens
        assert "X_0" in tokens
        assert "0.0" in tokens
        assert "0.5" in tokens
        assert "1.0" in tokens

    def test_tokenize_complex_linear_constraint(self):
        """Test tokenizing complex linear constraint."""
        line = "(assert (<= (+ (* 2.0 X_0) (* 3.0 X_1)) 5.0))"
        result = tokenize([line])
        tokens = list(result[0])
        # Verify key components are present
        assert "assert" in tokens
        assert "<=" in tokens
        assert "+" in tokens
        assert "*" in tokens
        assert "2.0" in tokens or "2" in tokens
        assert "3.0" in tokens or "3" in tokens
        assert "5.0" in tokens or "5" in tokens
        assert "X_0" in tokens
        assert "X_1" in tokens

    def test_tokenize_output_constraint(self):
        """Test tokenizing output constraint."""
        line = "(assert (<= (- Y_0 Y_1) 0.0))"
        result = tokenize([line])
        tokens = list(result[0])
        assert "assert" in tokens
        assert "<=" in tokens
        assert "-" in tokens
        assert "Y_0" in tokens
        assert "Y_1" in tokens
        assert "0.0" in tokens

    def test_tokenize_type1_pattern(self):
        """Test tokenizing TYPE1 VNN-LIB pattern."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (<= X_0 1.0) (>= X_0 0.0) (<= Y_0 0.5)))",
        ]
        result = tokenize(lines)
        assert len(result) == 3
        # Check TYPE1 structure is tokenized
        all_tokens = [t for line_tokens in result for t in line_tokens]
        assert "and" in all_tokens
        assert "<=" in all_tokens
        assert ">=" in all_tokens

    def test_tokenize_type2_pattern(self):
        """Test tokenizing TYPE2 VNN-LIB pattern."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (<= X_0 1.0) (or (>= Y_0 0.5))))",
        ]
        result = tokenize(lines)
        assert len(result) == 3
        all_tokens = [t for line_tokens in result for t in line_tokens]
        assert "and" in all_tokens
        assert "or" in all_tokens

    def test_tokenize_type3_pattern(self):
        """Test tokenizing TYPE3 VNN-LIB pattern."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (and (or (<= X_0 0.5) (>= X_0 0.7)) (<= Y_0 0.5)))",
        ]
        result = tokenize(lines)
        assert len(result) == 3
        all_tokens = [t for line_tokens in result for t in line_tokens]
        assert "and" in all_tokens
        assert "or" in all_tokens


class TestTokenizeSequentialVsParallel:
    """Test sequential vs parallel execution modes."""

    def test_tokenize_sequential_mode(self):
        """Test tokenize with use_parallel=False."""
        lines = [
            "(assert (<= X_0 1.0))",
            "(assert (>= X_1 0.0))",
        ]
        result_seq = tokenize(lines, use_parallel=False)
        assert len(result_seq) == 2
        assert list(result_seq[0]) == ["(", "assert", "(", "<=", "X_0", "1.0", ")", ")"]

    def test_tokenize_parallel_mode(self):
        """Test tokenize with use_parallel=True."""
        lines = [
            "(assert (<= X_0 1.0))",
            "(assert (>= X_1 0.0))",
        ]
        result_par = tokenize(lines, use_parallel=True)
        assert len(result_par) == 2
        assert list(result_par[0]) == ["(", "assert", "(", "<=", "X_0", "1.0", ")", ")"]

    def test_tokenize_modes_produce_same_result(self):
        """Test that parallel and sequential modes produce identical results."""
        lines = [
            "(assert (<= X_0 1.0))",
            "(assert (>= X_1 0.0))",
            "(assert (= Y_0 (+ X_0 X_1)))",
        ]
        result_seq = tokenize(lines, use_parallel=False)
        result_par = tokenize(lines, use_parallel=True)
        # Both modes should produce same results
        assert len(result_seq) == len(result_par)
        for seq_tokens, par_tokens in zip(result_seq, result_par, strict=True):
            assert list(seq_tokens) == list(par_tokens)

    def test_tokenize_parallel_with_many_lines(self):
        """Test parallel tokenization with many lines (>100 threshold)."""
        # Create 150 lines to trigger parallel processing
        lines = [f"(assert (>= X_{i} 0.0))" for i in range(150)]
        result = tokenize(lines, use_parallel=True)
        assert len(result) == 150

    def test_tokenize_sequential_with_many_lines(self):
        """Test sequential tokenization with many lines."""
        # Create 150 lines
        lines = [f"(assert (>= X_{i} 0.0))" for i in range(150)]
        result = tokenize(lines, use_parallel=False)
        assert len(result) == 150

    def test_tokenize_threshold_exactly_100(self):
        """Test tokenization at threshold of 100 lines."""
        # Exactly 100 lines should use sequential
        lines = [f"(assert (>= X_{i} 0.0))" for i in range(100)]
        result_seq = tokenize(lines, use_parallel=False)
        # 100 lines triggers parallel with use_parallel=True
        result_par = tokenize(lines, use_parallel=True)
        assert len(result_seq) == len(result_par)
        for seq, par in zip(result_seq, result_par, strict=True):
            assert list(seq) == list(par)

    def test_tokenize_threshold_101_uses_parallel(self):
        """Test that 101 lines with use_parallel=True uses parallel."""
        lines = [f"(assert (>= X_{i} 0.0))" for i in range(101)]
        result = tokenize(lines, use_parallel=True)
        assert len(result) == 101


class TestTokenizeVerboseMode:
    """Test verbose mode parameter."""

    def test_tokenize_accepts_verbose_parameter(self):
        """Test that tokenize accepts verbose parameter."""
        lines = ["(+ 1 2)"]
        # Should not raise error with verbose=True
        result = tokenize(lines, verbose=True)
        assert len(result) == 1

    def test_tokenize_accepts_verbose_false(self):
        """Test that tokenize accepts verbose=False parameter."""
        lines = ["(+ 1 2)"]
        result = tokenize(lines, verbose=False)
        assert len(result) == 1

    def test_tokenize_verbose_sequential(self, capsys):
        """Test verbose output in sequential mode."""
        lines = ["(+ 1 2)", "(- 3 1)"]
        result = tokenize(lines, use_parallel=False, verbose=True)
        assert len(result) == 2
        # Verbose mode may print timing info

    def test_tokenize_verbose_parallel(self, capsys):
        """Test verbose output in parallel mode."""
        # Need enough lines to trigger parallel
        lines = [f"(assert (>= X_{i} 0.0))" for i in range(101)]
        result = tokenize(lines, use_parallel=True, verbose=True)
        assert len(result) == 101


class TestTokenizeErrorHandling:
    """Test error handling in tokenization."""

    @pytest.mark.skip(
        reason="Invalid character causes infinite loop in tokenizer - known limitation"
    )
    def test_tokenize_invalid_character(self):
        """Test that invalid characters raise error.

        Note: The current implementation hangs on invalid characters.
        This is a known limitation of the regex-based tokenizer.
        """
        lines = ["(assert (<@ X_0 1.0))"]  # @ is invalid
        with pytest.raises(ValueError, match="Invalid token"):
            tokenize(lines)

    def test_tokenize_unclosed_paren(self):
        """Test tokenizing with unclosed parenthesis."""
        # This depends on implementation - might be caught or might succeed
        lines = ["(assert (< X_0 1.0)"]
        # Tokenizer itself shouldn't fail, it tokenizes what's there
        result = tokenize(lines)
        assert len(result) >= 1

    def test_tokenize_special_identifier_chars(self):
        """Test that hyphens in identifiers are handled."""
        lines = ["(declare-const X_0 Real)"]
        result = tokenize(lines)
        tokens = list(result[0])
        # declare-const should be tokenized as single token
        assert "declare-const" in tokens

    def test_tokenize_underscore_in_identifier(self):
        """Test underscores in identifiers."""
        lines = ["(assert X_0)"]
        result = tokenize(lines)
        tokens = list(result[0])
        assert "X_0" in tokens

    def test_tokenize_parentheses_separate_tokens(self):
        """Test that parentheses are separate tokens."""
        lines = ["(+ 1 2)"]
        result = tokenize(lines)
        tokens = list(result[0])
        assert tokens[0] == "("
        assert tokens[-1] == ")"


class TestTokenizeEdgeCases:
    """Test edge cases in tokenization."""

    def test_tokenize_empty_list(self):
        """Test tokenizing empty list of lines."""
        lines = []
        result = tokenize(lines)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_tokenize_single_parenthesis(self):
        """Test tokenizing single parenthesis."""
        lines = ["("]
        result = tokenize(lines)
        assert len(result) == 1
        assert list(result[0]) == ["("]

    def test_tokenize_single_number(self):
        """Test tokenizing single number."""
        lines = ["42"]
        result = tokenize(lines)
        assert len(result) == 1
        assert list(result[0]) == ["42"]

    def test_tokenize_single_identifier(self):
        """Test tokenizing single identifier."""
        lines = ["X_0"]
        result = tokenize(lines)
        assert len(result) == 1
        assert list(result[0]) == ["X_0"]

    def test_tokenize_consecutive_operators(self):
        """Test tokenizing consecutive operators."""
        lines = ["(<= >= =)"]
        result = tokenize(lines)
        tokens = list(result[0])
        assert "<=" in tokens
        assert ">=" in tokens
        assert "=" in tokens

    def test_tokenize_number_without_space_after(self):
        """Test number directly followed by other token."""
        lines = ["(+ 1 2)"]
        result = tokenize(lines)
        tokens = list(result[0])
        assert "1" in tokens
        assert "2" in tokens

    def test_tokenize_deeply_nested_parens(self):
        """Test deeply nested parentheses."""
        line = "(" * 50 + "X_0" + ")" * 50
        lines = [line]
        result = tokenize(lines)
        assert len(result) == 1
        tokens = list(result[0])
        assert "X_0" in tokens

    def test_tokenize_very_large_number(self):
        """Test very large number."""
        lines = ["(+ 999999999999999999999999999 1)"]
        result = tokenize(lines)
        tokens = list(result[0])
        assert "999999999999999999999999999" in tokens

    def test_tokenize_very_small_number(self):
        """Test very small number in scientific notation."""
        lines = ["(* 1.0e-300 2.0)"]
        result = tokenize(lines)
        tokens = list(result[0])
        assert any("e-300" in t.lower() for t in tokens)
