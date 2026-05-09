"""Unit tests for AST tokenization.

Tests the public tokenize() API for converting line lists to token sequences.
Follows principles: broad code coverage, all logics, error handling, quick tests,
fine-grained information, public API only, no deselected tests.
"""

import pytest

from torchvnnlib.ast._tokenize import tokenize


class TestTokenizeBasic:
    """Test tokenize() with basic inputs."""

    def test_tokenize_single_line(self):
        """Test tokenizing a single line."""
        lines = ["(+ 1 2)"]
        result = tokenize(lines)
        assert len(result) == 1
        assert list(result[0]) == ["(", "+", "1", "2", ")"]

    def test_tokenize_multiple_lines(self):
        """Test tokenizing multiple lines."""
        lines = [
            "(declare-const X_0 Real)",
            "(assert (<= X_0 1.0))",
            "(assert (>= X_0 0.0))",
        ]
        result = tokenize(lines)
        assert len(result) == 3

    @pytest.mark.parametrize(
        "lines",
        [
            ["(+ 1 2)"],
            ["(<= X_0 1.0)"],
            ["(declare-const X_0 Real)"],
            ["(and (expr1) (expr2))"],
        ],
    )
    def test_tokenize_diverse_single_lines(self, lines):
        """Test tokenizing diverse single-line inputs."""
        result = tokenize(lines)
        assert len(result) >= 1
        assert all(tokens for tokens in result)

    def test_tokenize_empty_lines_filtered(self):
        """Test that purely empty lines are filtered, whitespace lines produce empty tokens."""
        lines = ["(+ 1 2)", "", "(- 3 1)", "   ", "\t"]
        result = tokenize(lines)
        # Purely empty strings are filtered, but whitespace-only strings produce empty deques
        assert len(result) == 4
        # First line is valid expression
        assert list(result[0]) == ["(", "+", "1", "2", ")"]
        # Third line is valid expression
        assert list(result[1]) == ["(", "-", "3", "1", ")"]
        # Whitespace lines produce empty token deques
        assert list(result[2]) == [""]
        assert list(result[3]) == [""]

    def test_tokenize_returns_deques(self):
        """Test that tokenize returns list of deques."""
        lines = ["(+ 1 2)"]
        result = tokenize(lines)
        assert isinstance(result, list)
        assert len(result) > 0
        assert hasattr(result[0], "__iter__")


class TestTokenizeNumbers:
    """Test tokenizing various number formats."""

    @pytest.mark.parametrize(
        "line",
        [
            "(+ 0 1)",
            "(+ 42 100)",
            "(+ 999999 1000000)",
        ],
    )
    def test_tokenize_integers(self, line):
        """Test tokenizing integer numbers."""
        result = tokenize([line])
        tokens = list(result[0])
        # Verify integers are in tokens
        assert any(t.isdigit() for t in tokens)

    @pytest.mark.parametrize(
        "line",
        [
            "(+ 0.0 1.5)",
            "(+ 3.14 2.71)",
            "(<= X_0 0.5)",
        ],
    )
    def test_tokenize_floats(self, line):
        """Test tokenizing floating point numbers."""
        result = tokenize([line])
        tokens = list(result[0])
        # Should contain at least one float token (containing a decimal point)
        assert any("." in t for t in tokens)

    @pytest.mark.parametrize(
        "line",
        [
            "(+ -1 -2)",
            "(- -5.5 -10)",
            "(<= X_0 -0.5)",
        ],
    )
    def test_tokenize_negative_numbers(self, line):
        """Test tokenizing negative numbers."""
        result = tokenize([line])
        tokens = list(result[0])
        # Should contain negative numbers
        assert any(
            t.startswith("-") and (t[1:].replace(".", "").isdigit() or "e" in t.lower())
            for t in tokens
        )

    @pytest.mark.parametrize(
        "line",
        [
            "(* 1.5e-10 X_0)",
            "(+ 1e308 1e-308)",
            "(* 2.5E+3 3.2e-5)",
        ],
    )
    def test_tokenize_scientific_notation(self, line):
        """Test tokenizing scientific notation."""
        result = tokenize([line])
        tokens = list(result[0])
        # Should contain scientific notation
        assert any("e" in t.lower() for t in tokens)


class TestTokenizeOperators:
    """Test tokenizing all operator types."""

    @pytest.mark.parametrize(
        ("operator", "line_template"),
        [
            ("+", "({} 1 2)"),
            ("-", "({} 1 2)"),
            ("*", "({} 1 2)"),
            ("/", "({} 1 2)"),
            ("<=", "({} X_0 1.0)"),
            (">=", "({} X_0 1.0)"),
            ("=", "({} X_0 1.0)"),
            ("<", "({} X_0 1.0)"),
            (">", "({} X_0 1.0)"),
        ],
    )
    def test_tokenize_operators(self, operator, line_template):
        """Test tokenizing operators: arithmetic and comparison (STR11: merged pair)."""
        line = line_template.format(operator)
        result = tokenize([line])
        tokens = list(result[0])
        assert operator in tokens

    @pytest.mark.parametrize(
        "keyword",
        ["and", "or", "declare-const", "assert", "Real"],
    )
    def test_tokenize_keywords_and_identifiers(self, keyword):
        """Test tokenizing keywords and identifiers."""
        line = f"({keyword} X_0 Y_0)"
        result = tokenize([line])
        tokens = list(result[0])
        assert keyword in tokens


class TestTokenizeVariables:
    """Test tokenizing variable names."""

    @pytest.mark.parametrize(
        "var_name",
        ["X_0", "X_1", "X_999", "Y_0", "Y_1", "Y_10000"],
    )
    def test_tokenize_variable_names(self, var_name):
        """Test tokenizing various variable names."""
        line = f"(assert {var_name})"
        result = tokenize([line])
        tokens = list(result[0])
        assert var_name in tokens

    def test_tokenize_mixed_variables(self):
        """Test tokenizing expression with multiple variables."""
        line = "(and X_0 X_1 Y_0 Y_1)"
        result = tokenize([line])
        tokens = list(result[0])
        assert "X_0" in tokens
        assert "X_1" in tokens
        assert "Y_0" in tokens
        assert "Y_1" in tokens


class TestTokenizeValidInput:
    """Test that valid inputs are tokenized correctly."""

    @pytest.mark.parametrize(
        ("line", "expected_keywords"),
        [
            ("(assert (and X_0 Y_1 Z_99))", ["X_0", "Y_1", "Z_99"]),
            (
                "(declare-const X Real assert and or <= >= = < > + - * /)",
                ["declare-const", "Real", "assert"],
            ),
        ],
    )
    def test_tokenize_identifiers_and_keywords(self, line, expected_keywords):
        """Test tokenizing identifiers and keywords (STR11: merged pair)."""
        result = tokenize([line])
        tokens = list(result[0])
        for kw in expected_keywords:
            assert kw in tokens


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

    def test_tokenize_complex_constraint(self):
        """Test tokenizing complex linear constraint."""
        line = "(assert (<= (+ (* 2.0 X_0) (* 3.0 X_1)) 5.0))"
        result = tokenize([line])
        tokens = list(result[0])
        # Verify key components are present
        assert "2.0" in tokens or "2" in tokens
        assert "3.0" in tokens or "3" in tokens
        assert "5.0" in tokens or "5" in tokens

    def test_tokenize_output_constraint(self):
        """Test tokenizing output constraint."""
        line = "(assert (<= (- Y_0 Y_1) 0.0))"
        result = tokenize([line])
        tokens = list(result[0])
        assert "Y_0" in tokens
        assert "Y_1" in tokens


class TestTokenizeParallelMode:
    """Test sequential vs parallel execution modes."""

    @pytest.mark.parametrize("use_parallel", [False, True])
    def test_tokenize_parallel_modes(self, use_parallel):
        """Test tokenize with parallel/sequential modes."""
        lines = ["(assert (<= X_0 1.0))", "(assert (>= X_1 0.0))"]
        result = tokenize(lines, use_parallel=use_parallel)
        assert len(result) == 2
        assert list(result[0]) == ["(", "assert", "(", "<=", "X_0", "1.0", ")", ")"]

    # [REVIEW] Deleted: test_tokenize_sequential_mode, test_tokenize_parallel_mode.
    # STR1: merged 2 of 7 group members into parametrized test_tokenize_parallel_modes.

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


class TestTokenizeVerboseMode:
    """Test verbose mode parameter."""

    @pytest.mark.parametrize("verbose", [True, False])
    def test_tokenize_accepts_verbose(self, verbose):
        """Test that tokenize accepts verbose parameter."""
        lines = ["(+ 1 2)"]
        result = tokenize(lines, verbose=verbose)
        assert len(result) == 1

    # [REVIEW] Deleted: test_tokenize_accepts_verbose_parameter,
    # test_tokenize_accepts_verbose_false_parameter. STR2: merged 2 MED_DUP into parametrized.
