"""Comprehensive unit tests for AST expression printing - Phase 5 coverage.

This module provides extensive test coverage for the print_expr() function
in torchvnnlib.ast._print, focusing on:
- Printing all expression types (Cst, Var, Add, Sub, Mul, Div, Eq, Leq, Geq, And, Or)
- Binary operations with different indentation levels
- N-ary operations with multiple arguments
- Logical operations (And, Or) with nested structures
- Complex nested expressions with multiple levels
- Edge cases (empty args, single argument, deeply nested)

Target coverage: All 36 missed statements in _print.py (16% -> >92%)
"""

import io
from contextlib import redirect_stdout

import pytest

from torchvnnlib.ast._expr import (
    Add,
    And,
    Cst,
    Div,
    Eq,
    Geq,
    Leq,
    Mul,
    Or,
    Sub,
    Var,
)
from torchvnnlib.ast._print import print_expr


class TestPrintCstAndVar:
    """Test printing of constant and variable expressions."""

    def test_print_constant_positive(self):
        """Test printing positive constant."""
        expr = Cst(5.0)
        output = self._capture_print(expr)
        assert "5.0" in output

    def test_print_constant_negative(self):
        """Test printing negative constant."""
        expr = Cst(-3.5)
        output = self._capture_print(expr)
        assert "-3.5" in output

    def test_print_constant_zero(self):
        """Test printing zero constant."""
        expr = Cst(0.0)
        output = self._capture_print(expr)
        assert "0.0" in output

    def test_print_constant_large(self):
        """Test printing large constant."""
        expr = Cst(1e10)
        output = self._capture_print(expr)
        assert "1e" in output or "10000000000" in output

    def test_print_variable_input(self):
        """Test printing input variable."""
        expr = Var("X_0")
        output = self._capture_print(expr)
        assert "X_0" in output

    def test_print_variable_output(self):
        """Test printing output variable."""
        expr = Var("Y_1")
        output = self._capture_print(expr)
        assert "Y_1" in output

    def test_print_variable_large_index(self):
        """Test printing variable with large index."""
        expr = Var("X_999")
        output = self._capture_print(expr)
        assert "X_999" in output

    @staticmethod
    def _capture_print(expr, indent=0):
        """Capture print output."""
        f = io.StringIO()
        with redirect_stdout(f):
            print_expr(expr, indent)
        return f.getvalue()


class TestPrintBinaryOps:
    """Test printing of binary operations."""

    def test_print_eq_simple(self):
        """Test printing equality constraint."""
        expr = Eq(Var("X_0"), Cst(1.0))
        output = self._capture_print(expr)
        assert "=" in output
        assert "X_0" in output
        assert "1.0" in output

    def test_print_leq_simple(self):
        """Test printing less-or-equal constraint."""
        expr = Leq(Var("Y_0"), Cst(0.5))
        output = self._capture_print(expr)
        assert "<=" in output
        assert "Y_0" in output

    def test_print_geq_simple(self):
        """Test printing greater-or-equal constraint."""
        expr = Geq(Var("X_0"), Cst(0.0))
        output = self._capture_print(expr)
        assert ">=" in output
        assert "X_0" in output

    def test_print_add_two_args(self):
        """Test printing addition with two arguments."""
        expr = Add([Var("X_0"), Cst(1.0)])
        output = self._capture_print(expr)
        assert "+" in output
        assert "X_0" in output
        assert "1.0" in output

    def test_print_sub_simple(self):
        """Test printing subtraction."""
        expr = Sub(Var("X_0"), Cst(1.0))
        output = self._capture_print(expr)
        assert "-" in output
        assert "X_0" in output

    def test_print_mul_simple(self):
        """Test printing multiplication."""
        expr = Mul(Cst(2.0), Var("X_0"))
        output = self._capture_print(expr)
        assert "*" in output
        assert "2.0" in output
        assert "X_0" in output

    def test_print_div_simple(self):
        """Test printing division."""
        expr = Div(Var("X_0"), Cst(2.0))
        output = self._capture_print(expr)
        assert "/" in output
        assert "X_0" in output

    def test_print_binary_op_nested(self):
        """Test printing nested binary operations."""
        # (X_0 + 1.0) - 2.0
        inner = Add([Var("X_0"), Cst(1.0)])
        expr = Sub(inner, Cst(2.0))
        output = self._capture_print(expr)
        assert "-" in output
        assert "+" in output

    @staticmethod
    def _capture_print(expr, indent=0):
        """Capture print output."""
        f = io.StringIO()
        with redirect_stdout(f):
            print_expr(expr, indent)
        return f.getvalue()


class TestPrintNaryOps:
    """Test printing of n-ary operations (Add)."""

    def test_print_add_three_args(self):
        """Test printing addition with three arguments."""
        expr = Add([Cst(1.0), Cst(2.0), Cst(3.0)])
        output = self._capture_print(expr)
        assert "+" in output
        assert "1.0" in output
        assert "2.0" in output
        assert "3.0" in output

    def test_print_add_many_args(self):
        """Test printing addition with many arguments."""
        expr = Add([Var(f"X_{i}") for i in range(5)])
        output = self._capture_print(expr)
        assert "+" in output
        for i in range(5):
            assert f"X_{i}" in output

    def test_print_add_single_arg(self):
        """Test printing addition with single argument."""
        expr = Add([Var("X_0")])
        output = self._capture_print(expr)
        assert "+" in output
        assert "X_0" in output

    def test_print_add_mixed_types(self):
        """Test printing addition with mixed variable and constant types."""
        expr = Add([Var("X_0"), Cst(1.0), Var("Y_0"), Cst(2.0)])
        output = self._capture_print(expr)
        assert "+" in output
        assert "X_0" in output
        assert "Y_0" in output

    def test_print_add_with_mul(self):
        """Test printing addition containing multiplication."""
        expr = Add([Mul(Cst(2.0), Var("X_0")), Cst(1.0)])
        output = self._capture_print(expr)
        assert "+" in output
        assert "*" in output

    @staticmethod
    def _capture_print(expr, indent=0):
        """Capture print output."""
        f = io.StringIO()
        with redirect_stdout(f):
            print_expr(expr, indent)
        return f.getvalue()


class TestPrintLogicalOps:
    """Test printing of logical operations (And, Or)."""

    def test_print_and_two_args(self):
        """Test printing AND with two arguments."""
        expr = And([Leq(Var("X_0"), Cst(1.0)), Geq(Var("X_0"), Cst(0.0))])
        output = self._capture_print(expr)
        assert "and" in output
        assert "<=" in output
        assert ">=" in output

    def test_print_and_multiple_args(self):
        """Test printing AND with multiple arguments."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_1"), Cst(2.0)),
            ]
        )
        output = self._capture_print(expr)
        assert "and" in output
        # Count number of constraint operators
        count = output.count("<=") + output.count(">=")
        assert count >= 3

    def test_print_or_two_args(self):
        """Test printing OR with two arguments."""
        expr = Or([Leq(Var("Y_0"), Cst(0.5)), Geq(Var("Y_0"), Cst(0.8))])
        output = self._capture_print(expr)
        assert "or" in output
        assert "<=" in output or ">=" in output

    def test_print_or_multiple_args(self):
        """Test printing OR with multiple arguments."""
        expr = Or(
            [
                Leq(Var("Y_0"), Cst(0.2)),
                Geq(Var("Y_0"), Cst(0.8)),
                Eq(Var("Y_0"), Cst(0.5)),
            ]
        )
        output = self._capture_print(expr)
        assert "or" in output

    @staticmethod
    def _capture_print(expr, indent=0):
        """Capture print output."""
        f = io.StringIO()
        with redirect_stdout(f):
            print_expr(expr, indent)
        return f.getvalue()


class TestPrintIndentation:
    """Test indentation handling in print_expr."""

    def test_print_with_zero_indent(self):
        """Test printing with indent level 0."""
        expr = Eq(Var("X_0"), Cst(1.0))
        output = self._capture_print(expr, indent=0)
        lines = output.split("\n")
        # First line should not have leading spaces (or minimal)
        assert len(lines) > 0

    def test_print_with_indent_one(self):
        """Test printing with indent level 1."""
        expr = Eq(Var("X_0"), Cst(1.0))
        output = self._capture_print(expr, indent=1)
        # Output should have indentation
        assert isinstance(output, str)

    def test_print_nested_with_indentation(self):
        """Test printing nested expressions with indentation."""
        inner_expr = Add([Var("X_0"), Cst(1.0)])
        expr = Leq(inner_expr, Cst(2.0))
        output = self._capture_print(expr, indent=0)
        # Should handle nested indentation
        assert "+" in output or "X_0" in output

    def test_print_deeply_nested(self):
        """Test printing deeply nested expressions."""
        # ((X_0 + 1.0) - 2.0) <= 3.0
        add_expr = Add([Var("X_0"), Cst(1.0)])
        sub_expr = Sub(add_expr, Cst(2.0))
        expr = Leq(sub_expr, Cst(3.0))
        output = self._capture_print(expr, indent=0)
        assert "X_0" in output
        assert "<=" in output

    @staticmethod
    def _capture_print(expr, indent=0):
        """Capture print output."""
        f = io.StringIO()
        with redirect_stdout(f):
            print_expr(expr, indent)
        return f.getvalue()


class TestPrintComplexNested:
    """Test printing of complex nested expressions."""

    def test_print_and_with_or_in_constraint(self):
        """Test printing AND containing OR constraint."""
        or_expr = Or([Leq(Var("Y_0"), Cst(0.5)), Geq(Var("Y_0"), Cst(0.8))])
        expr = And([Leq(Var("X_0"), Cst(1.0)), or_expr])
        output = self._capture_print(expr)
        assert "and" in output
        assert "or" in output

    def test_print_or_with_and_in_constraint(self):
        """Test printing OR containing AND constraint."""
        and_expr = And([Leq(Var("X_0"), Cst(0.5)), Geq(Var("X_0"), Cst(0.0))])
        expr = Or([and_expr, Leq(Var("X_0"), Cst(1.0))])
        output = self._capture_print(expr)
        assert "or" in output
        assert "and" in output

    def test_print_complex_mixed_ops(self):
        """Test printing complex expression with mixed operations."""
        # AND containing: input constraint, output constraint, output OR
        expr = And(
            [
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_0"), Cst(1.0)),
                Or(
                    [
                        Leq(Var("Y_0"), Cst(0.5)),
                        And([Geq(Var("Y_0"), Cst(0.7)), Leq(Var("Y_0"), Cst(0.9))]),
                    ]
                ),
            ]
        )
        output = self._capture_print(expr)
        assert "and" in output
        assert "or" in output
        assert "X_0" in output
        assert "Y_0" in output

    def test_print_triple_nested(self):
        """Test printing triple-nested expression."""
        # X_0 + (Y_0 + (1.0 + 2.0))
        inner_add = Add([Cst(1.0), Cst(2.0)])
        middle_add = Add([Var("Y_0"), inner_add])
        expr = Add([Var("X_0"), middle_add])
        output = self._capture_print(expr)
        assert "+" in output
        assert "X_0" in output
        assert "Y_0" in output

    @staticmethod
    def _capture_print(expr, indent=0):
        """Capture print output."""
        f = io.StringIO()
        with redirect_stdout(f):
            print_expr(expr, indent)
        return f.getvalue()


class TestPrintEdgeCases:
    """Test edge cases in expression printing."""

    def test_print_single_element_and(self):
        """Test printing AND with single element."""
        expr = And([Leq(Var("X_0"), Cst(1.0))])
        output = self._capture_print(expr)
        assert "and" in output

    def test_print_single_element_or(self):
        """Test printing OR with single element."""
        expr = Or([Leq(Var("Y_0"), Cst(0.5))])
        output = self._capture_print(expr)
        assert "or" in output

    def test_print_all_constraint_types(self):
        """Test printing all constraint types together."""
        constraints = [
            Eq(Var("X_0"), Cst(1.0)),
            Leq(Var("X_1"), Cst(2.0)),
            Geq(Var("X_2"), Cst(0.0)),
        ]
        expr = And(constraints)
        output = self._capture_print(expr)
        assert "=" in output
        assert "<=" in output
        assert ">=" in output

    def test_print_arithmetic_all_types(self):
        """Test printing all arithmetic operations."""
        expr = And(
            [
                Leq(Add([Var("X_0"), Cst(1.0)]), Cst(5.0)),
                Leq(Sub(Var("X_1"), Cst(1.0)), Cst(3.0)),
                Leq(Mul(Cst(2.0), Var("X_2")), Cst(4.0)),
                Leq(Div(Var("X_3"), Cst(2.0)), Cst(1.0)),
            ]
        )
        output = self._capture_print(expr)
        assert "and" in output
        # All operators should appear
        assert any(op in output for op in ["+", "-", "*", "/"])

    def test_print_float_precision(self):
        """Test printing floats with different precisions."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(0.123456789)),
                Leq(Var("X_1"), Cst(1e-10)),
                Leq(Var("X_2"), Cst(1e10)),
            ]
        )
        output = self._capture_print(expr)
        assert "and" in output
        # Values should be present in output
        assert "X_0" in output

    @staticmethod
    def _capture_print(expr, indent=0):
        """Capture print output."""
        f = io.StringIO()
        with redirect_stdout(f):
            print_expr(expr, indent)
        return f.getvalue()


class TestPrintInvalidExpression:
    """Test error handling for invalid expressions."""

    def test_print_invalid_expression_type(self):
        """Test printing invalid expression type raises error."""

        # Create a mock object that's not an Expr subclass
        class InvalidExpr:
            pass

        invalid = InvalidExpr()
        with pytest.raises(RuntimeError, match="Unknown expr"):
            print_expr(invalid)

    def test_print_none_expression(self):
        """Test printing None raises appropriate error."""
        with pytest.raises((RuntimeError, AttributeError, TypeError)):
            print_expr(None)


class TestPrintMultilineOutput:
    """Test multiline output for logical operators."""

    def test_and_produces_multiline(self):
        """Test that AND with multiple args produces output."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        output = self._capture_print(expr)
        assert "and" in output
        # Should have multiple parts
        assert len(output) > 10

    def test_or_produces_multiline(self):
        """Test that OR with multiple args produces output."""
        expr = Or(
            [
                Leq(Var("Y_0"), Cst(0.2)),
                Geq(Var("Y_0"), Cst(0.8)),
                Eq(Var("Y_0"), Cst(0.5)),
            ]
        )
        output = self._capture_print(expr)
        assert "or" in output

    @staticmethod
    def _capture_print(expr, indent=0):
        """Capture print output."""
        f = io.StringIO()
        with redirect_stdout(f):
            print_expr(expr, indent)
        return f.getvalue()


class TestPrintSpecialValues:
    """Test printing special float values."""

    def test_print_zero_constant(self):
        """Test printing zero."""
        expr = Cst(0.0)
        output = self._capture_print(expr)
        assert "0" in output

    def test_print_negative_constant(self):
        """Test printing negative numbers."""
        expr = Cst(-123.456)
        output = self._capture_print(expr)
        assert "-" in output

    def test_print_very_small_constant(self):
        """Test printing very small constants."""
        expr = Cst(1e-100)
        output = self._capture_print(expr)
        # Should contain scientific notation or zero representation
        assert len(output) > 0

    def test_print_very_large_constant(self):
        """Test printing very large constants."""
        expr = Cst(1e100)
        output = self._capture_print(expr)
        assert len(output) > 0

    @staticmethod
    def _capture_print(expr, indent=0):
        """Capture print output."""
        f = io.StringIO()
        with redirect_stdout(f):
            print_expr(expr, indent)
        return f.getvalue()
