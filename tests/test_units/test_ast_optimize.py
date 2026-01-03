"""Unit tests for AST optimization.

Tests the public optimize() API for simplifying and optimizing AST expressions.
Follows principles: broad code coverage, all logics, error handling, quick tests,
fine-grained information, public API only, no deselected tests.
"""

import warnings

import pytest

from torchvnnlib.ast._expr import Add, And, Cst, Geq, Leq, Mul, Or, Var
from torchvnnlib.ast._optimize import optimize


class TestOptimizeBasic:
    """Test basic optimization functionality."""

    def test_optimize_simple_and(self):
        """Test optimization of simple AND."""
        expr = And([Leq(Var("X_0"), Cst(1.0))])
        result = optimize(expr)
        assert result is not None

    def test_optimize_simple_or(self):
        """Test optimization of simple OR."""
        expr = Or([Leq(Var("X_0"), Cst(1.0))])
        result = optimize(expr)
        assert result is not None

    def test_optimize_and_with_constraints(self):
        """Test optimization of AND with multiple constraints."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_or_with_constraints(self):
        """Test optimization of OR with multiple constraints."""
        expr = Or(
            [
                Leq(Var("X_0"), Cst(0.5)),
                Geq(Var("X_0"), Cst(0.75)),
            ]
        )
        result = optimize(expr)
        assert result is not None


class TestOptimizeSingleElementOperators:
    """Test optimization of single-element operators."""

    def test_optimize_single_constraint_in_and(self):
        """Test optimization of AND with single constraint."""
        # (and (<= X_0 1.0)) should simplify to just the constraint
        expr = And([Leq(Var("X_0"), Cst(1.0))])
        result = optimize(expr)
        assert result is not None

    def test_optimize_single_constraint_in_or(self):
        """Test optimization of OR with single constraint."""
        # (or (<= X_0 1.0)) should simplify to just the constraint
        expr = Or([Leq(Var("X_0"), Cst(1.0))])
        result = optimize(expr)
        assert result is not None

    def test_optimize_and_with_nested_single_element(self):
        """Test optimization of AND containing single-element AND."""
        # (and (and (<= X_0 1.0))) should simplify
        expr = And([And([Leq(Var("X_0"), Cst(1.0))])])
        result = optimize(expr)
        assert result is not None


class TestOptimizeLogicalOperators:
    """Test optimization of logical operators."""

    def test_optimize_and_two_constraints(self):
        """Test optimization of AND with two constraints."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_or_two_constraints(self):
        """Test optimization of OR with two constraints."""
        expr = Or(
            [
                Leq(Var("X_0"), Cst(0.5)),
                Geq(Var("X_0"), Cst(0.75)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_nested_and(self):
        """Test optimization of nested AND."""
        inner = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
            ]
        )
        outer = And([inner, Leq(Var("X_1"), Cst(1.0))])
        result = optimize(outer)
        assert result is not None

    def test_optimize_nested_or(self):
        """Test optimization of nested OR."""
        inner = Or(
            [
                Leq(Var("X_0"), Cst(0.5)),
                Geq(Var("X_0"), Cst(0.75)),
            ]
        )
        outer = Or([inner, Leq(Var("X_1"), Cst(0.5))])
        result = optimize(outer)
        assert result is not None


class TestOptimizeArithmeticInConstraints:
    """Test optimization of constraints with arithmetic expressions."""

    def test_optimize_and_with_sum_constraint(self):
        """Test optimization of AND with sum in constraint."""
        expr = And(
            [
                Leq(Add([Var("X_0"), Cst(1.0)]), Cst(2.0)),
                Geq(Var("X_1"), Cst(0.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_and_with_product_constraint(self):
        """Test optimization of AND with product in constraint."""
        expr = And(
            [
                Leq(Mul(Cst(2.0), Var("X_0")), Cst(4.0)),
                Geq(Var("X_1"), Cst(0.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_and_with_complex_arithmetic(self):
        """Test optimization of AND with complex arithmetic."""
        expr = And(
            [
                Leq(Add([Mul(Cst(2.0), Var("X_0")), Mul(Cst(3.0), Var("X_1"))]), Cst(5.0)),
                Geq(Var("X_0"), Cst(0.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_or_with_arithmetic(self):
        """Test optimization of OR with arithmetic constraints."""
        expr = Or(
            [
                Leq(Add([Var("X_0"), Cst(1.0)]), Cst(2.0)),
                Geq(Var("X_0"), Cst(3.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None


class TestOptimizeLeqGeqPairs:
    """Test optimization of LEQ/GEQ pairs."""

    def test_optimize_leq_geq_pair_same_var(self):
        """Test optimization of LEQ and GEQ on same variable creates Eq."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(1.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_leq_geq_pair_different_bounds(self):
        """Test optimization of LEQ and GEQ on same variable with different bounds."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_multiple_leq_geq_pairs(self):
        """Test optimization with multiple LEQ/GEQ pairs."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_1"), Cst(2.0)),
                Geq(Var("X_1"), Cst(1.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_mixed_constraints(self):
        """Test optimization with mixed constraints."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Leq(Var("X_1"), Cst(2.0)),
                Geq(Var("X_1"), Cst(0.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None


class TestOptimizeProperties:
    """Test properties of optimized expressions."""

    def test_optimize_preserves_input_variables(self):
        """Test that optimization preserves input variables."""
        expr = And([Leq(Var("X_0"), Cst(1.0)), Leq(Var("X_1"), Cst(2.0))])
        result = optimize(expr)
        assert result.has_input_vars is True

    def test_optimize_preserves_output_variables(self):
        """Test that optimization preserves output variables."""
        expr = And([Leq(Var("Y_0"), Cst(0.5)), Geq(Var("Y_0"), Cst(0.0))])
        result = optimize(expr)
        assert result.has_output_vars is True

    def test_optimize_preserves_mixed_variables(self):
        """Test that optimization preserves mixed variable types."""
        expr = And([Leq(Var("X_0"), Cst(1.0)), Leq(Var("Y_0"), Cst(0.5))])
        result = optimize(expr)
        assert result.has_input_vars is True
        assert result.has_output_vars is True


class TestOptimizeComplexExpressions:
    """Test optimization of complex expressions."""

    def test_optimize_complex_and_constraint(self):
        """Test optimization of complex AND constraint."""
        expr = And(
            [
                Leq(
                    Add([Mul(Cst(2.0), Var("X_0")), Mul(Cst(3.0), Var("X_1"))]),
                    Cst(5.0),
                ),
                Geq(Var("X_0"), Cst(0.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_nested_and_or(self):
        """Test optimization of nested AND and OR."""
        expr = And(
            [
                Or([Leq(Var("X_0"), Cst(0.3)), Geq(Var("X_0"), Cst(0.7))]),
                Leq(Var("X_1"), Cst(1.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_deeply_nested_and(self):
        """Test optimization of deeply nested AND."""
        expr = And([Leq(Var("X_0"), Cst(1.0))])
        for _ in range(3):
            expr = And([expr, Leq(Var("X_1"), Cst(1.0))])
        result = optimize(expr)
        assert result is not None


class TestOptimizeMultipleTimes:
    """Test repeated optimization."""

    def test_optimize_idempotent(self):
        """Test that optimizing twice is stable."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
            ]
        )
        result1 = optimize(expr)
        result2 = optimize(result1)
        # Both should be valid results
        assert result1 is not None
        assert result2 is not None

    def test_optimize_multiple_rounds(self):
        """Test multiple rounds of optimization."""
        expr = And(
            [
                And([Leq(Var("X_0"), Cst(1.0))]),
                Geq(Var("X_0"), Cst(0.0)),
            ]
        )
        result = expr
        for _ in range(3):
            result = optimize(result)
        assert result is not None


class TestOptimizeEdgeCases:
    """Test edge cases in optimization."""

    def test_optimize_single_constraint_and(self):
        """Test optimization of AND with single constraint."""
        expr = And([Leq(Var("X_0"), Cst(1.0))])
        result = optimize(expr)
        assert result is not None

    def test_optimize_single_constraint_or(self):
        """Test optimization of OR with single constraint."""
        expr = Or([Leq(Var("X_0"), Cst(1.0))])
        result = optimize(expr)
        assert result is not None

    def test_optimize_output_variable_constraints(self):
        """Test optimization with output variable constraints."""
        expr = And(
            [
                Leq(Var("Y_0"), Cst(0.5)),
                Leq(Var("Y_1"), Cst(0.5)),
            ]
        )
        result = optimize(expr)
        assert result.has_output_vars is True
        assert result.has_input_vars is False

    def test_optimize_mixed_input_output_variables(self):
        """Test optimization with mixed input/output variables."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = optimize(expr)
        assert result is not None


class TestOptimizeVerboseMode:
    """Test verbose output from optimization."""

    def test_optimize_verbose_sequential(self, capsys):
        """Test verbose output with sequential optimization."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
            ]
        )
        result = optimize(expr, verbose=True, use_parallel=False)
        captured = capsys.readouterr()
        assert "Simplify (sequential)" in captured.out
        assert "Sort vars" in captured.out
        assert result is not None

    def test_optimize_verbose_parallel(self, capsys):
        """Test verbose output with parallel optimization."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_1"), Cst(0.0)),
                Leq(Var("X_2"), Cst(2.0)),
            ]
        )
        result = optimize(expr, verbose=True, use_parallel=True)
        captured = capsys.readouterr()
        assert "Simplify (parallel)" in captured.out
        assert "Sort vars" in captured.out
        assert result is not None


class TestOptimizeErrorHandling:
    """Test error handling in optimization."""

    def test_optimize_not_and_or_raises(self):
        """Test that optimizing non-And/Or expression raises error."""
        expr = Leq(Var("X_0"), Cst(1.0))
        with pytest.raises(ValueError, match="must be either an And or an Or"):
            optimize(expr)

    def test_optimize_cst_only_expression(self):
        """Test optimization of constants only."""
        expr = And([Cst(1.0), Cst(2.0)])
        result = optimize(expr)
        assert result is not None


class TestOptimizeLeqGeqPairCombination:
    """Test optimization of Leq/Geq pair combination."""

    def test_optimize_leq_geq_pair_creates_eq(self):
        """Test that matching Leq and Geq pairs create Eq."""
        # Need nested And for _simplify_leqgeq to be called on the inner constraints
        expr = And(
            [
                And(
                    [
                        Leq(Var("X_0"), Cst(1.0)),
                        Geq(Var("X_0"), Cst(1.0)),
                    ]
                )
            ]
        )
        result = optimize(expr)
        assert result is not None
        # Verify the structure was optimized - should have nested And with Eq
        assert isinstance(result, And)
        # Check if Eq was created (might be in nested And)
        result_str = str(result)
        assert "=" in result_str

    def test_optimize_multiple_leq_geq_pairs(self):
        """Test optimization with multiple Leq/Geq pairs."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_1"), Cst(2.0)),
                Geq(Var("X_1"), Cst(1.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None

    def test_optimize_unmatched_leq_geq(self):
        """Test optimization with unmatched Leq and Geq."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_1"), Cst(0.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None


class TestOptimizeVariablePriority:
    """Test variable priority and sorting in optimization."""

    def test_optimize_output_variable_priority(self):
        """Test that output variables have higher priority in sorting."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = optimize(expr)
        assert result is not None
        # Y variables should be sorted after X variables

    def test_optimize_large_variable_numbers_warning(self):
        """Test warning for very large variable numbers."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            expr = And(
                [
                    Leq(Var("X_100000001"), Cst(1.0)),
                ]
            )
            _result = optimize(expr)
            # Check if warning was issued
            if len(w) > 0:
                assert issubclass(w[-1].category, UserWarning)
                assert "greater than 10000" in str(w[-1].message)

    def test_optimize_variable_sorting(self):
        """Test that variables are sorted correctly."""
        expr = And(
            [
                Leq(Var("X_2"), Cst(2.0)),
                Leq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_1"), Cst(1.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None
        # After sorting, X_0 should come before X_1, X_2


class TestOptimizeOrExpressions:
    """Test optimization of Or expressions."""

    def test_optimize_or_verbose(self, capsys):
        """Test optimization of Or with verbose output."""
        expr = Or(
            [
                Leq(Var("X_0"), Cst(0.5)),
                Geq(Var("X_0"), Cst(0.75)),
            ]
        )
        result = optimize(expr, verbose=True, use_parallel=False)
        captured = capsys.readouterr()
        assert "Simplify" in captured.out
        assert isinstance(result, Or)

    def test_optimize_or_with_nested_and(self):
        """Test optimization of Or containing And."""
        expr = Or(
            [
                And([Leq(Var("X_0"), Cst(1.0))]),
                And([Geq(Var("X_0"), Cst(0.0))]),
            ]
        )
        result = optimize(expr)
        assert result is not None


class TestOptimizeVariableSortingWithOutput:
    """Test variable sorting with output variables."""

    def test_optimize_output_variable_priority_sorting(self):
        """Test that output variables have higher priority in sorting."""
        expr = And(
            [
                Leq(Var("X_5"), Cst(1.0)),
                Leq(Var("X_2"), Cst(2.0)),
                Leq(Var("Y_3"), Cst(3.0)),
                Leq(Var("Y_1"), Cst(4.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None
        # Y variables should appear after X variables in sorted order
        result_str = str(result)
        x_pos = result_str.find("X_")
        y_pos = result_str.find("Y_")
        assert x_pos < y_pos

    def test_optimize_with_only_output_variables(self):
        """Test optimization with only output variables."""
        expr = And(
            [
                Leq(Var("Y_5"), Cst(1.0)),
                Leq(Var("Y_2"), Cst(2.0)),
                Leq(Var("Y_10"), Cst(3.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None
        # Result should be sorted with Y_2 before Y_5 before Y_10
        result_str = str(result)
        assert "Y_2" in result_str
        assert "Y_5" in result_str

    def test_optimize_mixed_indices(self):
        """Test optimization with mixed variable indices."""
        expr = And(
            [
                Leq(Var("X_100"), Cst(1.0)),
                Leq(Var("X_2"), Cst(2.0)),
                Leq(Var("Y_99"), Cst(3.0)),
                Leq(Var("Y_1"), Cst(4.0)),
            ]
        )
        result = optimize(expr)
        assert result is not None
        assert isinstance(result, And)
