"""Unit tests for AST flattening.

Tests the public flatten() API for converting nested AST structures to flattened form.
Follows principles: broad code coverage, all logics, error handling, quick tests,
fine-grained information, public API only, no deselected tests.

Note: flatten() requires And expressions with both input (X_*) and output (Y_*) variables.
"""

from torchvnnlib.ast._expr import Add, And, Cst, Geq, Leq, Mul, Or, Var
from torchvnnlib.ast._flatten import flatten


class TestFlattenBasic:
    """Test basic flattening functionality."""

    def test_flatten_simple_input_output(self):
        """Test flattening simple And with input and output."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_multiple_inputs_outputs(self):
        """Test flattening And with multiple inputs and outputs."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_1"), Cst(2.0)),
                Leq(Var("Y_0"), Cst(0.5)),
                Leq(Var("Y_1"), Cst(0.3)),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_with_arithmetic_in_constraints(self):
        """Test flattening And with arithmetic in constraints."""
        expr = And(
            [
                Leq(Add([Var("X_0"), Cst(1.0)]), Cst(2.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None


class TestFlattenLogicalOperators:
    """Test flattening of logical operators."""

    def test_flatten_and_with_multiple_constraints(self):
        """Test flattening AND with multiple input constraints."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_and_with_input_output_constraints(self):
        """Test flattening AND with multiple input and output constraints."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_1"), Cst(2.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_and_with_or_in_output(self):
        """Test flattening AND with OR in output constraints."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Or(
                    [
                        Leq(Var("Y_0"), Cst(0.3)),
                        Geq(Var("Y_0"), Cst(0.7)),
                    ]
                ),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_and_with_or_in_input(self):
        """Test flattening AND with OR in input constraints."""
        expr = And(
            [
                Or(
                    [
                        Leq(Var("X_0"), Cst(0.5)),
                        Geq(Var("X_0"), Cst(0.75)),
                    ]
                ),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None


class TestFlattenNestedLogical:
    """Test flattening of nested logical structures."""

    def test_flatten_or_in_constraint(self):
        """Test flattening AND with OR in constraint."""
        expr = And(
            [
                Or(
                    [
                        Leq(Var("X_0"), Cst(0.5)),
                        Geq(Var("X_0"), Cst(0.75)),
                    ]
                ),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_and_with_or_output_only(self):
        """Test flattening AND where output has only OR."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Or(
                    [
                        Leq(Var("Y_0"), Cst(0.3)),
                        Geq(Var("Y_0"), Cst(0.7)),
                    ]
                ),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_complex_or_structure(self):
        """Test flattening AND with complex OR structure."""
        expr = And(
            [
                Or(
                    [
                        Leq(Var("X_0"), Cst(0.5)),
                        Geq(Var("X_0"), Cst(0.75)),
                    ]
                ),
                Or(
                    [
                        Leq(Var("Y_0"), Cst(0.3)),
                        And([Geq(Var("Y_0"), Cst(0.7)), Leq(Var("Y_0"), Cst(0.9))]),
                    ]
                ),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_mixed_constraints(self):
        """Test flattening AND with mixed direct and OR constraints."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Or(
                    [
                        Leq(Var("X_1"), Cst(0.5)),
                        Geq(Var("X_1"), Cst(0.75)),
                    ]
                ),
                Leq(Var("Y_0"), Cst(0.5)),
                Leq(Var("Y_1"), Cst(0.3)),
            ]
        )
        result = flatten(expr)
        assert result is not None


class TestFlattenArithmeticInConstraints:
    """Test flattening arithmetic expressions within constraints."""

    def test_flatten_with_sum_in_input(self):
        """Test flattening And with addition in input constraint."""
        expr = And(
            [
                Leq(Add([Var("X_0"), Cst(1.0)]), Cst(2.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_with_product_in_input(self):
        """Test flattening And with multiplication in input constraint."""
        expr = And(
            [
                Leq(Mul(Cst(2.0), Var("X_0")), Cst(4.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_with_complex_arithmetic(self):
        """Test flattening And with complex arithmetic."""
        expr = And(
            [
                Leq(
                    Add([Mul(Cst(2.0), Var("X_0")), Mul(Cst(3.0), Var("X_1"))]),
                    Cst(5.0),
                ),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_with_arithmetic_in_output(self):
        """Test flattening And with arithmetic in output constraint."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Leq(Add([Var("Y_0"), Cst(1.0)]), Cst(2.0)),
            ]
        )
        result = flatten(expr)
        assert result is not None


class TestFlattenInputAndOutputVariables:
    """Test flattening with different variable types."""

    def test_flatten_multiple_input_variables(self):
        """Test flattening with multiple input variables."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Leq(Var("X_1"), Cst(1.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_multiple_output_variables(self):
        """Test flattening with multiple output variables."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Leq(Var("Y_0"), Cst(0.5)),
                Leq(Var("Y_1"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_many_mixed_variables(self):
        """Test flattening with many mixed input/output variables."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_1"), Cst(2.0)),
                Leq(Var("Y_0"), Cst(0.5)),
                Leq(Var("Y_1"), Cst(0.3)),
            ]
        )
        result = flatten(expr)
        assert result is not None


class TestFlattenResultStructure:
    """Test the structure of flattened results."""

    def test_flatten_preserves_properties(self):
        """Test that flatten preserves variable properties."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        # Result should preserve variable properties
        assert result is not None
        assert result.has_input_vars is True
        assert result.has_output_vars is True

    def test_flatten_with_complex_structure(self):
        """Test flatten with complex structure."""
        expr = And(
            [
                Or(
                    [
                        Leq(Var("X_0"), Cst(0.5)),
                        Geq(Var("X_0"), Cst(0.75)),
                    ]
                ),
                Or(
                    [
                        Leq(Var("Y_0"), Cst(0.3)),
                        And([Geq(Var("Y_0"), Cst(0.7)), Leq(Var("Y_0"), Cst(0.9))]),
                    ]
                ),
            ]
        )
        result = flatten(expr)
        assert result is not None
        assert result.has_input_vars is True
        assert result.has_output_vars is True


class TestFlattenEdgeCases:
    """Test edge cases in flattening."""

    def test_flatten_minimal_valid_structure(self):
        """Test flattening with minimal valid structure."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Leq(Var("Y_0"), Cst(0.5)),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_many_constraints(self):
        """Test flattening with many constraints."""
        constraints = [Leq(Var(f"X_{i}"), Cst(float(i + 1))) for i in range(3)]
        constraints.extend(Leq(Var(f"Y_{i}"), Cst(0.5 - i * 0.1)) for i in range(2))
        expr = And(constraints)
        result = flatten(expr)
        assert result is not None

    def test_flatten_with_or_in_both_constraints(self):
        """Test flattening with OR in both input and output."""
        expr = And(
            [
                Or(
                    [
                        Leq(Var("X_0"), Cst(0.3)),
                        Geq(Var("X_0"), Cst(0.7)),
                    ]
                ),
                Or(
                    [
                        Leq(Var("Y_0"), Cst(0.2)),
                        Geq(Var("Y_0"), Cst(0.8)),
                    ]
                ),
            ]
        )
        result = flatten(expr)
        assert result is not None

    def test_flatten_or_with_nested_and(self):
        """Test flattening OR with nested AND in output."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Or(
                    [
                        Leq(Var("Y_0"), Cst(0.2)),
                        And([Geq(Var("Y_0"), Cst(0.7)), Leq(Var("Y_0"), Cst(0.9))]),
                    ]
                ),
            ]
        )
        result = flatten(expr)
        assert result is not None
