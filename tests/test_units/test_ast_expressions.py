"""Unit tests for AST expression classes.

Tests all expression classes in torchvnnlib.ast._expr module including:
- Cst (Constant)
- Var (Variable)
- Binary operators (Add, Sub, Mul, Div)
- Comparison operators (Leq, Geq, Eq)
- Logical operators (And, Or)
"""

import pytest

from torchvnnlib.ast._expr import Add, And, Cst, Div, Eq, Geq, Leq, Mul, Or, Sub, Var


class TestCstConstant:
    """Test Cst (Constant) expression class."""

    def test_cst_creation_positive(self):
        """Test creating positive constant."""
        c = Cst(5.0)
        assert c.value == 5.0

    def test_cst_creation_negative(self):
        """Test creating negative constant."""
        c = Cst(-3.5)
        assert c.value == -3.5

    def test_cst_creation_zero(self):
        """Test creating zero constant."""
        c = Cst(0.0)
        assert c.value == 0.0

    def test_cst_creation_float(self):
        """Test creating float constant."""
        c = Cst(3.14159)
        assert c.value == pytest.approx(3.14159)

    def test_cst_equality(self):
        """Test equality comparison between constants."""
        c1 = Cst(5.0)
        c2 = Cst(5.0)
        assert c1 == c2

    def test_cst_inequality(self):
        """Test inequality between different constants."""
        c1 = Cst(5.0)
        c2 = Cst(6.0)
        assert c1 != c2

    def test_cst_hash(self):
        """Test hashing constant for dict/set usage."""
        c1 = Cst(5.0)
        c2 = Cst(5.0)
        d = {c1: "value"}
        assert d[c2] == "value"

    def test_cst_has_no_vars(self):
        """Test that constant has no input or output variables."""
        c = Cst(5.0)
        assert c.has_input_vars is False
        assert c.has_output_vars is False


class TestVarVariable:
    """Test Var (Variable) expression class."""

    def test_var_creation_input(self):
        """Test creating input variable."""
        v = Var("X_0")
        assert v.name == "X_0"

    def test_var_creation_output(self):
        """Test creating output variable."""
        v = Var("Y_0")
        assert v.name == "Y_0"

    def test_var_index_extraction(self):
        """Test extracting index from variable name."""
        v = Var("X_123")
        assert v.index == 123

    def test_var_index_extraction_large(self):
        """Test extracting large index from variable name."""
        v = Var("Y_9999")
        assert v.index == 9999

    def test_var_type_input(self):
        """Test variable type detection for input."""
        v = Var("X_5")
        assert v.var_type == "X"

    def test_var_type_output(self):
        """Test variable type detection for output."""
        v = Var("Y_5")
        assert v.var_type == "Y"

    def test_var_equality(self):
        """Test equality between variables."""
        v1 = Var("X_0")
        v2 = Var("X_0")
        assert v1 == v2

    def test_var_inequality_different_names(self):
        """Test inequality between different variable names."""
        v1 = Var("X_0")
        v2 = Var("X_1")
        assert v1 != v2

    def test_var_hash(self):
        """Test hashing variable for dict/set usage."""
        v1 = Var("X_0")
        v2 = Var("X_0")
        s = {v1}
        assert v2 in s

    def test_var_has_input_vars_true(self):
        """Test that input variable has input_vars."""
        v = Var("X_5")
        assert v.has_input_vars is True
        assert v.has_output_vars is False

    def test_var_has_output_vars_true(self):
        """Test that output variable has output_vars."""
        v = Var("Y_5")
        assert v.has_output_vars is True
        assert v.has_input_vars is False

    def test_var_repr(self):
        """Test string representation of variable."""
        v = Var("X_42")
        assert "X_42" in repr(v)


class TestBinaryOperators:
    """Test binary operator expressions."""

    def test_add_two_constants(self):
        """Test adding two constants."""
        expr = Add([Cst(2.0), Cst(3.0)])
        assert len(expr.args) == 2

    def test_add_var_and_constant(self):
        """Test adding variable and constant."""
        expr = Add([Var("X_0"), Cst(1.0)])
        assert expr.has_input_vars is True
        assert expr.has_output_vars is False

    def test_add_two_vars(self):
        """Test adding two variables."""
        expr = Add([Var("X_0"), Var("X_1")])
        assert expr.has_input_vars is True

    def test_add_nary(self):
        """Test Add with multiple arguments."""
        expr = Add([Cst(1.0), Cst(2.0), Cst(3.0)])
        assert len(expr.args) == 3

    def test_add_has_input_vars_propagation(self):
        """Test variable propagation in Add."""
        expr = Add([Var("X_0"), Cst(1.0)])
        assert expr.has_input_vars is True

    def test_sub_basic(self):
        """Test subtraction operation."""
        expr = Sub(Var("X_0"), Cst(1.0))
        assert expr.has_input_vars is True

    def test_sub_two_vars(self):
        """Test subtracting two variables."""
        expr = Sub(Var("Y_0"), Var("Y_1"))
        assert expr.has_output_vars is True

    def test_mul_constant_and_var(self):
        """Test multiplication with constant and variable."""
        expr = Mul(Cst(2.0), Var("X_0"))
        assert expr.has_input_vars is True

    def test_mul_two_vars(self):
        """Test multiplying two variables."""
        expr = Mul(Var("X_0"), Var("X_1"))
        assert expr.has_input_vars is True

    def test_div_basic(self):
        """Test division operation."""
        expr = Div(Var("X_0"), Cst(2.0))
        assert expr.has_input_vars is True

    def test_div_var_by_var(self):
        """Test dividing variable by variable."""
        expr = Div(Var("Y_0"), Var("Y_1"))
        assert expr.has_output_vars is True

    def test_nested_arithmetic(self):
        """Test nested arithmetic operations."""
        inner = Add([Var("X_0"), Cst(1.0)])
        outer = Mul(Cst(2.0), inner)
        assert outer.has_input_vars is True

    def test_complex_nested_arithmetic(self):
        """Test complex nested arithmetic."""
        e1 = Mul(Cst(2.0), Var("X_0"))
        e2 = Mul(Cst(3.0), Var("X_1"))
        expr = Add([e1, e2])
        assert expr.has_input_vars is True


class TestComparisonOperators:
    """Test comparison operator expressions."""

    def test_leq_basic(self):
        """Test less than or equal constraint."""
        expr = Leq(Var("X_0"), Cst(1.0))
        assert expr.has_input_vars is True

    def test_leq_output_vars(self):
        """Test Leq with output variables."""
        expr = Leq(Var("Y_0"), Cst(0.5))
        assert expr.has_output_vars is True

    def test_geq_basic(self):
        """Test greater than or equal constraint."""
        expr = Geq(Var("X_0"), Cst(0.0))
        assert expr.has_input_vars is True

    def test_geq_two_vars(self):
        """Test Geq with two variables."""
        expr = Geq(Var("Y_0"), Var("Y_1"))
        assert expr.has_output_vars is True

    def test_eq_basic(self):
        """Test equality constraint."""
        expr = Eq(Var("X_0"), Cst(0.5))
        assert expr.has_input_vars is True

    def test_eq_two_vars(self):
        """Test equality between two variables."""
        expr = Eq(Var("Y_0"), Var("Y_1"))
        assert expr.has_output_vars is True

    def test_eq_two_outputs(self):
        """Test equality between output variables."""
        expr = Eq(Var("Y_0"), Var("Y_1"))
        assert expr.has_output_vars is True
        assert expr.has_input_vars is False

    def test_leq_complex_expr(self):
        """Test Leq with complex left side."""
        left = Add([Mul(Cst(2.0), Var("X_0")), Var("X_1")])
        expr = Leq(left, Cst(5.0))
        assert expr.has_input_vars is True

    def test_geq_complex_expr(self):
        """Test Geq with complex expression."""
        left = Sub(Var("Y_0"), Var("Y_1"))
        expr = Geq(left, Cst(0.0))
        assert expr.has_output_vars is True

    def test_eq_output_constraint(self):
        """Test Eq for output constraint."""
        expr = Eq(Sub(Var("Y_0"), Var("Y_1")), Cst(0.0))
        assert expr.has_output_vars is True


class TestLogicalOperators:
    """Test logical operator expressions (And, Or)."""

    def test_and_basic(self):
        """Test And with two constraints."""
        expr = And([Leq(Var("X_0"), Cst(1.0)), Geq(Var("X_0"), Cst(0.0))])
        assert expr.has_input_vars is True

    def test_and_nary(self):
        """Test And with multiple constraints."""
        expr = And(
            [
                Leq(Var("X_0"), Cst(1.0)),
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_1"), Cst(1.0)),
            ]
        )
        assert len(expr.args) == 3
        assert expr.has_input_vars is True

    def test_or_basic(self):
        """Test Or with two constraints."""
        expr = Or([Leq(Var("X_0"), Cst(0.5)), Geq(Var("X_0"), Cst(0.75))])
        assert expr.has_input_vars is True

    def test_or_nary(self):
        """Test Or with multiple constraints."""
        expr = Or(
            [
                Leq(Var("X_0"), Cst(0.3)),
                Leq(Var("X_0"), Cst(0.7)),
                Leq(Var("X_0"), Cst(1.0)),
            ]
        )
        assert len(expr.args) == 3

    def test_and_output_vars(self):
        """Test And with output variables."""
        expr = And([Leq(Var("Y_0"), Cst(0.5)), Geq(Var("Y_0"), Cst(0.0))])
        assert expr.has_output_vars is True
        assert expr.has_input_vars is False

    def test_nested_and(self):
        """Test nested And expressions."""
        inner = And([Leq(Var("X_0"), Cst(1.0)), Geq(Var("X_0"), Cst(0.0))])
        outer = And([inner, Leq(Var("X_1"), Cst(1.0))])
        assert outer.has_input_vars is True

    def test_nested_or(self):
        """Test nested Or expressions."""
        inner = Or([Leq(Var("X_0"), Cst(0.3)), Geq(Var("X_0"), Cst(0.7))])
        outer = Or([inner, Leq(Var("X_1"), Cst(0.5))])
        assert outer.has_input_vars is True

    def test_mixed_and_or(self):
        """Test mixed And/Or nesting."""
        and_expr = And([Leq(Var("X_0"), Cst(1.0)), Geq(Var("X_0"), Cst(0.0))])
        or_expr = Or([and_expr, Leq(Var("X_1"), Cst(0.5))])
        assert or_expr.has_input_vars is True

    def test_and_iteration(self):
        """Test iterating over And operands."""
        expr = And([Cst(1.0), Cst(2.0), Cst(3.0)])
        operands = list(expr)
        assert len(operands) == 3

    def test_or_iteration(self):
        """Test iterating over Or operands."""
        expr = Or([Leq(Var("X_0"), Cst(0.5)), Geq(Var("X_0"), Cst(0.75))])
        operands = list(expr)
        assert len(operands) == 2


class TestEdgeCasesAndProperties:
    """Test edge cases and expression properties."""

    def test_deeply_nested_expressions(self):
        """Test deeply nested expression structure."""
        expr = Cst(1.0)
        for _ in range(5):
            expr = Add([expr, Cst(1.0)])
        assert expr.has_input_vars is False
        assert expr.has_output_vars is False

    def test_mixed_input_output_vars(self):
        """Test expression with both input and output variables."""
        expr = Add([Var("X_0"), Var("Y_0")])
        assert expr.has_input_vars is True
        assert expr.has_output_vars is True

    def test_hash_consistency(self):
        """Test that equal expressions have equal hashes."""
        v1 = Var("X_0")
        v2 = Var("X_0")
        assert hash(v1) == hash(v2)

    def test_equality_transitive(self):
        """Test transitivity of equality."""
        c1 = Cst(5.0)
        c2 = Cst(5.0)
        c3 = Cst(5.0)
        assert c1 == c2
        assert c2 == c3
        assert c1 == c3

    def test_var_repr_contains_name(self):
        """Test that variable repr contains its name."""
        v = Var("X_42")
        assert "X_42" in repr(v) or "42" in repr(v)

    def test_cst_repr_contains_value(self):
        """Test that constant repr contains its value."""
        c = Cst(3.14)
        repr_str = repr(c)
        assert "3.14" in repr_str or "3" in repr_str

    def test_single_element_nary_add(self):
        """Test Add with single element."""
        expr = Add([Var("X_0")])
        assert len(expr.args) == 1

    def test_single_element_nary_and(self):
        """Test And with single element."""
        expr = And([Leq(Var("X_0"), Cst(1.0))])
        assert len(expr.args) == 1

    def test_variable_index_zero(self):
        """Test variable with index zero."""
        v = Var("X_0")
        assert v.index == 0

    def test_cst_scientific_notation(self):
        """Test constant with very small value."""
        c = Cst(1.5e-10)
        assert c.value == pytest.approx(1.5e-10)
