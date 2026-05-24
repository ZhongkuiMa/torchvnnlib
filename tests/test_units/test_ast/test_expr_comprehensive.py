"""Comprehensive unit tests for AST expression classes - Phase 4 coverage.

This module provides extensive test coverage for all expression classes
in torchvnnlib.ast._expr module including:
- Cst (Constant) expressions
- Var (Variable) expressions
- BinaryOp classes: Add, Sub, Mul, Div, Eq, Leq, Geq
- NaryOp classes: And, Or
- Expression equality and hashing
- Property caching (has_input_vars, has_output_vars)
- Complex nested expressions
- Edge cases (zero, negative, division, etc.)

Target coverage: All 47 missed statements in _expr.py (70% -> >92%)
"""

from typing import cast

import pytest

from torchvnnlib.ast._expr import (
    Add,
    And,
    BinaryOp,
    Cst,
    Div,
    Eq,
    Expr,
    Geq,
    Leq,
    Mul,
    NaryOp,
    Or,
    Sub,
    Var,
)


class TestCstConstantComprehensive:
    """Comprehensive tests for Cst (Constant) expression class."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(5.0, id="positive"),
            pytest.param(-3.5, id="negative"),
            pytest.param(0.0, id="zero"),
            pytest.param(3.14159, id="float"),
            pytest.param(1e10, id="large"),
            pytest.param(1e-10, id="small"),
        ],
    )
    def test_cst_creation(self, value):
        """Test creating constants with various values."""
        c = Cst(value)
        assert c.value == pytest.approx(value)

    def test_cst_equality_same_value(self):
        """Test equality comparison between constants with same value."""
        c1 = Cst(5.0)
        c2 = Cst(5.0)
        assert c1 == c2

    def test_cst_equality_different_value(self):
        """Test equality comparison between constants with different values."""
        c1 = Cst(5.0)
        c2 = Cst(6.0)
        assert c1 != c2

    def test_cst_equality_with_non_cst(self):
        """Test inequality with non-Cst objects."""
        c = Cst(5.0)
        assert c != 5.0
        assert c != "5.0"
        assert c != Var("X_0")

    def test_cst_hash_consistency(self):
        """Test that equal constants have equal hashes."""
        c1 = Cst(5.0)
        c2 = Cst(5.0)
        assert hash(c1) == hash(c2)

    def test_cst_hash_in_dict(self):
        """Test using constant as dictionary key."""
        c1 = Cst(5.0)
        c2 = Cst(5.0)
        d = {c1: "value"}
        assert d[c2] == "value"

    def test_cst_hash_in_set(self):
        """Test using constant in set."""
        c1 = Cst(5.0)
        c2 = Cst(5.0)
        s = {c1}
        assert c2 in s

    def test_cst_has_no_input_vars(self):
        """Test that constant has no input variables."""
        c = Cst(5.0)
        assert c.has_input_vars is False

    def test_cst_has_no_output_vars(self):
        """Test that constant has no output variables."""
        c = Cst(5.0)
        assert c.has_output_vars is False

    def test_cst_repr(self):
        """Test string representation of constant."""
        c = Cst(3.14)
        repr_str = repr(c)
        assert "3.14" in repr_str or "3" in repr_str

    @pytest.mark.parametrize(
        "prop_name",
        [
            pytest.param("has_input_vars", id="input"),
            pytest.param("has_output_vars", id="output"),
        ],
    )
    def test_cst_caching(self, prop_name):
        """Test that has_input_vars / has_output_vars is cached (STR2: merged pair)."""
        c = Cst(5.0)
        result1 = getattr(c, prop_name)
        result2 = getattr(c, prop_name)
        assert result1 is False
        assert result1 is result2
        internal = "_" + prop_name
        assert getattr(c, internal) is False

    def test_cst_repr_zero(self):
        """Test repr of zero constant."""
        c = Cst(0.0)
        assert repr(c) == "0.0"

    def test_cst_repr_negative(self):
        """Test repr of negative constant."""
        c = Cst(-5.5)
        assert repr(c) == "-5.5"


class TestVarVariableComprehensive:
    """Comprehensive tests for Var (Variable) expression class."""

    @pytest.mark.parametrize(
        ("name", "expected_index"),
        [
            pytest.param("X_0", 0, id="input"),
            pytest.param("Y_0", 0, id="output"),
            pytest.param("X_9999", 9999, id="large_index"),
        ],
    )
    def test_var_creation(self, name, expected_index):
        """Test creating variables with various names and indices."""
        v = Var(name)
        assert v.name == name
        assert v.index == expected_index

    def test_var_invalid_name_no_underscore(self):
        """Test that variable without underscore raises ValueError."""
        with pytest.raises(ValueError, match="Variable name must start with"):
            Var("X0")

    def test_var_invalid_name_wrong_prefix(self):
        """Test that variable with wrong prefix raises ValueError."""
        with pytest.raises(ValueError, match="Variable name must start with"):
            Var("Z_0")

    def test_var_invalid_name_no_digits(self):
        """Test that variable without digits after underscore raises ValueError."""
        with pytest.raises(ValueError, match="Variable name must start with"):
            Var("X_A")

    def test_var_index_extraction(self):
        """Test extracting index from variable name."""
        v = Var("X_123")
        assert v.index == 123

    def test_var_index_extraction_zero(self):
        """Test extracting index zero from variable name."""
        v = Var("X_0")
        assert v.index == 0

    def test_var_type_input(self):
        """Test variable type detection for input."""
        v = Var("X_5")
        assert v.var_type == "X"

    def test_var_type_output(self):
        """Test variable type detection for output."""
        v = Var("Y_5")
        assert v.var_type == "Y"

    def test_var_equality_same_name(self):
        """Test equality between variables with same name."""
        v1 = Var("X_0")
        v2 = Var("X_0")
        assert v1 == v2

    def test_var_inequality_different_names(self):
        """Test inequality between variables with different names."""
        v1 = Var("X_0")
        v2 = Var("X_1")
        assert v1 != v2

    def test_var_inequality_different_types(self):
        """Test inequality between input and output variables."""
        v1 = Var("X_0")
        v2 = Var("Y_0")
        assert v1 != v2

    def test_var_inequality_with_non_var(self):
        """Test inequality with non-Var objects."""
        v = Var("X_0")
        assert v != "X_0"
        assert v != 0
        assert v != Cst(0.0)

    def test_var_hash_consistency(self):
        """Test that equal variables have equal hashes."""
        v1 = Var("X_0")
        v2 = Var("X_0")
        assert hash(v1) == hash(v2)

    def test_var_hash_in_set(self):
        """Test using variable in set."""
        v1 = Var("X_0")
        v2 = Var("X_0")
        s = {v1}
        assert v2 in s

    def test_var_hash_in_dict(self):
        """Test using variable as dictionary key."""
        v1 = Var("X_0")
        v2 = Var("X_0")
        d = {v1: "value"}
        assert d[v2] == "value"

    def test_var_has_input_vars_for_input(self):
        """Test that input variable has input_vars."""
        v = Var("X_5")
        assert v.has_input_vars is True

    def test_var_has_no_output_vars_for_input(self):
        """Test that input variable has no output_vars."""
        v = Var("X_5")
        assert v.has_output_vars is False

    def test_var_has_output_vars_for_output(self):
        """Test that output variable has output_vars."""
        v = Var("Y_5")
        assert v.has_output_vars is True

    def test_var_has_no_input_vars_for_output(self):
        """Test that output variable has no input_vars."""
        v = Var("Y_5")
        assert v.has_input_vars is False

    def test_var_repr_input(self):
        """Test string representation of input variable."""
        v = Var("X_42")
        assert repr(v) == "X_42"

    def test_var_repr_output(self):
        """Test string representation of output variable."""
        v = Var("Y_99")
        assert repr(v) == "Y_99"

    @pytest.mark.parametrize(
        ("name", "prop_name"),
        [
            pytest.param("X_0", "has_input_vars", id="input"),
            pytest.param("Y_0", "has_output_vars", id="output"),
        ],
    )
    def test_var_caching(self, name, prop_name):
        """Test that has_input_vars / has_output_vars is cached (STR2: merged pair)."""
        v = Var(name)
        result1 = getattr(v, prop_name)
        result2 = getattr(v, prop_name)
        assert result1 is True
        assert result1 is result2
        internal = "_" + prop_name
        assert getattr(v, internal) is True


class TestBinaryOperatorsComprehensive:
    """Comprehensive tests for binary operator expressions."""

    def test_sub_basic(self):
        """Test subtraction operation."""
        expr = Sub(Var("X_0"), Cst(1.0))
        assert cast(Var, expr.left).name == "X_0"
        assert cast(Cst, expr.right).value == 1.0

    def test_sub_equality(self):
        """Test equality of subtraction expressions."""
        expr1 = Sub(Var("X_0"), Cst(1.0))
        expr2 = Sub(Var("X_0"), Cst(1.0))
        assert expr1 == expr2

    def test_sub_inequality_different_operands(self):
        """Test inequality with different operands."""
        expr1 = Sub(Var("X_0"), Cst(1.0))
        expr2 = Sub(Var("X_0"), Cst(2.0))
        assert expr1 != expr2

    def test_sub_inequality_reversed_operands(self):
        """Test inequality with reversed operands."""
        expr1 = Sub(Var("X_0"), Cst(1.0))
        expr2 = Sub(Cst(1.0), Var("X_0"))
        assert expr1 != expr2

    @pytest.mark.parametrize(
        ("op_class", "left", "right"),
        [
            pytest.param(Sub, Var("X_0"), Cst(1.0), id="sub"),
            pytest.param(Mul, Cst(2.0), Var("X_0"), id="mul"),
            pytest.param(Div, Var("X_0"), Cst(2.0), id="div"),
            pytest.param(Eq, Var("X_0"), Cst(0.5), id="eq"),
            pytest.param(Leq, Var("X_0"), Cst(1.0), id="leq"),
            pytest.param(Geq, Var("X_0"), Cst(0.0), id="geq"),
        ],
    )
    def test_hash_consistency(self, op_class, left, right):
        """Test that equal binary op expressions have equal hashes."""
        assert hash(op_class(left, right)) == hash(op_class(left, right))

    def test_sub_has_input_vars_from_left(self):
        """Test variable propagation from left operand."""
        expr = Sub(Var("X_0"), Cst(1.0))
        assert expr.has_input_vars is True

    def test_sub_has_input_vars_from_right(self):
        """Test variable propagation from right operand."""
        expr = Sub(Cst(1.0), Var("X_0"))
        assert expr.has_input_vars is True

    def test_sub_has_input_vars_from_both(self):
        """Test variable propagation from both operands."""
        expr = Sub(Var("X_0"), Var("X_1"))
        assert expr.has_input_vars is True

    def test_sub_no_vars(self):
        """Test sub with no variables."""
        expr = Sub(Cst(5.0), Cst(3.0))
        assert expr.has_input_vars is False
        assert expr.has_output_vars is False

    def test_mul_basic(self):
        """Test multiplication operation."""
        expr = Mul(Cst(2.0), Var("X_0"))
        assert cast(Cst, expr.left).value == 2.0
        assert cast(Var, expr.right).name == "X_0"

    def test_mul_zero(self):
        """Test multiplication by zero."""
        expr = Mul(Cst(0.0), Var("X_0"))
        assert expr.has_input_vars is True

    def test_div_basic(self):
        """Test division operation."""
        expr = Div(Var("X_0"), Cst(2.0))
        assert cast(Var, expr.left).name == "X_0"
        assert cast(Cst, expr.right).value == 2.0

    def test_div_by_one(self):
        """Test division by one."""
        expr = Div(Var("X_0"), Cst(1.0))
        assert expr.has_input_vars is True

    def test_div_zero_dividend(self):
        """Test division with zero as dividend."""
        expr = Div(Cst(0.0), Var("X_0"))
        assert expr.has_input_vars is True

    def test_eq_basic(self):
        """Test equality constraint."""
        expr = Eq(Var("X_0"), Cst(0.5))
        assert cast(Var, expr.left).name == "X_0"
        assert cast(Cst, expr.right).value == 0.5

    def test_leq_basic(self):
        """Test less than or equal constraint."""
        expr = Leq(Var("X_0"), Cst(1.0))
        assert cast(Var, expr.left).name == "X_0"
        assert cast(Cst, expr.right).value == 1.0

    def test_leq_with_zero(self):
        """Test Leq with zero value."""
        expr = Leq(Var("X_0"), Cst(0.0))
        assert expr.has_input_vars is True

    def test_leq_with_negative(self):
        """Test Leq with negative value."""
        expr = Leq(Var("X_0"), Cst(-5.0))
        assert expr.has_input_vars is True

    def test_geq_basic(self):
        """Test greater than or equal constraint."""
        expr = Geq(Var("X_0"), Cst(0.0))
        assert cast(Var, expr.left).name == "X_0"
        assert cast(Cst, expr.right).value == 0.0

    def test_binary_op_repr_sub(self):
        """Test repr of subtraction."""
        expr = Sub(Var("X_0"), Cst(1.0))
        repr_str = repr(expr)
        assert "-" in repr_str

    def test_binary_op_repr_mul(self):
        """Test repr of multiplication."""
        expr = Mul(Cst(2.0), Var("X_0"))
        repr_str = repr(expr)
        assert "*" in repr_str

    def test_binary_op_repr_div(self):
        """Test repr of division."""
        expr = Div(Var("X_0"), Cst(2.0))
        repr_str = repr(expr)
        assert "/" in repr_str

    def test_binary_op_repr_eq(self):
        """Test repr of equality."""
        expr = Eq(Var("X_0"), Cst(0.5))
        repr_str = repr(expr)
        assert "=" in repr_str

    def test_binary_op_repr_leq(self):
        """Test repr of less than or equal."""
        expr = Leq(Var("X_0"), Cst(1.0))
        repr_str = repr(expr)
        assert "<=" in repr_str

    def test_binary_op_repr_geq(self):
        """Test repr of greater than or equal."""
        expr = Geq(Var("X_0"), Cst(0.0))
        repr_str = repr(expr)
        assert ">=" in repr_str

    def test_binary_op_caching_input_vars(self):
        """Test variable caching in binary operations."""
        expr = Sub(Var("X_0"), Cst(1.0))
        result1 = expr.has_input_vars
        result2 = expr.has_input_vars
        assert result1 is True
        assert result1 is result2

    def test_binary_op_caching_output_vars(self):
        """Test variable caching in binary operations."""
        expr = Leq(Var("Y_0"), Cst(1.0))
        result1 = expr.has_output_vars
        result2 = expr.has_output_vars
        assert result1 is True
        assert result1 is result2


class TestNaryOperatorsComprehensive:
    """Comprehensive tests for n-ary operator expressions."""

    @pytest.mark.parametrize(
        ("op_class", "args"),
        [
            pytest.param(Add, [Cst(1.0), Cst(2.0)], id="add"),
            pytest.param(And, [Cst(1.0), Cst(2.0)], id="and"),
            pytest.param(Or, [Cst(1.0), Cst(2.0)], id="or"),
        ],
    )
    def test_hash_consistency(self, op_class, args):
        """Test that equal n-ary expressions have equal hashes."""
        assert hash(op_class(args)) == hash(op_class(args))

    @pytest.mark.parametrize(
        ("op_class", "args"),
        [
            pytest.param(Add, [Cst(1.0), Cst(2.0)], id="add"),
            pytest.param(And, [Cst(1.0), Cst(2.0)], id="and"),
            pytest.param(Or, [Cst(1.0), Cst(2.0)], id="or"),
        ],
    )
    def test_equality(self, op_class, args):
        """Test equality of n-ary expressions."""
        assert op_class(args) == op_class(args)

    @pytest.mark.parametrize(
        ("op_class", "args", "symbol"),
        [
            pytest.param(Add, [Cst(1.0), Cst(2.0)], "+", id="add"),
            pytest.param(And, [Cst(1.0), Cst(2.0)], "and", id="and"),
            pytest.param(Or, [Cst(1.0), Cst(2.0)], "or", id="or"),
        ],
    )
    def test_repr(self, op_class, args, symbol):
        """Test string representation of n-ary expressions."""
        assert symbol in repr(op_class(args))

    def test_add_single_element(self):
        """Test Add with single element."""
        expr = Add([Var("X_0")])
        assert len(expr.args) == 1
        assert expr.has_input_vars is True

    def test_add_many_elements(self):
        """Test Add with many elements."""
        expr = Add([Cst(1.0), Cst(2.0), Cst(3.0), Cst(4.0), Cst(5.0)])
        assert len(expr.args) == 5

    def test_add_inequality_different_order(self):
        """Test inequality when argument order differs."""
        expr1 = Add([Cst(1.0), Cst(2.0)])
        expr2 = Add([Cst(2.0), Cst(1.0)])
        # Note: Different order means different expressions
        assert expr1 != expr2

    def test_add_inequality_different_length(self):
        """Test inequality with different number of arguments."""
        expr1 = Add([Cst(1.0), Cst(2.0)])
        expr2 = Add([Cst(1.0), Cst(2.0), Cst(3.0)])
        assert expr1 != expr2

    @pytest.mark.parametrize(
        ("args", "expect_input"),
        [
            pytest.param([Cst(1.0), Var("X_0"), Cst(2.0)], True, id="with_var"),
            pytest.param([Cst(1.0), Cst(2.0)], False, id="all_constants"),
        ],
    )
    def test_add_has_input_vars(self, args, expect_input):
        """Test that Add has_input_vars detection (STR2: merged pair)."""
        expr = Add(args)
        assert expr.has_input_vars is expect_input

    def test_add_has_output_vars_any_operand(self):
        """Test that Add has output vars if any operand does."""
        expr = Add([Var("Y_0"), Cst(1.0)])
        assert expr.has_output_vars is True

    def test_add_mixed_vars(self):
        """Test Add with mixed input and output variables."""
        expr = Add([Var("X_0"), Var("Y_0")])
        assert expr.has_input_vars is True
        assert expr.has_output_vars is True

    def test_add_iteration(self):
        """Test iterating over Add operands."""
        operands: list[Expr] = [Cst(1.0), Cst(2.0), Cst(3.0)]
        expr = Add(operands)
        result = list(expr)
        assert result == operands

    def test_and_single_element(self):
        """Test And with single element."""
        expr = And([Leq(Var("X_0"), Cst(1.0))])
        assert len(expr.args) == 1
        assert expr.has_input_vars is True

    def test_and_many_elements(self):
        """Test And with many elements."""
        constraints: list[Expr] = [
            Leq(Var("X_0"), Cst(1.0)),
            Geq(Var("X_0"), Cst(0.0)),
            Leq(Var("X_1"), Cst(2.0)),
        ]
        expr = And(constraints)
        assert len(expr.args) == 3

    def test_and_inequality_different_operands(self):
        """Test inequality with different operands."""
        expr1 = And([Leq(Var("X_0"), Cst(1.0))])
        expr2 = And([Leq(Var("X_0"), Cst(2.0))])
        assert expr1 != expr2

    @pytest.mark.parametrize(
        ("var_name", "prop_name"),
        [
            pytest.param("X_0", "has_input_vars", id="input"),
            pytest.param("Y_0", "has_output_vars", id="output"),
        ],
    )
    def test_and_has_vars(self, var_name, prop_name):
        """Test that And detects input/output variables (STR2: merged pair)."""
        expr = And([Leq(Var(var_name), Cst(1.0)), Geq(Var(var_name), Cst(0.0))])
        assert getattr(expr, prop_name) is True

    def test_and_mixed_vars(self):
        """Test And with mixed input and output variables."""
        expr = And([Leq(Var("X_0"), Cst(1.0)), Leq(Var("Y_0"), Cst(0.5))])
        assert expr.has_input_vars is True
        assert expr.has_output_vars is True

    def test_and_iteration(self):
        """Test iterating over And operands."""
        constraints: list[Expr] = [Leq(Var("X_0"), Cst(1.0)), Geq(Var("X_0"), Cst(0.0))]
        expr = And(constraints)
        result = list(expr)
        assert result == constraints

    def test_or_single_element(self):
        """Test Or with single element."""
        expr = Or([Leq(Var("X_0"), Cst(0.5))])
        assert len(expr.args) == 1
        assert expr.has_input_vars is True

    def test_or_many_elements(self):
        """Test Or with many elements."""
        constraints: list[Expr] = [
            Leq(Var("X_0"), Cst(0.3)),
            Leq(Var("X_0"), Cst(0.7)),
            Leq(Var("X_0"), Cst(1.0)),
        ]
        expr = Or(constraints)
        assert len(expr.args) == 3

    @pytest.mark.parametrize(
        ("var_name", "prop_name"),
        [
            pytest.param("X_0", "has_input_vars", id="input"),
            pytest.param("Y_0", "has_output_vars", id="output"),
        ],
    )
    def test_or_has_vars(self, var_name, prop_name):
        """Test that Or detects input/output variables (STR2: merged pair)."""
        expr = Or([Leq(Var(var_name), Cst(0.5)), Geq(Var(var_name), Cst(0.75))])
        assert getattr(expr, prop_name) is True

    def test_or_iteration(self):
        """Test iterating over Or operands."""
        constraints: list[Expr] = [Leq(Var("X_0"), Cst(0.5)), Geq(Var("X_0"), Cst(0.75))]
        expr = Or(constraints)
        result = list(expr)
        assert result == constraints

    @pytest.mark.parametrize(
        ("expr", "prop_name"),
        [
            pytest.param(Add([Var("X_0"), Cst(1.0)]), "has_input_vars", id="input"),
            pytest.param(And([Leq(Var("Y_0"), Cst(1.0))]), "has_output_vars", id="output"),
        ],
    )
    def test_nary_op_caching(self, expr, prop_name):
        """Test variable caching in nary operations (STR2: merged pair)."""
        result1 = getattr(expr, prop_name)
        result2 = getattr(expr, prop_name)
        assert result1 is True
        assert result1 is result2


class TestComplexNestedExpressions:
    """Test complex nested expression structures."""

    def test_deeply_nested_arithmetic(self):
        """Test deeply nested arithmetic expressions."""
        expr: Expr = Cst(1.0)
        for _ in range(5):
            expr = Add([expr, Cst(1.0)])
        assert expr.has_input_vars is False
        assert expr.has_output_vars is False

    def test_nested_arithmetic_with_vars(self):
        """Test nested arithmetic with variables."""
        inner = Add([Var("X_0"), Cst(1.0)])
        outer = Mul(Cst(2.0), inner)
        assert outer.has_input_vars is True
        assert outer.has_output_vars is False

    def test_complex_nested_arithmetic(self):
        """Test complex nested arithmetic structure."""
        e1 = Mul(Cst(2.0), Var("X_0"))
        e2 = Mul(Cst(3.0), Var("X_1"))
        expr = Add([e1, e2])
        assert expr.has_input_vars is True

    def test_nested_constraints(self):
        """Test nested constraint expressions."""
        inner = Add([Mul(Cst(2.0), Var("X_0")), Var("X_1")])
        expr = Leq(inner, Cst(5.0))
        assert expr.has_input_vars is True

    def test_nested_and_or_mixed(self):
        """Test mixed And/Or nesting."""
        and_expr = And([Leq(Var("X_0"), Cst(1.0)), Geq(Var("X_0"), Cst(0.0))])
        or_expr = Or([and_expr, Leq(Var("X_1"), Cst(0.5))])
        assert or_expr.has_input_vars is True

    def test_deeply_nested_and_or(self):
        """Test deeply nested And/Or structure."""
        level1 = And([Leq(Var("X_0"), Cst(1.0)), Geq(Var("X_0"), Cst(0.0))])
        level2 = Or([level1, Leq(Var("X_1"), Cst(0.5))])
        level3 = And([level2, Geq(Var("X_2"), Cst(-1.0))])
        assert level3.has_input_vars is True

    def test_mixed_input_output_vars(self):
        """Test expression with both input and output variables."""
        expr = Add([Var("X_0"), Var("Y_0")])
        assert expr.has_input_vars is True
        assert expr.has_output_vars is True

    def test_mixed_vars_in_constraint(self):
        """Test constraint mixing input and output variables."""
        expr = Leq(Add([Var("X_0"), Var("Y_0")]), Cst(1.0))
        assert expr.has_input_vars is True
        assert expr.has_output_vars is True

    def test_division_in_expression(self):
        """Test expression with division."""
        expr = Div(Add([Var("X_0"), Var("X_1")]), Cst(2.0))
        assert expr.has_input_vars is True

    def test_division_nested(self):
        """Test nested division expressions."""
        inner = Div(Var("X_0"), Cst(2.0))
        outer = Div(inner, Cst(3.0))
        assert outer.has_input_vars is True

    def test_subtraction_chain(self):
        """Test chained subtraction expressions."""
        expr = Sub(Sub(Var("X_0"), Var("X_1")), Var("X_2"))
        assert expr.has_input_vars is True

    def test_complex_linear_combination(self):
        """Test complex linear combination."""
        terms: list[Expr] = [
            Mul(Cst(2.0), Var("X_0")),
            Mul(Cst(3.0), Var("X_1")),
            Mul(Cst(-1.0), Var("X_2")),
        ]
        expr = Add(terms)
        assert expr.has_input_vars is True


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_hash_consistency_across_types(self):
        """Test that equal expressions have consistent hashes."""
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

    def test_equality_reflexive(self):
        """Test reflexivity of equality."""
        c = Cst(5.0)
        assert c == c

    def test_equality_symmetric(self):
        """Test symmetry of equality."""
        c1 = Cst(5.0)
        c2 = Cst(5.0)
        assert c1 == c2
        assert c2 == c1

    def test_cst_with_negative_zero(self):
        """Test constant with negative zero."""
        c1 = Cst(-0.0)
        c2 = Cst(0.0)
        assert c1 == c2

    def test_very_large_expression_nary(self):
        """Test n-ary operation with many arguments."""
        args: list[Expr] = [Cst(float(i)) for i in range(100)]
        expr = Add(args)
        assert len(expr.args) == 100
        assert expr.has_input_vars is False

    def test_very_large_expression_nary_with_var(self):
        """Test n-ary operation with many arguments including one variable."""
        args: list[Expr] = [Cst(float(i)) for i in range(100)]
        args.append(Var("X_0"))
        expr = Add(args)
        assert len(expr.args) == 101
        assert expr.has_input_vars is True

    def test_var_repr_contains_name(self):
        """Test that variable repr contains its name."""
        v = Var("X_42")
        assert "X_42" in repr(v)

    def test_cst_repr_contains_value(self):
        """Test that constant repr contains its value."""
        c = Cst(3.14)
        repr_str = repr(c)
        # Check for either exact or approximate representation
        assert "3.14" in repr_str or "3" in repr_str

    def test_scientific_notation_equality(self):
        """Test equality with scientific notation values."""
        c1 = Cst(1.5e-10)
        c2 = Cst(1.5e-10)
        assert c1 == c2

    def test_binary_op_inequality_with_non_binaryop(self):
        """Test inequality with non-BinaryOp."""
        expr = Sub(Var("X_0"), Cst(1.0))
        assert expr != Var("X_0")
        assert expr != Cst(1.0)

    def test_nary_op_inequality_with_non_naryop(self):
        """Test inequality with non-NaryOp."""
        expr = Add([Cst(1.0), Cst(2.0)])
        assert expr != Cst(3.0)
        assert expr != Var("X_0")

    def test_cst_inequality_with_near_value(self):
        """Test inequality with slightly different value."""
        c1 = Cst(5.0)
        c2 = Cst(5.0000001)
        # These are not equal (float comparison)
        assert c1 != c2

    def test_binary_op_base_class(self):
        """Test BinaryOp base class methods."""
        # BinaryOp is abstract but we can test through its subclasses
        expr = Sub(Var("X_0"), Cst(1.0))
        assert isinstance(expr, BinaryOp)
        assert expr.has_input_vars is True

    def test_nary_op_base_class(self):
        """Test NaryOp base class methods."""
        # NaryOp is abstract but we can test through its subclasses
        expr = Add([Var("X_0"), Cst(1.0)])
        assert isinstance(expr, NaryOp)
        assert expr.has_input_vars is True

    def test_expr_base_class(self):
        """Test Expr base class methods."""
        # Expr is abstract but we can test through its subclasses
        expr = Cst(5.0)
        assert isinstance(expr, Expr)
        assert expr.has_input_vars is False


class TestExpressionReprComprehensive:
    """Comprehensive tests for __repr__ methods."""

    def test_cst_repr_format(self):
        """Test Cst repr format."""
        assert repr(Cst(0.0)) == "0.0"
        assert repr(Cst(-5.5)) == "-5.5"

    def test_var_repr_format(self):
        """Test Var repr format."""
        assert repr(Var("X_0")) == "X_0"
        assert repr(Var("Y_42")) == "Y_42"

    def test_add_repr_format(self):
        """Test Add repr contains operator."""
        expr = Add([Cst(1.0), Cst(2.0)])
        repr_str = repr(expr)
        assert "+" in repr_str
        assert "1.0" in repr_str
        assert "2.0" in repr_str

    def test_sub_repr_format(self):
        """Test Sub repr contains operator."""
        expr = Sub(Cst(5.0), Cst(3.0))
        repr_str = repr(expr)
        assert "-" in repr_str
        assert "5.0" in repr_str
        assert "3.0" in repr_str

    def test_mul_repr_format(self):
        """Test Mul repr contains operator."""
        expr = Mul(Cst(2.0), Cst(3.0))
        repr_str = repr(expr)
        assert "*" in repr_str

    def test_div_repr_format(self):
        """Test Div repr contains operator."""
        expr = Div(Cst(6.0), Cst(2.0))
        repr_str = repr(expr)
        assert "/" in repr_str

    def test_eq_repr_format(self):
        """Test Eq repr contains operator."""
        expr = Eq(Var("X_0"), Cst(0.5))
        repr_str = repr(expr)
        assert "=" in repr_str

    def test_leq_repr_format(self):
        """Test Leq repr contains operator."""
        expr = Leq(Var("X_0"), Cst(1.0))
        repr_str = repr(expr)
        assert "<=" in repr_str

    def test_geq_repr_format(self):
        """Test Geq repr contains operator."""
        expr = Geq(Var("X_0"), Cst(0.0))
        repr_str = repr(expr)
        assert ">=" in repr_str

    def test_and_repr_format(self):
        """Test And repr contains keyword."""
        expr = And([Cst(1.0), Cst(2.0)])
        repr_str = repr(expr)
        assert "and" in repr_str

    def test_or_repr_format(self):
        """Test Or repr contains keyword."""
        expr = Or([Cst(1.0), Cst(2.0)])
        repr_str = repr(expr)
        assert "or" in repr_str

    def test_nested_expr_repr(self):
        """Test repr of nested expressions."""
        inner = Add([Cst(1.0), Cst(2.0)])
        outer = Mul(Cst(3.0), inner)
        repr_str = repr(outer)
        # Should contain both operators
        assert "*" in repr_str
        assert "+" in repr_str
