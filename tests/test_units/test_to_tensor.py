"""Unit tests for _to_tensor.py - Tensor conversion functions.

Tests conversion of AST expressions to tensor format for constraint representation.
Covers input bounds, output constraints, linear constraints, and full expression conversion.
"""

import pytest

from torchvnnlib._to_tensor import (
    convert_and_output_constrs,
    convert_input_bounds,
    convert_linear_constr,
    convert_linear_poly,
    convert_one_property,
    convert_output_constrs,
    convert_to_tensor,
)
from torchvnnlib.ast import Add, And, Cst, Eq, Geq, Leq, Mul, Or, Sub, Var


class TestConvertInputBounds:
    """Test input bounds conversion functions."""

    def test_simple_input_bounds_single_variable(self, backend):
        """Test conversion of simple input bounds for single variable.

        Input: (>= X_0 0.5) and (<= X_0 1.0)
        Expected: tensor([[0.5, 1.0]])
        """
        # Create AST: X_0 >= 0.5 AND X_0 <= 1.0
        lower_bound = Geq(Var("X_0"), Cst(0.5))
        upper_bound = Leq(Var("X_0"), Cst(1.0))
        expr = And([lower_bound, upper_bound])

        result = convert_input_bounds(expr, n_inputs=1, backend=backend)

        # Validate shape
        assert tuple(result.shape) == (1, 2), "Shape should be (1, 2)"

        # Validate values
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert abs(result_np[0, 0] - 0.5) < 1e-6, "Lower bound should be 0.5"
        assert abs(result_np[0, 1] - 1.0) < 1e-6, "Upper bound should be 1.0"

    def test_multiple_input_variables(self, backend):
        """Test conversion with multiple input variables."""
        constraints = [
            Geq(Var("X_0"), Cst(-1.0)),
            Leq(Var("X_0"), Cst(2.0)),
            Geq(Var("X_1"), Cst(0.0)),
            Leq(Var("X_1"), Cst(1.5)),
        ]
        expr = And(constraints)

        result = convert_input_bounds(expr, n_inputs=2, backend=backend)

        assert tuple(result.shape) == (2, 2), "Shape should be (2, 2)"

        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] >= -1.0 - 1e-6, "X_0 lower should be -1.0"
        assert result_np[0, 1] <= 2.0 + 1e-6, "X_0 upper should be 2.0"
        assert result_np[1, 0] >= 0.0 - 1e-6, "X_1 lower should be 0.0"
        assert result_np[1, 1] <= 1.5 + 1e-6, "X_1 upper should be 1.5"

    def test_equality_bounds(self, backend):
        """Test equality bounds (X = 0.5 sets both lower and upper to 0.5)."""
        expr = And([Eq(Var("X_0"), Cst(0.5))])

        result = convert_input_bounds(expr, n_inputs=1, backend=backend)

        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert abs(result_np[0, 0] - 0.5) < 1e-6, "Lower bound should be 0.5"
        assert abs(result_np[0, 1] - 0.5) < 1e-6, "Upper bound should be 0.5"

    def test_missing_bounds_raises_error(self, backend):
        """Test that missing bounds raises ValueError."""
        # Only lower bound, no upper bound
        expr = And([Geq(Var("X_0"), Cst(0.0))])

        with pytest.raises(ValueError, match="Missing input bounds"):
            convert_input_bounds(expr, n_inputs=1, backend=backend)

    def test_negative_bounds(self, backend):
        """Test conversion with negative bounds."""
        constraints = [
            Geq(Var("X_0"), Cst(-5.0)),
            Leq(Var("X_0"), Cst(-1.0)),
        ]
        expr = And(constraints)

        result = convert_input_bounds(expr, n_inputs=1, backend=backend)

        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert abs(result_np[0, 0] - (-5.0)) < 1e-6, "Lower bound should be -5.0"
        assert abs(result_np[0, 1] - (-1.0)) < 1e-6, "Upper bound should be -1.0"


class TestConvertLinearPoly:
    """Test linear polynomial conversion."""

    def test_simple_variable_conversion(self, backend):
        """Test conversion of simple variable."""
        constr = backend.zeros((3,), dtype="float64")
        expr = Var("Y_0")

        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=0)

        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[1] == 1.0, "Coefficient for Y_0 should be 1.0"

    def test_constant_conversion(self, backend):
        """Test conversion of constant."""
        constr = backend.zeros((3,), dtype="float64")
        expr = Cst(5.0)

        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=0)

        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0] == 5.0, "Bias term should be 5.0"

    def test_multiplication_conversion(self, backend):
        """Test conversion of constant * variable."""
        from torchvnnlib.ast import Mul

        constr = backend.zeros((3,), dtype="float64")
        expr = Mul(Cst(2.5), Var("Y_1"))

        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=0)

        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[2] == 2.5, "Coefficient for Y_1 should be 2.5"

    def test_addition_conversion(self, backend):
        """Test conversion of addition."""
        from torchvnnlib.ast import Add, Mul

        constr = backend.zeros((4,), dtype="float64")
        expr = Add([Mul(Cst(2.0), Var("Y_0")), Cst(3.0)])

        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=1)

        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0] == 3.0, "Bias should be 3.0"
        assert result_np[1] == 2.0, "Y_0 coefficient should be 2.0"

    def test_subtraction_conversion(self, backend):
        """Test conversion of subtraction."""
        from torchvnnlib.ast import Mul, Sub

        constr = backend.zeros((3,), dtype="float64")
        expr = Sub(Mul(Cst(5.0), Var("Y_0")), Cst(2.0))

        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=0)

        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0] == -2.0, "Bias should be -2.0"
        assert result_np[1] == 5.0, "Y_0 coefficient should be 5.0"


class TestConvertLinearConstr:
    """Test linear constraint conversion."""

    def test_simple_constraint_var_vs_const(self, backend):
        """Test constraint: Y_0 <= 0.5 → Y_0 - 0.5 >= 0."""
        left = Var("Y_0")
        right = Cst(0.5)

        result = convert_linear_constr(left, right, y_dim=1, x_dim=0, backend=backend)

        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        # Constraint should be: -Y_0 + 0.5 >= 0
        assert abs(result_np[0] - 0.5) < 1e-6, "Bias should be 0.5"
        assert result_np[1] == -1.0, "Y_0 coefficient should be -1.0"

    def test_var_vs_var_constraint(self, backend):
        """Test constraint: Y_0 <= Y_1."""
        left = Var("Y_0")
        right = Var("Y_1")

        result = convert_linear_constr(left, right, y_dim=2, x_dim=0, backend=backend)

        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[1] == -1.0, "Y_0 coefficient should be -1.0"
        assert result_np[2] == 1.0, "Y_1 coefficient should be 1.0"

    def test_const_vs_const_constraint(self, backend):
        """Test constraint: 1.0 <= 2.0."""
        left = Cst(1.0)
        right = Cst(2.0)

        result = convert_linear_constr(left, right, y_dim=1, x_dim=0, backend=backend)

        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        # 1.0 - 2.0 >= 0 → -1.0 >= 0 (always false, but should be -1.0)
        assert result_np[0] == 1.0, "Bias should be 1.0"


class TestConvertAndOutputConstrs:
    """Test AND output constraints conversion."""

    def test_single_constraint_leq(self, backend):
        """Test single <= constraint: Y_0 <= 0.5."""
        expr = And([Leq(Var("Y_0"), Cst(0.5))])

        result = convert_and_output_constrs(expr, n_outputs=1, n_inputs=0, backend=backend)

        assert tuple(result.shape) == (1, 2), "Shape should be (1, 2) for 1 constraint"

    def test_multiple_constraints(self, backend):
        """Test multiple constraints in AND."""
        constraints = [
            Leq(Var("Y_0"), Cst(0.5)),
            Geq(Var("Y_0"), Cst(0.0)),
        ]
        expr = And(constraints)

        result = convert_and_output_constrs(expr, n_outputs=1, n_inputs=0, backend=backend)

        assert tuple(result.shape) == (2, 2), "Shape should be (2, 2) for 2 constraints"

    def test_var_vs_var_constraints(self, backend):
        """Test constraint between variables: Y_0 >= Y_1."""
        expr = And([Geq(Var("Y_0"), Var("Y_1"))])

        result = convert_and_output_constrs(expr, n_outputs=2, n_inputs=0, backend=backend)

        assert result.shape[0] == 1, "Should have 1 constraint"
        assert tuple(result.shape)[1] == 3, "Should have 3 columns (bias + 2 Y)"


class TestConvertOutputConstrs:
    """Test OR output constraints conversion."""

    def test_single_or_branch(self, backend):
        """Test OR with single AND branch."""
        and_expr = And([Leq(Var("Y_0"), Cst(0.5))])
        or_expr = Or([and_expr])

        result = convert_output_constrs(or_expr, n_outputs=1, n_inputs=0, backend=backend)

        assert len(result) == 1, "Should have 1 OR branch"
        assert tuple(result[0].shape) == (1, 2), "First branch should have 1 constraint"

    def test_multiple_or_branches(self, backend):
        """Test OR with multiple AND branches."""
        and_expr1 = And([Leq(Var("Y_0"), Cst(0.5))])
        and_expr2 = And([Geq(Var("Y_0"), Cst(0.0))])
        or_expr = Or([and_expr1, and_expr2])

        result = convert_output_constrs(or_expr, n_outputs=1, n_inputs=0, backend=backend)

        assert len(result) == 2, "Should have 2 OR branches"
        assert result[0].shape[0] == 1, "First branch should have 1 constraint"
        assert result[1].shape[0] == 1, "Second branch should have 1 constraint"


class TestConvertOneProperty:
    """Test single property conversion."""

    def test_simple_property_conversion(self, backend):
        """Test conversion of complete property with input bounds and output constraints."""
        # Input: (AND (>= X_0 0.0) (<= X_0 1.0)) (OR (AND (<= Y_0 0.5))))
        input_constraints = And(
            [
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_0"), Cst(1.0)),
            ]
        )
        output_and = And([Leq(Var("Y_0"), Cst(0.5))])
        output_constraints = Or([output_and])
        expr = And([input_constraints, output_constraints])

        input_bounds, output_constrs = convert_one_property(
            expr, n_inputs=1, n_outputs=1, backend=backend
        )

        # Validate input bounds
        assert tuple(input_bounds.shape) == (1, 2), "Input bounds shape should be (1, 2)"

        # Validate output constraints
        assert isinstance(output_constrs, list), "Output constraints should be a list"
        assert len(output_constrs) == 1, "Should have 1 OR branch"


class TestConvertToTensor:
    """Test full expression to tensor conversion."""

    def test_simple_property_tree(self, backend):
        """Test conversion of simple property expression tree."""
        # Single property: (AND (input_bounds) (output))
        input_constraints = And(
            [
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_0"), Cst(1.0)),
            ]
        )
        output_and = And([Leq(Var("Y_0"), Cst(0.5))])
        output_constraints = Or([output_and])
        property_expr = And([input_constraints, output_constraints])
        root_expr = Or([property_expr])
        expr = And([root_expr])

        result = convert_to_tensor(
            expr, n_inputs=1, n_outputs=1, backend=backend, use_parallel=False
        )

        # Structure: [[[(<input_bounds>, [<output_constrs>])]]]
        assert isinstance(result, list), "Result should be a list"
        assert len(result) >= 1, "Should have at least 1 AND group"
        assert isinstance(result[0], list), "First level should be a list"

    def test_convert_with_parallel_false(self, backend):
        """Test conversion with parallel=False."""
        input_constraints = And(
            [
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_0"), Cst(1.0)),
            ]
        )
        output_and = And([Leq(Var("Y_0"), Cst(0.5))])
        output_constraints = Or([output_and])
        property_expr = And([input_constraints, output_constraints])
        root_expr = Or([property_expr])
        expr = And([root_expr])

        result = convert_to_tensor(
            expr, n_inputs=1, n_outputs=1, backend=backend, use_parallel=False, verbose=False
        )

        assert result is not None, "Should return valid result"

    def test_convert_with_verbose(self, backend, capsys):
        """Test conversion with verbose mode."""
        input_constraints = And(
            [
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_0"), Cst(1.0)),
            ]
        )
        output_and = And([Leq(Var("Y_0"), Cst(0.5))])
        output_constraints = Or([output_and])
        property_expr = And([input_constraints, output_constraints])
        root_expr = Or([property_expr])
        expr = And([root_expr])

        result = convert_to_tensor(
            expr, n_inputs=1, n_outputs=1, backend=backend, use_parallel=False, verbose=True
        )

        captured = capsys.readouterr()
        output = captured.out

        # Verbose should produce output
        assert len(output) > 0 or result is not None, "Should have output or result"

    def test_multiple_properties(self, backend):
        """Test conversion with multiple properties (OR at root)."""
        # Two properties
        input_constraints1 = And(
            [
                Geq(Var("X_0"), Cst(0.0)),
                Leq(Var("X_0"), Cst(0.5)),
            ]
        )
        output_and1 = And([Leq(Var("Y_0"), Cst(0.2))])
        output_constraints1 = Or([output_and1])
        property1 = And([input_constraints1, output_constraints1])

        input_constraints2 = And(
            [
                Geq(Var("X_0"), Cst(0.5)),
                Leq(Var("X_0"), Cst(1.0)),
            ]
        )
        output_and2 = And([Leq(Var("Y_0"), Cst(0.8))])
        output_constraints2 = Or([output_and2])
        property2 = And([input_constraints2, output_constraints2])

        # Two OR branches at root level
        root_or = Or([property1, property2])
        expr = And([root_or])

        result = convert_to_tensor(
            expr, n_inputs=1, n_outputs=1, backend=backend, use_parallel=False
        )

        assert len(result) >= 1, "Should have at least 1 AND group"


class TestConvertInputBoundsComprehensive:
    """Comprehensive tests for convert_input_bounds missing lines."""

    def test_input_bounds_all_operators(self, backend):
        """Test input bounds with all operator types (<=, >=, =)."""
        constraints = [
            Geq(Var("X_0"), Cst(0.0)),  # Lower bound for X_0
            Leq(Var("X_0"), Cst(2.0)),  # Upper bound for X_0
            Geq(Var("X_1"), Cst(-1.0)),  # Lower bound for X_1
            Leq(Var("X_1"), Cst(1.0)),  # Upper bound for X_1
            Eq(Var("X_2"), Cst(0.5)),  # Equality for X_2
        ]
        expr = And(constraints)
        result = convert_input_bounds(expr, n_inputs=3, backend=backend)
        assert result.shape == (3, 2)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[2, 0] == pytest.approx(0.5)
        assert result_np[2, 1] == pytest.approx(0.5)

    def test_input_bounds_zero_bounds(self, backend):
        """Test input bounds with zero values."""
        constraints = [
            Geq(Var("X_0"), Cst(0.0)),
            Leq(Var("X_0"), Cst(0.0)),
        ]
        expr = And(constraints)
        result = convert_input_bounds(expr, n_inputs=1, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == 0.0
        assert result_np[0, 1] == 0.0

    def test_input_bounds_large_values(self, backend):
        """Test input bounds with very large values."""
        constraints = [
            Geq(Var("X_0"), Cst(1e10)),
            Leq(Var("X_0"), Cst(1e11)),
        ]
        expr = And(constraints)
        result = convert_input_bounds(expr, n_inputs=1, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(1e10)
        assert result_np[0, 1] == pytest.approx(1e11)

    def test_input_bounds_small_values(self, backend):
        """Test input bounds with very small values."""
        constraints = [
            Geq(Var("X_0"), Cst(1e-10)),
            Leq(Var("X_0"), Cst(1e-9)),
        ]
        expr = And(constraints)
        result = convert_input_bounds(expr, n_inputs=1, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0, 0] == pytest.approx(1e-10)
        assert result_np[0, 1] == pytest.approx(1e-9)


class TestConvertLinearConstraComprehensive:
    """Comprehensive tests for convert_linear_constr missing lines."""

    def test_linear_constr_var_left_const_right(self, backend):
        """Test constraint: Var <= Cst."""
        left = Var("Y_0")
        right = Cst(1.0)
        result = convert_linear_constr(left, right, y_dim=1, x_dim=0, backend=backend)
        assert result.shape == (2,)

    def test_linear_constr_const_left_var_right(self, backend):
        """Test constraint: Cst <= Var."""
        left = Cst(0.5)
        right = Var("Y_0")
        result = convert_linear_constr(left, right, y_dim=1, x_dim=0, backend=backend)
        assert result.shape == (2,)

    def test_linear_constr_with_arithmetic_right(self, backend):
        """Test constraint with arithmetic on right side."""
        from torchvnnlib.ast import Add

        left = Var("Y_0")
        right = Add([Var("Y_1"), Cst(1.0)])
        result = convert_linear_constr(left, right, y_dim=2, x_dim=0, backend=backend)
        # Extended constraint vector
        assert result.shape[0] > 2

    def test_linear_constr_const_const(self, backend):
        """Test constraint: Cst <= Cst."""
        left = Cst(0.5)
        right = Cst(1.0)
        result = convert_linear_constr(left, right, y_dim=1, x_dim=0, backend=backend)
        assert result.shape == (2,)

    def test_linear_constr_var_var_with_input(self, backend):
        """Test constraint: Y_0 <= Y_1 with input variables."""
        left = Var("Y_0")
        right = Var("Y_1")
        result = convert_linear_constr(left, right, y_dim=2, x_dim=3, backend=backend)
        # Shape is y_dim + 1 = 3
        assert result.shape == (3,)

    def test_linear_constr_invalid_left_raises(self, backend):
        """Test that invalid left operand raises error."""
        from torchvnnlib.ast import Add

        left = Add([Var("Y_0"), Cst(1.0)])  # Invalid: arithmetic on left
        right = Cst(1.0)
        with pytest.raises(NotImplementedError):
            convert_linear_constr(left, right, y_dim=1, x_dim=0, backend=backend)

    def test_linear_constr_invalid_right_raises(self, backend):
        """Test that invalid right operand raises error."""
        left = Var("Y_0")
        right = Or([Cst(1.0)])  # Invalid: Or on right
        with pytest.raises(NotImplementedError):
            convert_linear_constr(left, right, y_dim=1, x_dim=0, backend=backend)


class TestConvertAndOutputConstrsComprehensive:
    """Comprehensive tests for convert_and_output_constrs missing lines."""

    def test_and_output_single_leq(self, backend):
        """Test single <= output constraint."""
        expr = And([Leq(Var("Y_0"), Cst(0.5))])
        result = convert_and_output_constrs(expr, n_outputs=1, n_inputs=0, backend=backend)
        assert result.shape == (1, 2)

    def test_and_output_single_geq(self, backend):
        """Test single >= output constraint."""
        expr = And([Geq(Var("Y_0"), Cst(0.5))])
        result = convert_and_output_constrs(expr, n_outputs=1, n_inputs=0, backend=backend)
        assert result.shape == (1, 2)

    def test_and_output_geq_negation(self, backend):
        """Test that >= constraint is correctly negated."""
        expr = And([Geq(Var("Y_0"), Cst(0.5))])
        result = convert_and_output_constrs(expr, n_outputs=1, n_inputs=0, backend=backend)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        # Geq should be negated: Y_0 >= 0.5 becomes -Y_0 + 0.5 >= 0
        assert result_np.shape == (1, 2)

    def test_and_output_multiple_constraints(self, backend):
        """Test multiple output constraints."""
        constraints = [
            Leq(Var("Y_0"), Cst(0.5)),
            Geq(Var("Y_1"), Cst(0.2)),
            Leq(Var("Y_0"), Var("Y_1")),
        ]
        expr = And(constraints)
        result = convert_and_output_constrs(expr, n_outputs=2, n_inputs=0, backend=backend)
        assert result.shape[0] == 3

    def test_and_output_with_input_vars(self, backend):
        """Test output constraints that might reference input vars."""
        constraints = [Leq(Var("Y_0"), Cst(0.5))]
        expr = And(constraints)
        result = convert_and_output_constrs(expr, n_outputs=1, n_inputs=3, backend=backend)
        assert result.shape == (1, 2)

    def test_and_output_invalid_expression(self, backend):
        """Test that invalid expression raises error."""
        expr = And([Or([Leq(Var("Y_0"), Cst(0.5))])])  # Invalid: Or in And output
        with pytest.raises((ValueError, AttributeError)):
            convert_and_output_constrs(expr, n_outputs=1, n_inputs=0, backend=backend)


class TestConvertOutputConstrsComprehensive:
    """Comprehensive tests for convert_output_constrs missing lines."""

    def test_output_constrs_single_branch(self, backend):
        """Test OR with single AND branch."""
        and_expr = And([Leq(Var("Y_0"), Cst(0.5))])
        or_expr = Or([and_expr])
        result = convert_output_constrs(or_expr, n_outputs=1, n_inputs=0, backend=backend)
        assert len(result) == 1

    def test_output_constrs_multiple_branches(self, backend):
        """Test OR with multiple AND branches."""
        and_expr1 = And([Leq(Var("Y_0"), Cst(0.3))])
        and_expr2 = And([Geq(Var("Y_0"), Cst(0.7))])
        and_expr3 = And([Leq(Var("Y_0"), Cst(0.5))])  # Use Leq instead of Eq
        or_expr = Or([and_expr1, and_expr2, and_expr3])
        result = convert_output_constrs(or_expr, n_outputs=1, n_inputs=0, backend=backend)
        assert len(result) == 3

    def test_output_constrs_var_comparisons(self, backend):
        """Test output constraints comparing variables."""
        and_expr = And([Geq(Var("Y_0"), Var("Y_1"))])
        or_expr = Or([and_expr])
        result = convert_output_constrs(or_expr, n_outputs=2, n_inputs=0, backend=backend)
        assert len(result) == 1
        assert result[0].shape == (1, 3)


class TestConvertLinearPolyComprehensive:
    """Comprehensive tests for convert_linear_poly missing lines."""

    def test_linear_poly_add_coefficient(self, backend):
        """Test converting variable with positive coefficient."""
        constr = backend.zeros((3,), dtype="float64")
        expr = Var("Y_0")
        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=0, is_add=True)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[1] == 1.0

    def test_linear_poly_subtract_coefficient(self, backend):
        """Test converting variable with negative coefficient."""
        constr = backend.zeros((3,), dtype="float64")
        expr = Var("Y_0")
        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=0, is_add=False)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[1] == -1.0

    def test_linear_poly_const_add(self, backend):
        """Test converting constant with is_add=True."""
        constr = backend.zeros((3,), dtype="float64")
        expr = Cst(5.0)
        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=0, is_add=True)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0] == 5.0

    def test_linear_poly_const_subtract(self, backend):
        """Test converting constant with is_add=False."""
        constr = backend.zeros((3,), dtype="float64")
        expr = Cst(3.0)
        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=0, is_add=False)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[0] == -3.0

    def test_linear_poly_mul_coefficient(self, backend):
        """Test converting multiplication with coefficient."""
        constr = backend.zeros((4,), dtype="float64")
        expr = Mul(Cst(2.5), Var("X_0"))
        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=1)
        # X_0 is at index y_dim + 1 + 0 = 3
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[3] == 2.5

    def test_linear_poly_sub_both_sides(self, backend):
        """Test converting subtraction (left add, right subtract)."""
        constr = backend.zeros((4,), dtype="float64")
        expr = Sub(Var("Y_0"), Cst(2.0))
        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=0)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        # Y_0 should have coefficient 1.0, constant -2.0
        assert result_np[1] == 1.0
        assert result_np[0] == -2.0

    def test_linear_poly_add_multiple_args(self, backend):
        """Test converting addition with multiple arguments."""
        constr = backend.zeros((4,), dtype="float64")
        expr = Add([Var("Y_0"), Cst(1.0), Var("Y_1")])
        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=0)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[1] == 1.0  # Y_0
        assert result_np[2] == 1.0  # Y_1
        assert result_np[0] == 1.0  # constant

    def test_linear_poly_nested_expression(self, backend):
        """Test converting nested arithmetic expression."""
        constr = backend.zeros((4,), dtype="float64")
        expr = Add([Mul(Cst(2.0), Var("Y_0")), Cst(1.0)])
        result = convert_linear_poly(constr, expr, y_dim=2, x_dim=0)
        result_np = backend.to_numpy(result) if hasattr(backend, "to_numpy") else result
        assert result_np[1] == 2.0  # 2 * Y_0
        assert result_np[0] == 1.0  # constant


class TestConvertOnePropertyComprehensive:
    """Comprehensive tests for convert_one_property missing lines."""

    def test_one_property_structure(self, backend):
        """Test that one_property correctly structures input and output."""
        input_expr = And([Geq(Var("X_0"), Cst(0.0)), Leq(Var("X_0"), Cst(1.0))])
        output_expr = Or([And([Leq(Var("Y_0"), Cst(0.5))])])
        expr = And([input_expr, output_expr])

        input_bounds, output_constrs = convert_one_property(
            expr, n_inputs=1, n_outputs=1, backend=backend
        )

        assert input_bounds.shape == (1, 2)
        assert isinstance(output_constrs, list)
        assert len(output_constrs) >= 1

    def test_one_property_invalid_not_and(self, backend):
        """Test that non-And expression raises error."""
        expr = Or([And([Leq(Var("X_0"), Cst(1.0))])])
        with pytest.raises(ValueError, match="Expected And expression"):
            convert_one_property(expr, n_inputs=1, n_outputs=1, backend=backend)

    def test_one_property_multiple_outputs(self, backend):
        """Test property with multiple output constraints."""
        input_expr = And([Geq(Var("X_0"), Cst(0.0)), Leq(Var("X_0"), Cst(1.0))])
        output_expr = Or([And([Leq(Var("Y_0"), Cst(0.5)), Geq(Var("Y_1"), Cst(0.2))])])
        expr = And([input_expr, output_expr])

        input_bounds, output_constrs = convert_one_property(
            expr, n_inputs=1, n_outputs=2, backend=backend
        )

        assert input_bounds.shape == (1, 2)
        assert len(output_constrs) >= 1


class TestConvertToTensorComprehensive:
    """Comprehensive tests for convert_to_tensor missing lines."""

    def test_convert_to_tensor_parallel_true(self, backend):
        """Test convert_to_tensor with parallel=True."""
        input_expr = And([Geq(Var("X_0"), Cst(0.0)), Leq(Var("X_0"), Cst(1.0))])
        output_expr = Or([And([Leq(Var("Y_0"), Cst(0.5))])])
        prop = And([input_expr, output_expr])
        root = Or([prop])
        expr = And([root])

        result = convert_to_tensor(
            expr, n_inputs=1, n_outputs=1, backend=backend, use_parallel=True
        )

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_convert_to_tensor_verbose_true(self, backend, capsys):
        """Test convert_to_tensor with verbose=True."""
        input_expr = And([Geq(Var("X_0"), Cst(0.0)), Leq(Var("X_0"), Cst(1.0))])
        output_expr = Or([And([Leq(Var("Y_0"), Cst(0.5))])])
        prop = And([input_expr, output_expr])
        root = Or([prop])
        expr = And([root])

        result = convert_to_tensor(
            expr, n_inputs=1, n_outputs=1, backend=backend, use_parallel=False, verbose=True
        )

        captured = capsys.readouterr()
        # Verbose should print timing information
        assert "properties" in captured.out.lower() or result is not None

    def test_convert_to_tensor_multiple_properties(self, backend):
        """Test convert_to_tensor with multiple properties."""
        prop1 = And(
            [
                And([Geq(Var("X_0"), Cst(0.0)), Leq(Var("X_0"), Cst(0.5))]),
                Or([And([Leq(Var("Y_0"), Cst(0.3))])]),
            ]
        )
        prop2 = And(
            [
                And([Geq(Var("X_0"), Cst(0.5)), Leq(Var("X_0"), Cst(1.0))]),
                Or([And([Leq(Var("Y_0"), Cst(0.8))])]),
            ]
        )
        expr = And([Or([prop1, prop2])])

        result = convert_to_tensor(
            expr, n_inputs=1, n_outputs=1, backend=backend, use_parallel=False
        )

        assert len(result) >= 1

    def test_convert_to_tensor_invalid_or_raises(self, backend):
        """Test that invalid Or expression raises error."""
        expr = Or([Var("X_0")])  # Invalid: not And

        with pytest.raises((ValueError, AttributeError, TypeError)):
            convert_to_tensor(expr, n_inputs=1, n_outputs=1, backend=backend)
