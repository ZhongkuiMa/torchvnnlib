__docformat__ = "restructuredtext"
__all__ = ["convert_to_tensor"]

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import torch
from torch import Tensor

from .ast import *


def _convert_input_bounds(expr: And, n_inputs: int) -> Tensor:
    # Pre-allocate tensor with NaN for efficient filling (optimization)
    input_bounds = torch.full((n_inputs, 2), float('nan'), dtype=torch.float64)

    for sub_expr in expr:
        if not (
            isinstance(sub_expr, Leq)
            or isinstance(sub_expr, Geq)
            or isinstance(sub_expr, Eq)
        ):
            raise ValueError(f"Invalid input bound expression: {sub_expr}")
        idx = sub_expr.left.index  # Use cached index
        value = float(sub_expr.right.value)  # noqa

        if isinstance(sub_expr, Leq):
            input_bounds[idx, 1] = value  # Direct tensor assignment
        elif isinstance(sub_expr, Geq):
            input_bounds[idx, 0] = value  # Direct tensor assignment
        elif isinstance(sub_expr, Eq):
            input_bounds[idx, 0] = value  # Direct tensor assignment
            input_bounds[idx, 1] = value  # Direct tensor assignment
        else:
            raise RuntimeError(f"Invalid {sub_expr} to obtain input bounds.")

    # Check for missing bounds (NaN values)
    if torch.isnan(input_bounds).any():
        nan_indices = torch.where(torch.isnan(input_bounds))
        raise ValueError(f"Missing input bounds at indices: {list(zip(nan_indices[0].tolist(), nan_indices[1].tolist()))}")

    return input_bounds


def _convert_linear_poly(
    constr: Tensor, expr: Expr, y_dim: int, x_dim: int, is_add: bool = True
) -> Tensor:
    """
    The constraints is a linear polynomial with (x_dim + y_dim + 1) dimensions.
    The first dimension is the bias term.
    The second dimension part is for the output variables.
    The last dimension part is for the input variables.
    The expression must be a linear polynomial.
    We aim to convert it to a vector of size (y_dim + x_dim + 1).
    The first dimension is the bias term.
    The second dimension part are the coefficients for the output variables.
    The last dimension part are the coefficients for the input variables.
    This vector represents the linear polynomial in the form of
    b + Ax.

    (- Y_0 (+ (* 12.36122962667928 X_3) (* 13.97706212118972 X_2) 39.460058770740844))
    """
    if isinstance(expr, Var):
        # Use cached index and var_type
        idx = expr.index
        if expr.var_type == "Y":
            constr[idx + 1] += 1 if is_add else -1
        elif expr.var_type == "X":
            constr[idx + y_dim + 1] += 1 if is_add else -1
        else:
            raise ValueError(f"Invalid variable type: {expr.var_type}")
    elif isinstance(expr, Cst):
        constr[0] += expr.value if is_add else -expr.value
    elif isinstance(expr, Add):
        for sub_expr in expr.args:
            _convert_linear_poly(constr, sub_expr, y_dim, x_dim, is_add)
    elif isinstance(expr, Mul):
        left = expr.left
        right = expr.right
        if isinstance(left, Cst) and isinstance(right, Var):
            idx = right.index  # Use cached index
            if right.var_type == "Y":
                constr[idx + 1] += left.value if is_add else -left.value
            elif right.var_type == "X":
                constr[idx + y_dim + 1] += left.value if is_add else -left.value
            else:
                raise ValueError(f"Invalid variable type: {right.var_type}")
    elif isinstance(expr, Sub):
        left = expr.left
        right = expr.right
        _convert_linear_poly(constr, left, y_dim, x_dim)
        _convert_linear_poly(constr, right, y_dim, x_dim, is_add=False)

    return constr


def _convert_linear_constr(left: Expr, right: Expr, y_dim: int, x_dim: int) -> Tensor:
    """
    By default, we handle left <= right. The dimension includes the bias term.
    We use b + Ax >= 0 as the default form.
    """
    constr = torch.zeros(y_dim + 1, dtype=torch.float64)

    def _is_asmd(expr: Expr) -> bool:
        return (
            isinstance(expr, Add)
            or isinstance(expr, Sub)
            or isinstance(expr, Mul)
            or isinstance(expr, Div)
        )

    if isinstance(left, Var):
        idx = left.index  # Use cached index
        constr[idx + 1] += -1
    elif isinstance(left, Cst):
        constr[0] += -left.value  # noqa
    else:
        raise NotImplementedError(f"Now only support Var and Cst for left: {left}")

    if isinstance(right, Var):
        idx = right.index  # Use cached index
        constr[idx + 1] += 1
    elif isinstance(right, Cst):
        constr[0] += right.value
    elif _is_asmd(right):
        extended_constr = torch.zeros(y_dim + x_dim + 1, dtype=torch.float64)
        extended_constr[: y_dim + 1] = constr
        constr = extended_constr
        constr = _convert_linear_poly(constr, right, y_dim, x_dim)
    else:
        raise NotImplementedError(f"Now only support Var and Cst for right: {right}")

    return constr


def _convert_and_output_constrs(expr: And, n_outputs: int, n_inputs: int) -> Tensor:
    """
    We treat all constraints in the form of b + Ax >= 0
    """
    y_dim = n_outputs
    x_dim = n_inputs
    num_constraints = len(expr.args)

    # Pre-allocate constraints tensor for better performance (optimization)
    # We'll determine the max dimension needed
    max_dim = y_dim + 1  # Start with minimum dimension
    for sub_expr in expr.args:
        # Check if we need extended dimensions (when right side has Add/Sub/Mul/Div)
        if hasattr(sub_expr, 'right'):
            right = sub_expr.right
            if isinstance(right, (Add, Sub, Mul, Div)):
                max_dim = max(max_dim, y_dim + x_dim + 1)
                break

    # Pre-allocate the full tensor
    output_constrs_list = []

    for i, sub_expr in enumerate(expr):
        left = sub_expr.left  # noqa
        right = sub_expr.right  # noqa

        if isinstance(sub_expr, Leq):
            constr = _convert_linear_constr(left, right, y_dim, x_dim)
        elif isinstance(sub_expr, Geq):
            constr = _convert_linear_constr(left, right, y_dim, x_dim)
            constr = -constr
            constr.masked_fill_(constr == 0.0, 0.0)  # Remove negative zero
        else:
            raise ValueError(f"Invalid output constraint expression: {sub_expr}")

        output_constrs_list.append(constr)

    # Stack all constraints at once
    output_constrs = torch.stack(output_constrs_list)

    return output_constrs


def _convert_output_constrs(expr: Or, n_outputs: int, n_inputs: int) -> list[Tensor]:
    """
    The output is a list because we may have multiple Or expressions with different
    number of constraints. But the size should be the same in common cases.
    """
    or_output_constrs = []
    for and_expr in expr.args:
        and_expr: And
        and_output_constrs = _convert_and_output_constrs(and_expr, n_outputs, n_inputs)
        or_output_constrs.append(and_output_constrs)

    return or_output_constrs


def _convert_one_property(
    expr: And, n_inputs: int, n_outputs: int
) -> tuple[Tensor, list[Tensor]]:
    """
    Convert one property to a pair of input and output constraints.
    The input constraints are input bounds and the output constraints are constraints
    for the output variables. In some cases, the output constraints may involve the
    input variables as well.
    """
    assert isinstance(expr, And), f"Expected And expression, got {type(expr)}"
    input_bounds_expr: And
    output_constrs_expr: Or
    input_bounds_expr = expr.args[0]  # noqa
    output_constrs_expr = expr.args[1]  # noqa

    input_bounds = _convert_input_bounds(input_bounds_expr, n_inputs)
    output_constrs = _convert_output_constrs(output_constrs_expr, n_outputs, n_inputs)

    return input_bounds, output_constrs


def convert_to_tensor(
    expr: And, n_inputs: int, n_outputs: int
) -> list[list[tuple[Tensor, list[Tensor]]]]:
    """
    Now we should get an Expr.
    The first level is And.
    The second level is Or.
    The third level is And.

    The first level is a list of property groups. It means that all of them expect to
    be true.

    The second level is some properties that are in one group for Or properties. It
    means that one of them expects to be true.

    The third level is a pair of two And expressions. One is for input constraints, and
    the other is for output constraints.
    """

    # and_properties = []
    # for or_expr in expr.args:
    #     or_groups = []
    #     or_expr: Or
    #
    #     for and_expr in or_expr.args:
    #         if not isinstance(and_expr, And):
    #             raise ValueError(f"Invalid expression: {and_expr}")
    #         input_bounds, output_constrs = _convert_one_property(
    #             and_expr, n_inputs, n_outputs
    #         )
    #         or_groups.append((input_bounds, output_constrs))
    #
    #     and_properties.append(or_groups)

    def _process_or_expr(
        or_expr: Or, n_inputs: int, n_outputs: int
    ) -> list[tuple[Any, Any]]:
        if not isinstance(or_expr, Or):
            raise ValueError(f"Expected Or expression, got {type(or_expr)}")

        convert = partial(_convert_one_property, n_inputs=n_inputs, n_outputs=n_outputs)

        with ThreadPoolExecutor() as executor:
            or_groups = list(executor.map(convert, or_expr.args))

        return or_groups

    def convert_all_properties_parallel(expr, n_inputs: int, n_outputs: int):
        process = partial(_process_or_expr, n_inputs=n_inputs, n_outputs=n_outputs)

        with ThreadPoolExecutor() as executor:
            and_properties = list(executor.map(process, expr.args))

        return and_properties

    return convert_all_properties_parallel(expr, n_inputs, n_outputs)
