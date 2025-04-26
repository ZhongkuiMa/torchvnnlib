__docformat__ = "restructuredtext"
__all__ = ["convert_to_tensor"]

import re

import torch
from torch import Tensor

from .ast import *


def _get_n_inputs_outputs(input_bounds: And, output_constrs: Or) -> tuple[int, int]:
    """
    Find all string matches for "X_????" and "Y_????"
    where "?" is an integer
    The largest number is the number of inputs or outputs
    """

    # Find all matches for "X_????" and "Y_????"
    matches = re.findall(r"X_(\d+)", repr(input_bounds))
    if not matches:
        raise ValueError(f"Not found input bound expression: {input_bounds}")
    n_inputs = max(int(match) for match in matches)
    matches = re.findall(r"Y_(\d+)", repr(output_constrs))
    if not matches:
        raise ValueError(f"Not found output constraint expression: {output_constrs}")
    n_outputs = max(int(match) for match in matches)

    return n_inputs + 1, n_outputs + 1


def _convert_input_bounds(expr: And, n_inputs: int) -> Tensor:
    intput_bounds = [[None] * 2 for _ in range(n_inputs)]
    for sub_expr in expr:
        if not (
            isinstance(sub_expr, Leq)
            or isinstance(sub_expr, Geq)
            or isinstance(sub_expr, Eq)
        ):
            raise ValueError(f"Invalid input bound expression: {sub_expr}")
        idx = int(sub_expr.left.name[2:])  # noqa
        value = float(sub_expr.right.value)  # noqa
        if isinstance(sub_expr, Leq):
            intput_bounds[idx][1] = value  # noqa
        elif isinstance(sub_expr, Geq):
            intput_bounds[idx][0] = value  # noqa
        elif isinstance(sub_expr, Eq):
            intput_bounds[idx][0] = value  # noqa
            intput_bounds[idx][1] = value  # noqa
        else:
            raise RuntimeError(f"Invalid {sub_expr} to obtain input bounds.")

    for i in range(n_inputs):
        for j in range(2):
            if intput_bounds[i][j] is None:
                raise ValueError(f"Missing input bound for input {i}: {intput_bounds}")

    return torch.tensor(intput_bounds, dtype=torch.float64)


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
        # Extract the index from the variable name
        idx = int(expr.name[2:])
        if "Y" in expr.name:
            constr[idx + 1] += 1 if is_add else -1
        elif "X" in expr.name:
            constr[idx + y_dim + 1] += 1 if is_add else -1
        else:
            raise ValueError(f"Invalid variable name: {expr.name}")
    elif isinstance(expr, Cst):
        constr[0] += expr.value if is_add else -expr.value
    elif isinstance(expr, Add):
        for sub_expr in expr.args:
            _convert_linear_poly(constr, sub_expr, y_dim, x_dim, is_add)
    elif isinstance(expr, Mul):
        left = expr.left
        right = expr.right
        if isinstance(left, Cst) and isinstance(right, Var):
            idx = int(right.name[2:])
            if "Y" in right.name:
                constr[idx + 1] += left.value if is_add else -left.value
            elif "X" in right.name:
                constr[idx + y_dim + 1] += left.value if is_add else -left.value
            else:
                raise ValueError(f"Invalid variable name: {right.name}")
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
        idx = int(left.name[2:])
        constr[idx + 1] = -1
    elif isinstance(left, Cst):
        constr[0] = -left.value  # noqa
    else:
        raise NotImplementedError(f"Now only support Var and Cst for left: {left}")

    if isinstance(right, Var):
        idx = int(right.name[2:])
        constr[idx + 1] = 1
    elif isinstance(right, Cst):
        constr[0] += right.value
    elif _is_asmd(right):
        constr = torch.zeros(y_dim + x_dim + 1, dtype=torch.float64)
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
    output_constrs = []
    for i, sub_expr in enumerate(expr):
        left = sub_expr.left  # noqa
        right = sub_expr.right  # noqa
        assert isinstance(left, Cst)
        left = Cst(-left.value)

        if isinstance(sub_expr, Leq):
            constr = _convert_linear_constr(left, right, y_dim, x_dim)
        elif isinstance(sub_expr, Geq):
            constr = _convert_linear_constr(left, right, y_dim, x_dim)
            constr = -constr
        else:
            raise ValueError(f"Invalid output constraint expression: {sub_expr}")

        output_constrs.append(constr)
    output_constrs = torch.stack(output_constrs)

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


def _convert_one_property(expr: And) -> tuple[Tensor, list[Tensor]]:
    """
    Convert one property to a pair of input and output constraints.
    The input constraints are input bounds and the output constraints are constraints
    for the output variables. In some cases, the output constraints may involve the
    input variables as well.
    """
    input_bounds_expr: And
    output_constrs_expr: Or
    input_bounds_expr = expr.args[0]  # noqa
    output_constrs_expr = expr.args[1]  # noqa

    n_inputs, n_outputs = _get_n_inputs_outputs(input_bounds_expr, output_constrs_expr)

    input_bounds = _convert_input_bounds(input_bounds_expr, n_inputs)
    output_constrs = _convert_output_constrs(output_constrs_expr, n_outputs, n_inputs)

    return input_bounds, output_constrs


def convert_to_tensor(expr: And) -> list[list[tuple[Tensor, Tensor]]]:
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

    and_properties = []
    for or_expr in expr.args:
        or_groups = []
        or_expr: Or
        for and_expr in or_expr.args:
            if not isinstance(and_expr, And):
                raise ValueError(f"Invalid expression: {and_expr}")
            input_bounds, output_constrs = _convert_one_property(and_expr)
            or_groups.append((input_bounds, output_constrs))
        and_properties.append(or_groups)

    return and_properties
