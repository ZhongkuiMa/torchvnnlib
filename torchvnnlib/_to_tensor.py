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
        if not (isinstance(sub_expr, Leq) or isinstance(sub_expr, Geq)):
            raise ValueError(f"Invalid input bound expression: {sub_expr}")
        idx = int(sub_expr.left.name[2:])  # noqa
        value = float(sub_expr.right.value)  # noqa
        if isinstance(sub_expr, Leq):
            intput_bounds[idx][1] = value  # noqa
        elif isinstance(sub_expr, Geq):
            intput_bounds[idx][0] = value  # noqa

    for i in range(n_inputs):
        for j in range(2):
            if intput_bounds[i][j] is None:
                raise ValueError(f"Missing input bound for input {i}: {intput_bounds}")

    return torch.tensor(intput_bounds, dtype=torch.float64)


def _convert_linear_constr(left: Expr, right: Expr, dim: int) -> Tensor:
    """
    By default, we handle left <= right. The dimension includes the bias term.
    We use b + Ax >= 0 as the default form.
    """
    constr = [0] * dim

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
    else:
        raise NotImplementedError(f"Now only support Var and Cst for right: {right}")

    return torch.tensor(constr, dtype=torch.float64)


def _convert_and_output_constrs(
    expr: And, n_outputs: int, n_inputs: int = None, need_inputs: bool = False
) -> Tensor:
    if n_inputs is not None or need_inputs:
        raise NotImplementedError(
            f"Now only support output constraints with no inputs: {expr}"
        )
    n_constrs = len(expr.args)
    dim = n_outputs + 1 if not need_inputs else n_outputs + n_inputs + 1  # noqa
    output_constrs = torch.zeros((n_constrs, n_outputs + 1), dtype=torch.float64)
    for i, sub_expr in enumerate(expr):
        if isinstance(sub_expr, Leq):
            constr = _convert_linear_constr(sub_expr.left, sub_expr.right, dim)
        elif isinstance(sub_expr, Geq):
            constr = _convert_linear_constr(sub_expr.right, sub_expr.left, dim)
        else:
            raise ValueError(f"Invalid output constraint expression: {sub_expr}")
        output_constrs[i] = constr

    return output_constrs


def _convert_output_constrs(expr: Or, n_outputs: int) -> list[Tensor]:
    """
    The output is a list because we may have multiple Or expressions with different
    number of constraints. But the size should be the same in common cases.
    """
    or_output_constrs = []
    for and_expr in expr.args:
        and_expr: And
        and_output_constrs = _convert_and_output_constrs(and_expr, n_outputs)
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
    output_constrs = _convert_output_constrs(output_constrs_expr, n_outputs)

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
