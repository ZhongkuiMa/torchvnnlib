__docformat__ = "restructuredtext"
__all__ = [
    "convert_input_bounds",
    "convert_linear_poly",
    "convert_linear_constr",
    "convert_and_output_constrs",
    "convert_output_constrs",
    "convert_one_property",
    "convert_to_tensor",
]

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from torch import Tensor

from .ast import And, Or, Expr, Var, Cst, Leq, Geq, Eq, Add, Sub, Mul, Div


def convert_input_bounds(expr: And, n_inputs: int) -> Tensor:
    """Convert input bound expressions to tensor.

    :param expr: AND expression containing input bound constraints
    :param n_inputs: Number of input variables
    :return: Input bounds tensor of shape (n_inputs, 2)
    """
    input_bounds = torch.full((n_inputs, 2), float("nan"), dtype=torch.float64)

    for sub_expr in expr:
        if not isinstance(sub_expr, (Leq, Geq, Eq)):
            raise ValueError(f"Invalid input bound expression: {sub_expr}")

        idx = sub_expr.left.index
        value = float(sub_expr.right.value)

        if isinstance(sub_expr, Leq):
            input_bounds[idx, 1] = value
        elif isinstance(sub_expr, Geq):
            input_bounds[idx, 0] = value
        elif isinstance(sub_expr, Eq):
            input_bounds[idx, 0] = value
            input_bounds[idx, 1] = value
        else:
            raise RuntimeError(f"Invalid expression for input bounds: {sub_expr}")

    if torch.isnan(input_bounds).any():
        nan_indices = torch.where(torch.isnan(input_bounds))
        indices_list = list(zip(nan_indices[0].tolist(), nan_indices[1].tolist()))
        raise ValueError(f"Missing input bounds at indices: {indices_list}")

    return input_bounds


def convert_linear_poly(
    constr: Tensor, expr: Expr, y_dim: int, x_dim: int, is_add: bool = True
) -> Tensor:
    """Convert linear polynomial expression to constraint vector.

    Converts expression to form b + Ax where the constraint vector has dimensions:
    [bias, y_coeff_1, ..., y_coeff_n, x_coeff_1, ..., x_coeff_m]

    :param constr: Constraint tensor to accumulate coefficients
    :param expr: Expression to convert
    :param y_dim: Number of output variables
    :param x_dim: Number of input variables
    :param is_add: Whether to add or subtract coefficients
    :return: Updated constraint tensor
    """
    if isinstance(expr, Var):
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
            convert_linear_poly(constr, sub_expr, y_dim, x_dim, is_add)
    elif isinstance(expr, Mul):
        left = expr.left
        right = expr.right
        if isinstance(left, Cst) and isinstance(right, Var):
            idx = right.index
            if right.var_type == "Y":
                constr[idx + 1] += left.value if is_add else -left.value
            elif right.var_type == "X":
                constr[idx + y_dim + 1] += left.value if is_add else -left.value
            else:
                raise ValueError(f"Invalid variable type: {right.var_type}")
    elif isinstance(expr, Sub):
        left = expr.left
        right = expr.right
        convert_linear_poly(constr, left, y_dim, x_dim)
        convert_linear_poly(constr, right, y_dim, x_dim, is_add=False)

    return constr


def convert_linear_constr(left: Expr, right: Expr, y_dim: int, x_dim: int) -> Tensor:
    """Convert linear constraint to standard form b + Ax >= 0.

    Handles constraints of form left <= right.

    :param left: Left-hand side expression
    :param right: Right-hand side expression
    :param y_dim: Number of output variables
    :param x_dim: Number of input variables
    :return: Constraint tensor in form b + Ax >= 0
    """
    constr = torch.zeros(y_dim + 1, dtype=torch.float64)

    def _is_arithmetic(expr: Expr) -> bool:
        return isinstance(expr, (Add, Sub, Mul, Div))

    if isinstance(left, Var):
        idx = left.index
        constr[idx + 1] += -1
    elif isinstance(left, Cst):
        constr[0] += -left.value
    else:
        raise NotImplementedError(f"Only Var and Cst supported for left: {left}")

    if isinstance(right, Var):
        idx = right.index
        constr[idx + 1] += 1
    elif isinstance(right, Cst):
        constr[0] += right.value
    elif _is_arithmetic(right):
        extended_constr = torch.zeros(y_dim + x_dim + 1, dtype=torch.float64)
        extended_constr[: y_dim + 1] = constr
        constr = extended_constr
        constr = convert_linear_poly(constr, right, y_dim, x_dim)
    else:
        raise NotImplementedError(f"Only Var and Cst supported for right: {right}")

    return constr


def convert_and_output_constrs(expr: And, n_outputs: int, n_inputs: int) -> Tensor:
    """Convert AND output constraints to tensor.

    All constraints are in form b + Ax >= 0.

    :param expr: AND expression containing output constraints
    :param n_outputs: Number of output variables
    :param n_inputs: Number of input variables
    :return: Stacked constraint tensor
    """
    y_dim = n_outputs
    x_dim = n_inputs

    output_constrs_list = []

    for sub_expr in expr:
        left = sub_expr.left
        right = sub_expr.right

        if isinstance(sub_expr, Leq):
            constr = convert_linear_constr(left, right, y_dim, x_dim)
        elif isinstance(sub_expr, Geq):
            constr = convert_linear_constr(left, right, y_dim, x_dim)
            constr = -constr
            constr.masked_fill_(constr == 0.0, 0.0)
        else:
            raise ValueError(f"Invalid output constraint expression: {sub_expr}")

        output_constrs_list.append(constr)

    output_constrs = torch.stack(output_constrs_list)

    return output_constrs


def convert_output_constrs(expr: Or, n_outputs: int, n_inputs: int) -> list[Tensor]:
    """Convert OR output constraints to list of tensors.

    :param expr: OR expression containing AND groups
    :param n_outputs: Number of output variables
    :param n_inputs: Number of input variables
    :return: List of constraint tensors
    """
    or_output_constrs = []
    for and_expr in expr.args:
        and_output_constrs = convert_and_output_constrs(and_expr, n_outputs, n_inputs)
        or_output_constrs.append(and_output_constrs)

    return or_output_constrs


def convert_one_property(
    expr: And, n_inputs: int, n_outputs: int
) -> tuple[Tensor, list[Tensor]]:
    """Convert one property to input and output constraints.

    :param expr: AND expression containing input bounds and output constraints
    :param n_inputs: Number of input variables
    :param n_outputs: Number of output variables
    :return: Tuple of (input_bounds, output_constraints)
    """
    if not isinstance(expr, And):
        raise ValueError(f"Expected And expression, got {type(expr)}")

    input_bounds_expr = expr.args[0]
    output_constrs_expr = expr.args[1]

    input_bounds = convert_input_bounds(input_bounds_expr, n_inputs)
    output_constrs = convert_output_constrs(output_constrs_expr, n_outputs, n_inputs)

    return input_bounds, output_constrs


def convert_to_tensor(
    expr: And,
    n_inputs: int,
    n_outputs: int,
    verbose: bool = False,
    use_parallel: bool = True,
) -> list[list[tuple[Tensor, list[Tensor]]]]:
    """Convert expression tree to nested tensor structure.

    Expression hierarchy:
    - Level 1 (AND): Property groups (all must be true)
    - Level 2 (OR): Properties in group (one must be true)
    - Level 3 (AND): Input bounds and output constraints

    :param expr: Root AND expression
    :param n_inputs: Number of input variables
    :param n_outputs: Number of output variables
    :param verbose: Print timing information
    :param use_parallel: Use parallel processing
    :return: Nested list of (input_bounds, output_constraints) tuples
    """

    def _process_or_expr(
        or_expr: Or, n_inputs: int, n_outputs: int
    ) -> list[tuple[Tensor, list[Tensor]]]:
        if not isinstance(or_expr, Or):
            raise ValueError(f"Expected Or expression, got {type(or_expr)}")

        convert = partial(convert_one_property, n_inputs=n_inputs, n_outputs=n_outputs)

        if use_parallel:
            with ThreadPoolExecutor() as executor:
                or_groups = list(executor.map(convert, or_expr.args))
        else:
            or_groups = [convert(arg) for arg in or_expr.args]

        return or_groups

    def convert_all_properties(
        expr: And, n_inputs: int, n_outputs: int
    ) -> list[list[tuple[Tensor, list[Tensor]]]]:
        process = partial(_process_or_expr, n_inputs=n_inputs, n_outputs=n_outputs)

        t = time.perf_counter()
        if use_parallel:
            with ThreadPoolExecutor() as executor:
                and_properties = list(executor.map(process, expr.args))
            if verbose:
                print(f"Convert properties (parallel): {time.perf_counter() - t:.4f}s")
        else:
            and_properties = [process(arg) for arg in expr.args]
            if verbose:
                print(
                    f"Convert properties (sequential): {time.perf_counter() - t:.4f}s"
                )

        return and_properties

    return convert_all_properties(expr, n_inputs, n_outputs)
