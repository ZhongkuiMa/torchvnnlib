"""Conversion of parsed VNN-LIB AST expressions to tensor representations."""

__docformat__ = "restructuredtext"
__all__ = [
    "convert_and_output_constrs",
    "convert_input_bounds",
    "convert_linear_constr",
    "convert_linear_poly",
    "convert_one_property",
    "convert_output_constrs",
    "convert_to_tensor",
]

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Final, cast

from torchvnnlib._backend import Backend, TensorLike
from torchvnnlib._constraint_row import apply_input_bound, normalize_neg_zero, validate_input_bounds
from torchvnnlib.ast import Add, And, Cst, Div, Eq, Expr, Geq, Leq, Mul, Or, Sub, Var

_logger = logging.getLogger(__name__)

# Lookup once instead of rebuilding inside ``convert_input_bounds`` per call.
_INPUT_BOUND_OP: Final[dict[type[Leq | Geq | Eq], str]] = {
    Leq: "<=",
    Geq: ">=",
    Eq: "=",
}


def convert_input_bounds(expr: And, n_inputs: int, backend: Backend) -> TensorLike:
    """Convert input bound expressions to tensor.

    :param expr: AND expression containing input bound constraints.

    :param n_inputs: Number of input variables.

    :param backend: Backend instance for tensor operations.

    :return: Input bounds tensor of shape (n_inputs, 2)
    """
    input_bounds = backend.full((n_inputs, 2), float("nan"), dtype="float64")

    for sub_expr in expr:
        if not isinstance(sub_expr, Leq | Geq | Eq):
            raise ValueError(f"Invalid input bound expression (expected Leq/Geq/Eq): {sub_expr}")
        # Input bounds must be ``Var op Cst`` with the Var on the left. Reject
        # ``Cst op Var`` (legal SMT-LIB but unsupported here), ``Var op Var``
        # (would silently pick wrong index), and arithmetic on either side.
        if not isinstance(sub_expr.left, Var):
            raise ValueError(
                f"Input bound expressions must have a single variable on the left, got: {sub_expr}"
            )
        if not isinstance(sub_expr.right, Cst):
            raise ValueError(
                f"Input bound expressions must have a constant on the right, got: {sub_expr}"
            )
        idx = sub_expr.left.index
        if not 0 <= idx < n_inputs:
            raise ValueError(f"Input bound variable index {idx} out of range [0, {n_inputs})")
        apply_input_bound(
            input_bounds,
            idx,
            _INPUT_BOUND_OP[type(sub_expr)],  # type: ignore[arg-type]
            float(sub_expr.right.value),
        )

    validate_input_bounds(input_bounds, backend)

    return normalize_neg_zero(input_bounds)


def _update_constr_for_var(
    constr: TensorLike, var: Var, y_dim: int, x_dim: int, coeff: float
) -> None:
    """Update constraint accumulator for a variable with given coefficient.

    ``var.var_type`` is constrained to ``Literal["X", "Y"]`` by ``Var.__init__``.
    Out-of-range indices raise ``ValueError`` instead of writing past the slot
    boundary or silently clobbering an unrelated column.
    """
    if var.var_type == "Y":
        if not 0 <= var.index < y_dim:
            raise ValueError(
                f"Output variable index {var.index} out of range [0, {y_dim}) "
                f"in constraint accumulator for {var.name}"
            )
        constr[var.index + 1] += coeff
    else:
        if not 0 <= var.index < x_dim:
            raise ValueError(
                f"Input variable index {var.index} out of range [0, {x_dim}) "
                f"in constraint accumulator for {var.name}"
            )
        constr[var.index + y_dim + 1] += coeff


def convert_linear_poly(
    constr: TensorLike, expr: Expr, y_dim: int, x_dim: int, is_add: bool = True
) -> TensorLike:
    """Convert linear polynomial expression to constraint vector.

    Converts expression to form b + Ax where the constraint vector has dimensions:
    [bias, y_coeff_1, ..., y_coeff_n, x_coeff_1, ..., x_coeff_m]

    :param constr: Constraint tensor to accumulate coefficients.

    :param expr: Expression to convert.

    :param y_dim: Number of output variables.

    :param x_dim: Number of input variables.

    :param is_add: Whether to add or subtract coefficients.

    :return: Updated constraint tensor
    """
    coeff_sign = 1 if is_add else -1

    if isinstance(expr, Var):
        _update_constr_for_var(constr, expr, y_dim, x_dim, coeff_sign)
    elif isinstance(expr, Cst):
        constr[0] += expr.value * coeff_sign
    elif isinstance(expr, Add):
        for sub_expr in expr.args:
            convert_linear_poly(constr, sub_expr, y_dim, x_dim, is_add)
    elif isinstance(expr, Mul):
        left = expr.left
        right = expr.right
        # Accept both ``Cst * Var`` and ``Var * Cst``; the multiplication is
        # commutative and either form may survive AST normalisation. Anything
        # else (e.g. ``Var * Var``) is non-linear and must not silently
        # disappear into the constraint accumulator.
        if isinstance(left, Cst) and isinstance(right, Var):
            _update_constr_for_var(constr, right, y_dim, x_dim, left.value * coeff_sign)
        elif isinstance(left, Var) and isinstance(right, Cst):
            _update_constr_for_var(constr, left, y_dim, x_dim, right.value * coeff_sign)
        else:
            raise NotImplementedError(f"Non-linear Mul (only Cst*Var supported): {expr}")
    elif isinstance(expr, Sub):
        convert_linear_poly(constr, expr.left, y_dim, x_dim, is_add)
        convert_linear_poly(constr, expr.right, y_dim, x_dim, is_add=not is_add)
    else:
        raise NotImplementedError(f"Unsupported expression in linear polynomial: {expr!r}")

    return constr


def convert_linear_constr(
    left: Expr, right: Expr, y_dim: int, x_dim: int, backend: Backend
) -> TensorLike:
    """Convert linear constraint to standard form b + Ax >= 0.

    Handles constraints of form left <= right.

    :param left: Left-hand side expression.

    :param right: Right-hand side expression.

    :param y_dim: Number of output variables.

    :param x_dim: Number of input variables.

    :param backend: Backend instance for tensor operations.

    :return: Constraint tensor in form b + Ax >= 0
    """
    constr: TensorLike = backend.zeros((y_dim + 1,), dtype="float64")

    if isinstance(left, Var):
        # Output constraints come in as ``left <= right`` with ``left`` a Y-var.
        # An X-var here would silently corrupt the Y-coefficient column.
        if left.var_type != "Y":
            raise NotImplementedError(
                f"Only Y-variables supported on the left of an output constraint, got: {left.name}"
            )
        if not 0 <= left.index < y_dim:
            raise ValueError(
                f"Output variable index {left.index} out of range [0, {y_dim}) "
                f"on left of constraint"
            )
        constr[left.index + 1] += -1
    elif isinstance(left, Cst):
        constr[0] += -left.value
    else:
        raise NotImplementedError(f"Only Var and Cst supported for left: {left}")

    if isinstance(right, Var):
        if right.var_type != "Y":
            raise NotImplementedError(
                f"Only Y-variables supported on the right of an output constraint, "
                f"got: {right.name}"
            )
        if not 0 <= right.index < y_dim:
            raise ValueError(
                f"Output variable index {right.index} out of range [0, {y_dim}) "
                f"on right of constraint"
            )
        constr[right.index + 1] += 1
    elif isinstance(right, Cst):
        constr[0] += right.value
    elif isinstance(right, Add | Sub | Mul | Div):
        extended_constr: TensorLike = backend.zeros((y_dim + x_dim + 1,), dtype="float64")
        extended_constr[: y_dim + 1] = constr  # type: ignore[assignment]  # noqa: RUF100
        constr = convert_linear_poly(extended_constr, right, y_dim, x_dim)
    else:
        raise NotImplementedError(f"Only Var and Cst supported for right: {right}")

    return constr


def convert_and_output_constrs(
    expr: And, n_outputs: int, n_inputs: int, backend: Backend
) -> TensorLike:
    """Convert AND output constraints to tensor.

    All constraints are in form b + Ax >= 0.

    :param expr: AND expression containing output constraints.

    :param n_outputs: Number of output variables.

    :param n_inputs: Number of input variables.

    :param backend: Backend instance for tensor operations.

    :return: Stacked constraint tensor
    """
    y_dim = n_outputs
    x_dim = n_inputs

    output_constrs_list = []

    for sub_expr in expr:
        left = sub_expr.left
        right = sub_expr.right

        if isinstance(sub_expr, Leq):
            constr = convert_linear_constr(left, right, y_dim, x_dim, backend)
        elif isinstance(sub_expr, Geq):
            constr = convert_linear_constr(left, right, y_dim, x_dim, backend)
            constr = -constr
        elif isinstance(sub_expr, Eq):
            constr_leq = convert_linear_constr(left, right, y_dim, x_dim, backend)
            constr_geq = -constr_leq
            normalize_neg_zero(constr_leq)
            normalize_neg_zero(constr_geq)
            output_constrs_list.append(constr_geq)
            output_constrs_list.append(constr_leq)
            continue
        else:
            raise ValueError(f"Invalid output constraint expression: {sub_expr}")

        # Normalize negative zero on every emitted row so the two pipelines
        # produce bit-identical tensors (Geq flips signs and ``-0.0`` would
        # otherwise leak; Leq with bias=0 can also produce ``-0.0`` via the
        # accumulator). Single helper call keeps the asymmetry from coming
        # back.
        normalize_neg_zero(constr)
        output_constrs_list.append(constr)

    output_constrs = backend.stack(output_constrs_list)

    return output_constrs


def convert_output_constrs(
    expr: Or, n_outputs: int, n_inputs: int, backend: Backend
) -> list[TensorLike]:
    """Convert OR output constraints to list of tensors.

    :param expr: OR expression containing AND groups.

    :param n_outputs: Number of output variables.

    :param n_inputs: Number of input variables.

    :param backend: Backend instance for tensor operations.

    :return: List of constraint tensors
    """
    or_output_constrs = []
    for and_expr in expr.args:
        and_output_constrs = convert_and_output_constrs(
            cast(And, and_expr), n_outputs, n_inputs, backend
        )
        or_output_constrs.append(and_output_constrs)

    return or_output_constrs


def convert_one_property(
    expr: And, n_inputs: int, n_outputs: int, backend: Backend
) -> tuple[TensorLike, list[TensorLike]]:
    """Convert one property to input and output constraints.

    :param expr: AND expression containing input bounds and output constraints.

    :param n_inputs: Number of input variables.

    :param n_outputs: Number of output variables.

    :param backend: Backend instance for tensor operations.

    :return: Tuple of (input_bounds, output_constraints)
    """
    if not isinstance(expr, And):
        raise ValueError(f"Expected And expression, got {type(expr)}")

    input_bounds_expr = expr.args[0]
    if len(expr.args) < 2:
        raise ValueError(
            f"Expected And expression with at least 2 args (input bounds + output constraints), "
            f"got {len(expr.args)} args: {expr}"
        )
    output_constrs_expr = expr.args[1]

    input_bounds = convert_input_bounds(cast(And, input_bounds_expr), n_inputs, backend)
    output_constrs = convert_output_constrs(
        cast(Or, output_constrs_expr), n_outputs, n_inputs, backend
    )

    return input_bounds, output_constrs


def _process_or_expr(
    or_expr: Or, n_inputs: int, n_outputs: int, backend: Backend
) -> list[tuple[TensorLike, list[TensorLike]]]:
    """Process one OR expression into property tuples.

    :param or_expr: OR expression node.
    :param n_inputs: Number of input variables.
    :param n_outputs: Number of output variables.
    :param backend: Backend instance.
    :return: List of ``(input_bounds, output_constraints)`` tuples.
    """
    if not isinstance(or_expr, Or):
        raise ValueError(f"Expected Or expression, got {type(or_expr)}")

    convert = partial(
        convert_one_property,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        backend=backend,
    )

    return [convert(cast(And, arg)) for arg in or_expr.args]


def convert_to_tensor(
    expr: And,
    n_inputs: int,
    n_outputs: int,
    backend: Backend,
    verbose: bool = False,
    use_parallel: bool = True,
) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
    """Convert expression tree to nested tensor structure.

    Expression hierarchy:
    - Level 1 (AND): Property groups (all must be true)
    - Level 2 (OR): Properties in group (one must be true)
    - Level 3 (AND): Input bounds and output constraints

    :param expr: Root AND expression.

    :param n_inputs: Number of input variables.

    :param n_outputs: Number of output variables.

    :param backend: Backend instance for tensor operations.

    :param verbose: Print timing information.

    :param use_parallel: Use parallel processing.

    :return: Nested list of (input_bounds, output_constraints) tuples
    """

    def convert_all_properties(
        expr: And, n_inputs: int, n_outputs: int, backend: Backend
    ) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
        convert = partial(_process_or_expr, n_inputs=n_inputs, n_outputs=n_outputs, backend=backend)

        t = time.perf_counter()
        if use_parallel:
            with ThreadPoolExecutor() as executor:
                and_properties = list(executor.map(lambda x: convert(cast(Or, x)), expr.args))
            _logger.debug(f"Convert properties (parallel): {time.perf_counter() - t:.4f}s")
        else:
            and_properties = [convert(cast(Or, arg)) for arg in expr.args]
            _logger.debug(f"Convert properties (sequential): {time.perf_counter() - t:.4f}s")

        return and_properties

    return convert_all_properties(expr, n_inputs, n_outputs, backend)
