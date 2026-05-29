"""Regex patterns and parsing helpers shared by ``fast_type`` processors."""

__docformat__ = "restructuredtext"
__all__ = [
    "INPUT_BOUND_INNER_PATTERN",
    "OUTPUT_BOUND_INNER_PATTERN",
    "OUTPUT_CONSTRAINT_INNER_PATTERN",
    "SIMPLE_INPUT_BOUND_PATTERN",
    "SIMPLE_OUTPUT_BOUND_PATTERN",
    "SIMPLE_OUTPUT_CONSTRAINT_PATTERN",
    "convert_simple_input_bounds",
    "iter_and_blocks",
    "parse_and_block",
    "parse_dual_or_blocks",
    "parse_input_bounds_block",
    "parse_input_or_block",
    "parse_output_and_block",
    "parse_output_or_block",
]


import re
from collections.abc import Iterator

from torchvnnlib._backend import Backend, TensorLike
from torchvnnlib._constraint_row import (
    apply_input_bound,
    normalize_neg_zero,
    validate_input_bounds,
    write_compare_row,
    write_value_bound_row,
    write_value_bound_rows_eq,
)


def iter_and_blocks(lines: list[str], *, with_prefix: bool = False) -> Iterator[str]:
    """Yield ``(and ...)`` blocks parsed out of a top-level ``(or ...)`` string.

    Joins ``lines`` into one string, splits on the literal ``"(and "`` marker,
    drops the leading partition (everything before the first ``(and``), and
    skips blank fragments. The three OR-walker call sites (input-only OR,
    output-only OR, and top-level property OR) share this scaffolding.

    :param lines: Lines containing the OR block expressions.
    :param with_prefix: When True, prepend ``"(and "`` back onto each yielded
        fragment (callers that re-parse via :func:`parse_and_block` need the
        marker; callers that re-parse per-axis with
        :func:`parse_input_bounds_block` or :func:`parse_output_and_block`
        do not).
    :yield: One fragment per AND clause.
    """
    content = " ".join(lines)
    for part in content.split("(and ")[1:]:
        if not part.strip():
            continue
        yield ("(and " + part) if with_prefix else part


SIMPLE_INPUT_BOUND_PATTERN = re.compile(
    r"^\s*\(\s*assert\s+\(\s*(<=|>=|=)\s+(X_)(\d+)\s+([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\)\s*\)\s*$"
)

SIMPLE_OUTPUT_CONSTRAINT_PATTERN = re.compile(
    r"^\s*\(\s*assert\s+\(\s*(<=|>=)\s+(Y_)(\d+)\s+(Y_)(\d+)\s*\)\s*\)\s*$"
)

SIMPLE_OUTPUT_BOUND_PATTERN = re.compile(
    r"^\s*\(\s*assert\s+\(\s*(<=|>=|=)\s+(Y_)(\d+)\s+([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\)\s*\)\s*$"
)

OUTPUT_CONSTRAINT_INNER_PATTERN = re.compile(r"\(\s*(<=|>=)\s+(Y_)(\d+)\s+(Y_)(\d+)\s*\)")

OUTPUT_BOUND_INNER_PATTERN = re.compile(
    r"\(\s*(<=|>=|=)\s+(Y_)(\d+)\s+([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\)"
)

INPUT_BOUND_INNER_PATTERN = re.compile(
    r"\(\s*(<=|>=|=)\s+(X_)(\d+)\s+([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\)"
)


def convert_simple_input_bounds(
    simple_bounds: list[tuple], n_inputs: int, backend: Backend
) -> TensorLike:
    """Convert simple input bounds to tensor format.

    :param simple_bounds: Tuples ``(op, var_prefix, idx, value)`` from regex parsing.
    :param n_inputs: Number of input variables.
    :param backend: Backend instance for tensor operations.
    :return: Tensor of shape ``(n_inputs, 2)``; column 0 lower, column 1 upper.
    :raises ValueError: If any bound slot remains NaN after processing.
    """
    input_bounds = backend.full((n_inputs, 2), float("nan"), dtype="float64")

    for op, var_prefix, idx, value in simple_bounds:
        if var_prefix != "X_":
            continue
        apply_input_bound(input_bounds, idx, op, float(value))

    validate_input_bounds(input_bounds, backend)

    return normalize_neg_zero(input_bounds)


def parse_input_or_block(
    or_block_lines: list[str], n_inputs: int, backend: Backend
) -> list[TensorLike]:
    """Parse OR block to extract input bounds using direct pattern matching.

    :param or_block_lines: Lines containing OR block expressions.

    :param n_inputs: Number of input variables.

    :param backend: Backend instance for tensor operations.

    :return: List of input bound tensors
    """
    if not or_block_lines:
        return [backend.full((n_inputs, 2), float("nan"), dtype="float64")]

    input_bounds_list = []
    for part in iter_and_blocks(or_block_lines):
        bounds = parse_input_bounds_block(part, n_inputs, backend)
        if not backend.isnan(bounds).all():
            input_bounds_list.append(bounds)

    return input_bounds_list or [backend.full((n_inputs, 2), float("nan"), dtype="float64")]


def parse_output_or_block(
    or_block_lines: list[str], n_inputs: int, n_outputs: int, backend: Backend
) -> list[TensorLike]:
    """Parse OR block to extract output constraints using direct pattern matching.

    :param or_block_lines: Lines containing OR block expressions.

    :param n_inputs: Number of input variables.

    :param n_outputs: Number of output variables.

    :param backend: Backend instance for tensor operations.

    :return: List of output constraint tensors
    """
    if not or_block_lines:
        return [backend.zeros((1, n_outputs + 1), dtype="float64")]

    output_constrs = []
    for part in iter_and_blocks(or_block_lines):
        constr = parse_output_and_block(part, n_outputs, backend)
        if constr is not None and constr.shape[0] > 0:
            output_constrs.append(constr)

    return output_constrs or [backend.zeros((1, n_outputs + 1), dtype="float64")]


def parse_and_block(
    block: str, n_inputs: int, n_outputs: int, backend: Backend
) -> tuple[TensorLike, list[TensorLike]]:
    """Parse a single AND block to extract both input bounds and output constraints.

    :param block: String containing AND block content.

    :param n_inputs: Number of input variables.

    :param n_outputs: Number of output variables.

    :param backend: Backend instance for tensor operations.

    :return: Tuple of (input_bounds, output_constraints_list)
    """
    input_bounds = parse_input_bounds_block(block, n_inputs, backend)

    output_constraints: list[TensorLike] = []
    for op, _var_prefix, idx_str, value_str in OUTPUT_BOUND_INNER_PATTERN.findall(block):
        idx = int(idx_str)
        if idx >= n_outputs:
            continue
        value = float(value_str)

        if op == "=":
            row_geq = backend.zeros((1, n_outputs + 1), dtype="float64")
            row_leq = backend.zeros((1, n_outputs + 1), dtype="float64")
            write_value_bound_rows_eq(row_geq[0], row_leq[0], idx, value)
            normalize_neg_zero(row_geq)
            normalize_neg_zero(row_leq)
            output_constraints.append(row_geq)
            output_constraints.append(row_leq)
        else:
            row = backend.zeros((1, n_outputs + 1), dtype="float64")
            write_value_bound_row(row[0], idx, op, value)
            normalize_neg_zero(row)
            output_constraints.append(row)

    if not output_constraints:
        output_constraints = [backend.zeros((1, n_outputs + 1), dtype="float64")]

    return (input_bounds, output_constraints)


def parse_input_bounds_block(block: str, n_inputs: int, backend: Backend) -> TensorLike:
    """Parse a single AND block to extract input bounds.

    :param block: String containing AND block content.

    :param n_inputs: Number of input variables.

    :param backend: Backend instance for tensor operations.

    :return: Tensor of shape (n_inputs, 2) with lower and upper bounds
    """
    input_bounds = backend.full((n_inputs, 2), float("nan"), dtype="float64")

    matches = INPUT_BOUND_INNER_PATTERN.findall(block)
    for match in matches:
        op, _var_prefix, idx_str, value_str = match
        idx = int(idx_str)
        if idx >= n_inputs:
            continue
        apply_input_bound(input_bounds, idx, op, float(value_str))

    return input_bounds


def parse_output_and_block(block: str, n_outputs: int, backend: Backend) -> TensorLike:
    """Parse a single AND block to extract output constraints.

    :param block: String containing AND block content.

    :param n_outputs: Number of output variables.

    :param backend: Backend instance for tensor operations.

    :return: Tensor of constraints (rows x (n_outputs + 1))
    """
    constraints: list[TensorLike] = []

    for op, _vp1, idx1_str, _vp2, idx2_str in OUTPUT_CONSTRAINT_INNER_PATTERN.findall(block):
        row = backend.zeros((n_outputs + 1,), dtype="float64")
        write_compare_row(row, int(idx1_str), int(idx2_str), op)
        constraints.append(row)

    for op, _var_prefix, idx_str, value_str in OUTPUT_BOUND_INNER_PATTERN.findall(block):
        idx = int(idx_str)
        value = float(value_str)
        if op == "=":
            row_geq = backend.zeros((n_outputs + 1,), dtype="float64")
            row_leq = backend.zeros((n_outputs + 1,), dtype="float64")
            write_value_bound_rows_eq(row_geq, row_leq, idx, value)
            constraints.append(row_geq)
            constraints.append(row_leq)
        else:
            row = backend.zeros((n_outputs + 1,), dtype="float64")
            write_value_bound_row(row, idx, op, value)
            constraints.append(row)

    if constraints:
        stacked = backend.stack(constraints, axis=0)
        return normalize_neg_zero(stacked)
    return backend.zeros((1, n_outputs + 1), dtype="float64")


def parse_dual_or_blocks(
    lines: list[str], n_inputs: int, n_outputs: int, backend: Backend
) -> tuple[list[TensorLike], list[TensorLike]]:
    """Parse two OR blocks for inputs and outputs using direct pattern matching.

    :param lines: Preprocessed assertion lines.

    :param n_inputs: Number of input variables.

    :param n_outputs: Number of output variables.

    :param backend: Backend instance for tensor operations.

    :return: Tuple of (input_bounds_list, output_constrs_list)
    """
    content = " ".join(lines)
    or_parts = content.split("(or ")

    if len(or_parts) < 3:
        return (
            [backend.full((n_inputs, 2), float("nan"), dtype="float64")],
            [backend.zeros((1, n_outputs + 1), dtype="float64")],
        )

    input_or_lines = ["(assert (or " + or_parts[1]]
    output_or_lines = ["(assert (or " + or_parts[2]]

    input_bounds_list = parse_input_or_block(input_or_lines, n_inputs, backend)
    output_constrs_list = parse_output_or_block(output_or_lines, n_inputs, n_outputs, backend)

    return input_bounds_list, output_constrs_list
