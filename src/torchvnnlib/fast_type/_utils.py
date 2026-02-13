"""VNN-LIB Type Detection Module.

This module detects the structural type of VNN-LIB files to enable
type-specific fast-path processing.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "INPUT_BOUND_INNER_PATTERN",
    "OUTPUT_BOUND_INNER_PATTERN",
    "OUTPUT_CONSTRAINT_INNER_PATTERN",
    "SIMPLE_INPUT_BOUND_PATTERN",
    "SIMPLE_OUTPUT_BOUND_PATTERN",
    "SIMPLE_OUTPUT_CONSTRAINT_PATTERN",
    "VNNLIBType",
    "convert_simple_input_bounds",
    "parse_and_block",
    "parse_dual_or_blocks",
    "parse_input_bounds_block",
    "parse_input_or_block",
    "parse_or_block",
    "parse_output_and_block",
    "parse_output_or_block",
]


import re
from enum import Enum

from torchvnnlib._backend import Backend, TensorLike


class VNNLIBType(Enum):
    """VNN-LIB file types based on structure."""

    TYPE1 = "Type1: (and ...input_constraints... ...output_constraints...)"
    TYPE2 = "Type2: (and ...input_constraints... (or ...output_constraints...))"
    TYPE3 = "Type3: (and (or ...input_constraints...) ...output_constraints...)"
    TYPE4 = "Type4: (and (or ...input_constraints...) (or ...output_constraints...))"
    TYPE5 = "Type5: (or (and ...input_constraints... ...output_constraints...))"
    COMPLEX = "Complex: Requires AST-based processing"


TOP_LEVEL_OR_PATTERN = re.compile(r"^\s*\(\s*assert\s+\(\s*or\s+", re.IGNORECASE)
NESTED_OR_PATTERN = re.compile(r"\(\s*or\s+", re.IGNORECASE)
NESTED_AND_PATTERN = re.compile(r"\(\s*and\s+", re.IGNORECASE)

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


def _categorize_bounds_by_operator(
    simple_bounds: list[tuple],
) -> tuple[list[int], list[float], list[int], list[float], list[int], list[float]]:
    """Categorize bounds by operator type (<=, >=, =).

    :param simple_bounds: List of tuples (op, var_prefix, idx, value)
    :return: Tuple of (leq_indices, leq_values, geq_indices, geq_values, eq_indices, eq_values)
    """
    n_bounds = len(simple_bounds)
    leq_indices = [0] * n_bounds
    leq_values = [0.0] * n_bounds
    geq_indices = [0] * n_bounds
    geq_values = [0.0] * n_bounds
    eq_indices = [0] * n_bounds
    eq_values = [0.0] * n_bounds
    leq_count = geq_count = eq_count = 0

    for op, var_type, idx, value in simple_bounds:
        if var_type != "X_":
            continue

        if op == "<=":
            leq_indices[leq_count] = idx
            leq_values[leq_count] = value
            leq_count += 1
        elif op == ">=":
            geq_indices[geq_count] = idx
            geq_values[geq_count] = value
            geq_count += 1
        elif op == "=":
            eq_indices[eq_count] = idx
            eq_values[eq_count] = value
            eq_count += 1

    return (
        leq_indices[:leq_count],
        leq_values[:leq_count],
        geq_indices[:geq_count],
        geq_values[:geq_count],
        eq_indices[:eq_count],
        eq_values[:eq_count],
    )


def _apply_bound_constraints(
    input_bounds: TensorLike,
    leq_indices: list[int],
    leq_values: list[float],
    geq_indices: list[int],
    geq_values: list[float],
    eq_indices: list[int],
    eq_values: list[float],
    backend: Backend,
) -> None:
    """Apply bound constraints to input_bounds tensor in-place.

    :param input_bounds: Tensor to update in-place
    :param leq_indices: Indices for <= bounds
    :param leq_values: Values for <= bounds
    :param geq_indices: Indices for >= bounds
    :param geq_values: Values for >= bounds
    :param eq_indices: Indices for = bounds
    :param eq_values: Values for = bounds
    :param backend: Backend instance
    """
    if leq_indices:
        leq_val_array = backend.tensor(leq_values, dtype="float64")
        for i, idx in enumerate(leq_indices):
            input_bounds[idx, 1] = leq_val_array[i]

    if geq_indices:
        geq_val_array = backend.tensor(geq_values, dtype="float64")
        for i, idx in enumerate(geq_indices):
            input_bounds[idx, 0] = geq_val_array[i]

    if eq_indices:
        eq_val_array = backend.tensor(eq_values, dtype="float64")
        for i, idx in enumerate(eq_indices):
            input_bounds[idx, 0] = eq_val_array[i]
            input_bounds[idx, 1] = eq_val_array[i]


def convert_simple_input_bounds(
    simple_bounds: list[tuple], n_inputs: int, backend: Backend
) -> TensorLike:
    """Convert simple input bounds to tensor format.

    :param simple_bounds: List of tuples (op, var_prefix, idx, value) from regex parsing
    :param n_inputs: Number of input variables
    :param backend: Backend instance for tensor operations
    :return: Tensor of shape (n_inputs, 2) where [:, 0] is lower bounds and [:, 1] is upper bounds
    """
    input_bounds = backend.full((n_inputs, 2), float("nan"), dtype="float64")

    leq_indices, leq_values, geq_indices, geq_values, eq_indices, eq_values = (
        _categorize_bounds_by_operator(simple_bounds)
    )

    _apply_bound_constraints(
        input_bounds,
        leq_indices,
        leq_values,
        geq_indices,
        geq_values,
        eq_indices,
        eq_values,
        backend,
    )

    if backend.isnan(input_bounds).any():
        nan_indices = backend.where(backend.isnan(input_bounds))
        nan_list = list(zip(nan_indices[0].tolist(), nan_indices[1].tolist(), strict=False))
        raise ValueError(f"Missing input bounds at indices: {nan_list}")

    return input_bounds


def parse_input_or_block(
    or_block_lines: list[str], n_inputs: int, backend: Backend
) -> list[TensorLike]:
    """Parse OR block to extract input bounds using direct pattern matching.

    :param or_block_lines: Lines containing OR block expressions
    :param n_inputs: Number of input variables
    :param backend: Backend instance for tensor operations
    :return: List of input bound tensors
    """
    if not or_block_lines:
        return [backend.full((n_inputs, 2), float("nan"), dtype="float64")]

    content = " ".join(or_block_lines)
    parts = content.split("(and ")

    input_bounds_list = []

    for part in parts[1:]:
        if not part.strip():
            continue

        bounds = parse_input_bounds_block(part, n_inputs, backend)
        if not backend.isnan(bounds).all():
            input_bounds_list.append(bounds)

    return input_bounds_list or [backend.full((n_inputs, 2), float("nan"), dtype="float64")]


def parse_output_or_block(
    or_block_lines: list[str], n_inputs: int, n_outputs: int, backend: Backend
) -> list[TensorLike]:
    """Parse OR block to extract output constraints using direct pattern matching.

    :param or_block_lines: Lines containing OR block expressions
    :param n_inputs: Number of input variables
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
    :return: List of output constraint tensors
    """
    if not or_block_lines:
        return [backend.zeros((1, n_outputs + 1), dtype="float64")]

    content = " ".join(or_block_lines)
    parts = content.split("(and ")

    output_constrs = []

    for part in parts[1:]:
        if not part.strip():
            continue

        constr = parse_output_and_block(part, n_outputs, backend)
        if constr is not None and constr.shape[0] > 0:
            output_constrs.append(constr)

    return output_constrs or [backend.zeros((1, n_outputs + 1), dtype="float64")]


parse_or_block = parse_output_or_block


def parse_and_block(
    block: str, n_inputs: int, n_outputs: int, backend: Backend
) -> tuple[TensorLike, list[TensorLike]]:
    """Parse a single AND block to extract both input bounds and output constraints.

    :param block: String containing AND block content
    :param n_inputs: Number of input variables
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
    :return: Tuple of (input_bounds, output_constraints_list)
    """
    input_bounds = parse_input_bounds_block(block, n_inputs, backend)

    output_constraints = []
    matches = OUTPUT_BOUND_INNER_PATTERN.findall(block)
    for match in matches:
        op, _var_prefix, idx, value = match
        idx = int(idx)
        value = float(value)

        if idx >= n_outputs:
            continue

        constr_row = backend.zeros((1, n_outputs + 1), dtype="float64")

        if op == ">=":
            constr_row[0, 0] = -value
            constr_row[0, idx + 1] = 1.0
            output_constraints.append(constr_row)
        elif op == "<=":
            constr_row[0, 0] = value
            constr_row[0, idx + 1] = -1.0
            output_constraints.append(constr_row)
        elif op == "=":
            constr_row1 = backend.zeros((1, n_outputs + 1), dtype="float64")
            constr_row1[0, 0] = -value
            constr_row1[0, idx + 1] = 1.0
            output_constraints.append(constr_row1)

            constr_row2 = backend.zeros((1, n_outputs + 1), dtype="float64")
            constr_row2[0, 0] = value
            constr_row2[0, idx + 1] = -1.0
            output_constraints.append(constr_row2)

    if not output_constraints:
        output_constraints = [backend.zeros((1, n_outputs + 1), dtype="float64")]

    return (input_bounds, output_constraints)


def parse_input_bounds_block(block: str, n_inputs: int, backend: Backend) -> TensorLike:
    """Parse a single AND block to extract input bounds.

    :param block: String containing AND block content
    :param n_inputs: Number of input variables
    :param backend: Backend instance for tensor operations
    :return: Tensor of shape (n_inputs, 2) with lower and upper bounds
    """
    input_bounds = backend.full((n_inputs, 2), float("nan"), dtype="float64")

    matches = INPUT_BOUND_INNER_PATTERN.findall(block)
    for match in matches:
        op, _var_prefix, idx, value = match
        idx = int(idx)
        value = float(value)

        if idx >= n_inputs:
            continue

        if op == "<=":
            input_bounds[idx, 1] = value
        elif op == ">=":
            input_bounds[idx, 0] = value
        elif op == "=":
            input_bounds[idx, 0] = value
            input_bounds[idx, 1] = value

    return input_bounds


def parse_output_and_block(block: str, n_outputs: int, backend: Backend) -> TensorLike:
    """Parse a single AND block to extract output constraints.

    :param block: String containing AND block content
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
    :return: Tensor of constraints (rows x (n_outputs + 1))
    """
    constraints = []

    matches = OUTPUT_CONSTRAINT_INNER_PATTERN.findall(block)
    for match in matches:
        op, _var_prefix1, idx1, _var_prefix2, idx2 = match
        idx1, idx2 = int(idx1), int(idx2)

        constr_row = backend.zeros((n_outputs + 1,), dtype="float64")

        if op == "<=":
            constr_row[idx1 + 1] = -1.0
            constr_row[idx2 + 1] = 1.0
        elif op == ">=":
            constr_row[idx1 + 1] = 1.0
            constr_row[idx2 + 1] = -1.0

        constraints.append(constr_row)

    matches = OUTPUT_BOUND_INNER_PATTERN.findall(block)
    for match in matches:
        op, _var_prefix, idx, value = match
        idx = int(idx)
        value = float(value)

        if op == "<=":
            constr_row = backend.zeros((n_outputs + 1,), dtype="float64")
            constr_row[0] = value
            constr_row[idx + 1] = -1.0
            constraints.append(constr_row)
        elif op == ">=":
            constr_row = backend.zeros((n_outputs + 1,), dtype="float64")
            constr_row[0] = -value
            constr_row[idx + 1] = 1.0
            constraints.append(constr_row)
        elif op == "=":
            constr_row1 = backend.zeros((n_outputs + 1,), dtype="float64")
            constr_row1[0] = -value
            constr_row1[idx + 1] = 1.0
            constraints.append(constr_row1)

            constr_row2 = backend.zeros((n_outputs + 1,), dtype="float64")
            constr_row2[0] = value
            constr_row2[idx + 1] = -1.0
            constraints.append(constr_row2)

    if constraints:
        return backend.stack(constraints, axis=0)
    return backend.zeros((1, n_outputs + 1), dtype="float64")


def parse_dual_or_blocks(
    lines: list[str], n_inputs: int, n_outputs: int, backend: Backend
) -> tuple[list[TensorLike], list[TensorLike]]:
    """Parse two OR blocks for inputs and outputs using direct pattern matching.

    :param lines: Preprocessed assertion lines
    :param n_inputs: Number of input variables
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
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
