"""VNN-LIB Type Detection Module.

This module detects the structural type of VNN-LIB files to enable
type-specific fast-path processing.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "VNNLIBType",
    "SIMPLE_INPUT_BOUND_PATTERN",
    "SIMPLE_OUTPUT_CONSTRAINT_PATTERN",
    "SIMPLE_OUTPUT_BOUND_PATTERN",
    "convert_simple_input_bounds",
    "parse_or_block",
]


import re
from enum import Enum

import torch
from torch import Tensor


class VNNLIBType(Enum):
    """VNN-LIB file types based on structure."""

    TYPE1 = "Type1: (and ...input_constraints... ...output_constraints...)"
    TYPE2 = "Type2: (and ...input_constraints... (or ...output_constraints...))"
    TYPE3 = "Type3: (and (or ...input_constraints...) ...output_constraints...)"
    TYPE4 = "Type4: (and (or ...input_constraints...) (or ...output_constraints...))"
    TYPE5 = "Type5: (or (and ...input_constraints... ...output_constraints...))"
    COMPLEX = "Complex: Requires AST-based processing"


# Pattern to detect top-level OR/AND structures
TOP_LEVEL_OR_PATTERN = re.compile(r"^\s*\(\s*assert\s+\(\s*or\s+", re.IGNORECASE)
NESTED_OR_PATTERN = re.compile(r"\(\s*or\s+", re.IGNORECASE)
NESTED_AND_PATTERN = re.compile(r"\(\s*and\s+", re.IGNORECASE)

# Pre-compiled regex patterns for simple expressions
# Pattern for simple input bounds: (assert (<=/>=|= X_5 0.123))
SIMPLE_INPUT_BOUND_PATTERN = re.compile(
    r"^\s*\(\s*assert\s+\(\s*(<=|>=|=)\s+(X_)(\d+)\s+([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\)\s*\)\s*$"
)

# Pattern for simple output constraints with two Y variables: (assert (<=/>=  Y_0 Y_1))
SIMPLE_OUTPUT_CONSTRAINT_PATTERN = re.compile(
    r"^\s*\(\s*assert\s+\(\s*(<=|>=)\s+(Y_)(\d+)\s+(Y_)(\d+)\s*\)\s*\)\s*$"
)

# Pattern for simple output bounds with one Y variable: (assert (<=/>=|= Y_5 0.123))
SIMPLE_OUTPUT_BOUND_PATTERN = re.compile(
    r"^\s*\(\s*assert\s+\(\s*(<=|>=|=)\s+(Y_)(\d+)\s+([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\)\s*\)\s*$"
)


def convert_simple_input_bounds(
    simple_bounds: list[tuple], n_inputs: int
) -> Tensor:
    """Convert simple input bounds to tensor format.

    This is a shared utility used by multiple type processors to convert
    parsed input bounds into the standard tensor representation.

    Args:
        simple_bounds: List of tuples (op, var_prefix, idx, value) from regex parsing
        n_inputs: Number of input variables

    Returns:
        Tensor of shape (n_inputs, 2) where [:, 0] is lower bounds and [:, 1] is upper bounds

    Raises:
        ValueError: If any input variable is missing bounds
    """
    input_bounds = torch.full((n_inputs, 2), float("nan"), dtype=torch.float64)

    # Pre-allocate arrays for batch assignment
    n_bounds = len(simple_bounds)
    leq_indices = [0] * n_bounds
    leq_values = [0.0] * n_bounds
    geq_indices = [0] * n_bounds
    geq_values = [0.0] * n_bounds
    eq_indices = [0] * n_bounds
    eq_values = [0.0] * n_bounds

    leq_count = geq_count = eq_count = 0

    # Single pass collection
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

    # Batch tensor assignments
    if leq_count > 0:
        input_bounds[leq_indices[:leq_count], 1] = torch.tensor(
            leq_values[:leq_count], dtype=torch.float64
        )
    if geq_count > 0:
        input_bounds[geq_indices[:geq_count], 0] = torch.tensor(
            geq_values[:geq_count], dtype=torch.float64
        )
    if eq_count > 0:
        eq_vals = torch.tensor(eq_values[:eq_count], dtype=torch.float64)
        eq_idx_tensor = torch.tensor(eq_indices[:eq_count], dtype=torch.long)
        input_bounds[eq_idx_tensor, 0] = eq_vals
        input_bounds[eq_idx_tensor, 1] = eq_vals

    # Validation
    if torch.isnan(input_bounds).any():
        nan_indices = torch.where(torch.isnan(input_bounds))
        raise ValueError(
            f"Missing input bounds at indices: {list(zip(nan_indices[0].tolist(), nan_indices[1].tolist()))}"
        )

    return input_bounds


def parse_or_block(
    or_block_lines: list[str], n_inputs: int, n_outputs: int
) -> list[Tensor]:
    """Parse OR block to extract output constraints.

    Shared utility for Type2 and Type3 processors to parse OR blocks containing
    output constraints.

    Args:
        or_block_lines: Lines containing OR block expressions
        n_inputs: Number of input variables
        n_outputs: Number of output variables

    Returns:
        List of output constraint tensors
    """
    # Import here to avoid circular dependency
    from .._to_tensor import convert_and_output_constrs
    from ..ast import tokenize, parse, optimize, And, Or

    if not or_block_lines:
        return [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]

    # Parse OR block using standard AST path
    tokens_list = tokenize(or_block_lines, verbose=False, use_parallel=False)
    expr = parse(tokens_list, verbose=False, use_parallel=False)
    expr = optimize(expr, verbose=False, use_parallel=False)

    def _convert_to_constraint(arg):
        """Convert expression arg to constraint, wrapping in And if needed."""
        if isinstance(arg, And):
            return convert_and_output_constrs(arg, n_outputs, n_inputs)
        else:
            return convert_and_output_constrs(And([arg]), n_outputs, n_inputs)

    output_constrs = []

    if isinstance(expr, Or):
        # Direct OR expression
        for or_arg in expr.args:
            output_constrs.append(_convert_to_constraint(or_arg))
    elif isinstance(expr, And):
        # OR might be nested inside AND
        for arg in expr.args:
            if isinstance(arg, Or):
                for or_arg in arg.args:
                    output_constrs.append(_convert_to_constraint(or_arg))
            else:
                output_constrs.append(_convert_to_constraint(arg))
    else:
        # Single constraint
        output_constrs.append(_convert_to_constraint(expr))

    return (
        output_constrs
        if output_constrs
        else [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]
    )
