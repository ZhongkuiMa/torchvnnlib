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
    "INPUT_BOUND_INNER_PATTERN",
    "OUTPUT_CONSTRAINT_INNER_PATTERN",
    "OUTPUT_BOUND_INNER_PATTERN",
    "convert_simple_input_bounds",
    "parse_input_bounds_block",
    "parse_output_and_block",
    "parse_input_or_block",
    "parse_output_or_block",
    "parse_and_block",
    "parse_dual_or_blocks",
    "parse_or_block",  # Backward compatibility
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

# Patterns for parsing constraints within blocks (without assert wrapper)
# For Y_i <op> Y_j constraints
OUTPUT_CONSTRAINT_INNER_PATTERN = re.compile(
    r"\(\s*(<=|>=)\s+(Y_)(\d+)\s+(Y_)(\d+)\s*\)"
)

# For Y_i <op> value constraints
OUTPUT_BOUND_INNER_PATTERN = re.compile(
    r"\(\s*(<=|>=|=)\s+(Y_)(\d+)\s+([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\)"
)

# For X_i <op> value constraints (input bounds within blocks)
INPUT_BOUND_INNER_PATTERN = re.compile(
    r"\(\s*(<=|>=|=)\s+(X_)(\d+)\s+([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\)"
)


def convert_simple_input_bounds(simple_bounds: list[tuple], n_inputs: int) -> Tensor:
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


def parse_input_or_block(or_block_lines: list[str], n_inputs: int) -> list[Tensor]:
    """Parse OR block to extract input bounds using direct pattern matching.

    Pattern: (assert (or (and X_...) (and X_...) ...))
    Used by Type3 processor.

    Args:
        or_block_lines: Lines containing OR block expressions
        n_inputs: Number of input variables

    Returns:
        List of input bound tensors
    """
    if not or_block_lines:
        return [torch.full((n_inputs, 2), float("nan"), dtype=torch.float64)]

    # Merge all lines into single string for processing
    content = ' '.join(or_block_lines)

    # Split by '(and ' to extract individual AND blocks
    parts = content.split('(and ')

    input_bounds_list = []

    # First part is "(assert (or", skip it and process each AND block
    for part in parts[1:]:
        if not part.strip():
            continue

        # Extract input bounds from this AND block
        bounds = parse_input_bounds_block(part, n_inputs)
        if not torch.isnan(bounds).all():
            input_bounds_list.append(bounds)

    return (
        input_bounds_list
        if input_bounds_list
        else [torch.full((n_inputs, 2), float("nan"), dtype=torch.float64)]
    )


def parse_output_or_block(
    or_block_lines: list[str], n_inputs: int, n_outputs: int
) -> list[Tensor]:
    """Parse OR block to extract output constraints using direct pattern matching.

    Pattern: (assert (or (and Y_...) (and Y_...) ...))
    Used by Type2 processor.

    Args:
        or_block_lines: Lines containing OR block expressions
        n_inputs: Number of input variables (unused, kept for API consistency)
        n_outputs: Number of output variables

    Returns:
        List of output constraint tensors
    """
    if not or_block_lines:
        return [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]

    # Merge all lines into single string for processing
    content = ' '.join(or_block_lines)

    # Split by '(and ' to extract individual AND blocks
    parts = content.split('(and ')

    output_constrs = []

    # First part is "(assert (or", skip it and process each AND block
    for part in parts[1:]:
        if not part.strip():
            continue

        # Extract constraints from this AND block
        constr = parse_output_and_block(part, n_outputs)
        if constr is not None and constr.numel() > 0:
            output_constrs.append(constr)

    return (
        output_constrs
        if output_constrs
        else [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]
    )


# Backward compatibility alias
parse_or_block = parse_output_or_block


def parse_and_block(
    block: str, n_inputs: int, n_outputs: int
) -> tuple[Tensor, list[Tensor]]:
    """Parse a single AND block to extract both input bounds and output constraints.

    Handles patterns with both X and Y variables:
    - (<=/>=/= X_i value): Input bounds
    - (<=/>=/= Y_i value): Output bounds

    Used by Type5 processor for complete property blocks.

    Args:
        block: String containing AND block content
        n_inputs: Number of input variables
        n_outputs: Number of output variables

    Returns:
        Tuple of (input_bounds, output_constraints_list)
    """
    # Parse input bounds
    input_bounds = parse_input_bounds_block(block, n_inputs)

    # Parse output constraints (only simple bounds for Type5)
    output_constraints = []
    matches = OUTPUT_BOUND_INNER_PATTERN.findall(block)
    for match in matches:
        op, var_prefix, idx, value = match
        idx = int(idx)
        value = float(value)

        if idx >= n_outputs:
            continue

        constr_row = torch.zeros((1, n_outputs + 1), dtype=torch.float64)

        if op == ">=":
            # Y_i >= value  =>  -value + Y_i >= 0
            constr_row[0, 0] = -value
            constr_row[0, idx + 1] = 1.0
            output_constraints.append(constr_row)
        elif op == "<=":
            # Y_i <= value  =>  value - Y_i >= 0
            constr_row[0, 0] = value
            constr_row[0, idx + 1] = -1.0
            output_constraints.append(constr_row)
        elif op == "=":
            # Y_i = value  =>  two constraints
            constr_row1 = torch.zeros((1, n_outputs + 1), dtype=torch.float64)
            constr_row1[0, 0] = -value
            constr_row1[0, idx + 1] = 1.0
            output_constraints.append(constr_row1)

            constr_row2 = torch.zeros((1, n_outputs + 1), dtype=torch.float64)
            constr_row2[0, 0] = value
            constr_row2[0, idx + 1] = -1.0
            output_constraints.append(constr_row2)

    # If no output constraints, add zero constraint
    if not output_constraints:
        output_constraints = [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]

    return (input_bounds, output_constraints)


def parse_input_bounds_block(block: str, n_inputs: int) -> Tensor:
    """Parse a single AND block to extract input bounds.

    Handles patterns:
    - (<=/>=/= X_i value): X_i <op> constant constraint

    Args:
        block: String containing AND block content
        n_inputs: Number of input variables

    Returns:
        Tensor of shape (n_inputs, 2) with lower and upper bounds
    """
    input_bounds = torch.full((n_inputs, 2), float("nan"), dtype=torch.float64)

    # Find all input bound patterns in the block
    matches = INPUT_BOUND_INNER_PATTERN.findall(block)
    for match in matches:
        op, var_prefix, idx, value = match
        idx = int(idx)
        value = float(value)

        if idx >= n_inputs:
            continue

        if op == "<=":
            # X_idx <= value
            input_bounds[idx, 1] = value
        elif op == ">=":
            # X_idx >= value
            input_bounds[idx, 0] = value
        elif op == "=":
            # X_idx = value
            input_bounds[idx, 0] = value
            input_bounds[idx, 1] = value

    return input_bounds


def parse_output_and_block(block: str, n_outputs: int) -> Tensor:
    """Parse a single AND block to extract output constraints.

    Handles patterns:
    - (<=/>= Y_i Y_j): Y_i <op> Y_j constraint
    - (<=/>=/= Y_i value): Y_i <op> constant constraint

    Args:
        block: String containing AND block content
        n_outputs: Number of output variables

    Returns:
        Tensor of constraints (rows x (n_outputs + 1))
    """
    constraints = []

    # Find all constraint patterns in the block
    # Pattern 1: Y_i <op> Y_j (two Y variables)
    matches = OUTPUT_CONSTRAINT_INNER_PATTERN.findall(block)
    for match in matches:
        op, var_prefix1, idx1, var_prefix2, idx2 = match
        idx1, idx2 = int(idx1), int(idx2)

        # Create constraint row: [bias, coef_Y0, coef_Y1, ..., coef_Yn]
        constr_row = torch.zeros(n_outputs + 1, dtype=torch.float64)

        if op == "<=":
            # Y_idx1 <= Y_idx2  =>  Y_idx1 - Y_idx2 <= 0  =>  -Y_idx1 + Y_idx2 >= 0
            constr_row[idx1 + 1] = -1.0
            constr_row[idx2 + 1] = 1.0
        elif op == ">=":
            # Y_idx1 >= Y_idx2  =>  Y_idx1 - Y_idx2 >= 0
            constr_row[idx1 + 1] = 1.0
            constr_row[idx2 + 1] = -1.0

        constraints.append(constr_row)

    # Pattern 2: Y_i <op> value (Y variable with constant)
    matches = OUTPUT_BOUND_INNER_PATTERN.findall(block)
    for match in matches:
        op, var_prefix, idx, value = match
        idx = int(idx)
        value = float(value)

        if op == "<=":
            # Y_idx <= value  =>  -value + Y_idx >= 0
            constr_row = torch.zeros(n_outputs + 1, dtype=torch.float64)
            constr_row[0] = -value
            constr_row[idx + 1] = 1.0
            constraints.append(constr_row)
        elif op == ">=":
            # Y_idx >= value  =>  -value + Y_idx >= 0
            constr_row = torch.zeros(n_outputs + 1, dtype=torch.float64)
            constr_row[0] = -value
            constr_row[idx + 1] = 1.0
            constraints.append(constr_row)
        elif op == "=":
            # Y_idx = value  =>  two constraints: Y_idx >= value AND Y_idx <= value
            # First: Y_idx >= value  =>  -value + Y_idx >= 0
            constr_row1 = torch.zeros(n_outputs + 1, dtype=torch.float64)
            constr_row1[0] = -value
            constr_row1[idx + 1] = 1.0
            constraints.append(constr_row1)

            # Second: Y_idx <= value  =>  value - Y_idx >= 0
            constr_row2 = torch.zeros(n_outputs + 1, dtype=torch.float64)
            constr_row2[0] = value
            constr_row2[idx + 1] = -1.0
            constraints.append(constr_row2)

    # Stack all constraints into a single tensor
    if constraints:
        return torch.stack(constraints, dim=0)
    else:
        return torch.zeros((1, n_outputs + 1), dtype=torch.float64)


def parse_dual_or_blocks(
    lines: list[str], n_inputs: int, n_outputs: int
) -> tuple[list[Tensor], list[Tensor]]:
    """Parse two OR blocks for inputs and outputs using direct pattern matching.

    Pattern: (assert (and (or (and X_...) ...) (or (and Y_...) ...)))

    Shared utility for Type4 processor. Implemented using parse_input_or_block()
    and parse_output_or_block() for consistency.

    Args:
        lines: Preprocessed assertion lines
        n_inputs: Number of input variables
        n_outputs: Number of output variables

    Returns:
        Tuple of (input_bounds_list, output_constrs_list)
    """
    # Merge all lines into single string
    content = ' '.join(lines)

    # Split by '(or ' to get OR blocks
    or_parts = content.split('(or ')

    if len(or_parts) < 3:
        # Not enough OR blocks, return defaults
        return (
            [torch.full((n_inputs, 2), float("nan"), dtype=torch.float64)],
            [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]
        )

    # Extract the two OR blocks
    # or_parts[0] = "(assert (and "
    # or_parts[1] = first OR block (inputs)
    # or_parts[2] = second OR block (outputs)

    # Reconstruct OR blocks with proper prefix
    input_or_lines = ['(assert (or ' + or_parts[1]]
    output_or_lines = ['(assert (or ' + or_parts[2]]

    # Use the specialized functions
    input_bounds_list = parse_input_or_block(input_or_lines, n_inputs)
    output_constrs_list = parse_output_or_block(output_or_lines, n_inputs, n_outputs)

    return input_bounds_list, output_constrs_list
