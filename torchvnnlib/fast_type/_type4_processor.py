"""Type4 VNN-LIB Processor: OR inputs + OR outputs.

Type4: (and (or ...input_constraints...) (or ...output_constraints...))

Both inputs and outputs have OR structures (Cartesian product).
Creates properties for each combination of input×output region.
"""

__docformat__ = "restructuredtext"
__all__ = ["process_type4"]

import time

import torch
from torch import Tensor

from ._utils import parse_input_bounds_block, OUTPUT_CONSTRAINT_INNER_PATTERN, OUTPUT_BOUND_INNER_PATTERN


def process_type4(
    lines: list[str], n_inputs: int, n_outputs: int, verbose: bool = False
) -> list[list[tuple[Tensor, list[Tensor]]]]:
    """
    Fast processor for Type4 VNN-LIB files.

    Args:
        lines: Preprocessed assertion lines
        n_inputs: Number of input variables
        n_outputs: Number of output variables
        verbose: Print timing information

    Returns:
        Standardized format: [[property1, property2, ...]]
        Single AND group containing multiple OR properties (one per input region)
        Each property has the same output constraint options
    """
    if verbose:
        t_start = time.perf_counter()

    # Parse all lines to find two OR blocks
    t = time.perf_counter()
    input_bounds_list, output_constrs_list = _parse_dual_or_blocks(
        lines, n_inputs, n_outputs, verbose
    )
    if verbose:
        print(f"  Type4 detection:")
        print(f"    Input OR groups: {len(input_bounds_list)}")
        print(f"    Output OR options: {len(output_constrs_list)}")
        print(f"    Parsing: {time.perf_counter() - t:.4f}s")

    # Create one AND group with multiple OR properties
    # Each property is one input region paired with all output constraint options
    or_properties = [
        (input_bounds, output_constrs_list) for input_bounds in input_bounds_list
    ]
    and_properties = [or_properties]  # Single AND group

    if verbose:
        print(f"  Type4 total time: {time.perf_counter() - t_start:.4f}s")

    return and_properties


def _parse_dual_or_blocks(
    lines: list[str], n_inputs: int, n_outputs: int, verbose: bool = False
) -> tuple[list[Tensor], list[Tensor]]:
    """Parse two OR blocks for inputs and outputs using direct pattern matching.

    Pattern: (assert (and (or (and X_...) ...) (or (and Y_...) ...)))

    Uses direct string parsing instead of AST for better performance.
    """
    # Merge all lines into single string
    content = ' '.join(lines)

    # Find the two OR blocks by splitting strategically
    # Pattern: (assert (and (or ...) (or ...)))
    # We need to find where first OR ends and second OR begins

    # Split by '(or ' to get OR blocks
    or_parts = content.split('(or ')

    if len(or_parts) < 3:
        # Not enough OR blocks, return defaults
        return (
            [torch.full((n_inputs, 2), float("nan"), dtype=torch.float64)],
            [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]
        )

    # First OR block (inputs) is or_parts[1]
    # Second OR block (outputs) is or_parts[2]
    input_or_content = or_parts[1]
    output_or_content = or_parts[2]

    # Parse input OR block: split by '(and ' to get each AND clause
    input_and_parts = input_or_content.split('(and ')
    input_bounds_list = []

    for part in input_and_parts[1:]:  # Skip first part (before first AND)
        if not part.strip() or 'Y_' in part:  # Skip empty or output blocks
            continue
        bounds = parse_input_bounds_block(part, n_inputs)
        if not torch.isnan(bounds).all():
            input_bounds_list.append(bounds)

    # Parse output OR block: split by '(and ' to get each AND clause
    output_and_parts = output_or_content.split('(and ')
    output_constrs_list = []

    for part in output_and_parts[1:]:  # Skip first part (before first AND)
        if not part.strip() or 'X_' in part:  # Skip empty or input blocks
            continue
        constr = _parse_output_and_block(part, n_outputs)
        if constr is not None and constr.numel() > 0:
            output_constrs_list.append(constr)

    # Fallback: if parsing failed, use defaults
    if not input_bounds_list:
        input_bounds_list = [torch.full((n_inputs, 2), float("nan"), dtype=torch.float64)]
    if not output_constrs_list:
        output_constrs_list = [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]

    return input_bounds_list, output_constrs_list


def _parse_output_and_block(block: str, n_outputs: int) -> Tensor:
    """Parse output AND block - same logic as in _utils.py."""
    constraints = []

    # Pattern 1: Y_i <op> Y_j (two Y variables)
    matches = OUTPUT_CONSTRAINT_INNER_PATTERN.findall(block)
    for match in matches:
        op, var_prefix1, idx1, var_prefix2, idx2 = match
        idx1, idx2 = int(idx1), int(idx2)

        constr_row = torch.zeros(n_outputs + 1, dtype=torch.float64)

        if op == "<=":
            constr_row[idx1 + 1] = -1.0
            constr_row[idx2 + 1] = 1.0
        elif op == ">=":
            constr_row[idx1 + 1] = 1.0
            constr_row[idx2 + 1] = -1.0

        constraints.append(constr_row)

    # Pattern 2: Y_i <op> value
    matches = OUTPUT_BOUND_INNER_PATTERN.findall(block)
    for match in matches:
        op, var_prefix, idx, value = match
        idx = int(idx)
        value = float(value)

        if op == "<=":
            constr_row = torch.zeros(n_outputs + 1, dtype=torch.float64)
            constr_row[0] = -value
            constr_row[idx + 1] = 1.0
            constraints.append(constr_row)
        elif op == ">=":
            constr_row = torch.zeros(n_outputs + 1, dtype=torch.float64)
            constr_row[0] = -value
            constr_row[idx + 1] = 1.0
            constraints.append(constr_row)
        elif op == "=":
            # Two constraints
            constr_row1 = torch.zeros(n_outputs + 1, dtype=torch.float64)
            constr_row1[0] = -value
            constr_row1[idx + 1] = 1.0
            constraints.append(constr_row1)

            constr_row2 = torch.zeros(n_outputs + 1, dtype=torch.float64)
            constr_row2[0] = value
            constr_row2[idx + 1] = -1.0
            constraints.append(constr_row2)

    if constraints:
        return torch.stack(constraints, dim=0)
    else:
        return torch.zeros((1, n_outputs + 1), dtype=torch.float64)
