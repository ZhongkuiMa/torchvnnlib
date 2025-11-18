"""Type5 VNN-LIB Processor: Top-level OR wrapping complete properties.

Type5: (or (and ...input_constraints... ...output_constraints...))

Top-level OR with complete (input+output) property sets.
Creates separate property for each OR clause.

This processor uses REGEX-BASED PARSING instead of AST for maximum performance.
TYPE5 files have a simple, fixed structure that can be parsed with regex.
"""

__docformat__ = "restructuredtext"
__all__ = ["process_type5"]

import re
import time

import torch
from torch import Tensor


def process_type5(
    lines: list[str], n_inputs: int, n_outputs: int, verbose: bool = False
) -> list[list[tuple[Tensor, list[Tensor]]]]:
    """
    Fast processor for Type5 VNN-LIB files.

    Args:
        lines: Preprocessed assertion lines
        n_inputs: Number of input variables
        n_outputs: Number of output variables
        verbose: Print timing information

    Returns:
        Standardized format: [properties]
        Single AND group containing all OR clause properties
        Each property is a complete (input+output) specification
    """
    t_start = time.perf_counter() if verbose else None

    # Parse to extract OR clauses
    t = time.perf_counter() if verbose else None
    properties = _parse_top_level_or(lines, n_inputs, n_outputs, verbose)
    if verbose and t is not None:
        print(f"  Type5 detection:")
        print(f"    OR clauses (properties): {len(properties)}")
        print(f"    Parsing: {time.perf_counter() - t:.4f}s")

    # Package in expected format: ONE AND group containing all OR properties
    # This matches the structure: (assert (or (and...) (and...) ...))
    and_properties = [properties]

    if verbose and t_start is not None:
        print(f"  Type5 total time: {time.perf_counter() - t_start:.4f}s")

    return and_properties


def _parse_top_level_or(
    lines: list[str], n_inputs: int, n_outputs: int, verbose: bool = False
) -> list[tuple[Tensor, list[Tensor]]]:
    """Parse top-level OR to extract complete properties using ULTRA-FAST PATTERN MATCHING.

    TYPE5 files have structure:
    (assert (or
    (and (>= X_0 v) (<= X_0 v) ... (<= Y_0 v))
    (and (>= X_0 v) (<= X_0 v) ... (>= Y_0 v))
    ))

    CRITICAL OPTIMIZATION: Each property is on a SINGLE LINE starting with "(and".
    No need for complex parsing - just process each "(and" line independently!
    """
    properties = []

    # After preprocessing, TYPE5 files are merged into ONE line.
    # Split by "(and " to extract individual properties - this is fast in Python (C implementation)
    content = ' '.join(lines)
    parts = content.split('(and ')

    # First part is "(assert (or", skip it
    for part in parts[1:]:
        if not part:
            continue

        # Re-add "(and " prefix and parse
        prop = _parse_single_property_line('(and ' + part, n_inputs, n_outputs)
        if prop:
            properties.append(prop)

    return (
        properties
        if properties
        else [
            (
                torch.full((n_inputs, 2), float("nan"), dtype=torch.float64),
                [torch.zeros((1, n_outputs + 1), dtype=torch.float64)],
            )
        ]
    )


def _parse_single_property_line(
    line: str, n_inputs: int, n_outputs: int
) -> tuple[Tensor, list[Tensor]] | None:
    """Parse a single property from one line.

    Line format: (and (>= X_0 v1) (<= X_0 v2) ... (<= Y_0 v))

    ULTRA-FAST: Use simple string split operations, not regex.
    """
    # Initialize bounds tensors
    input_bounds = torch.full((n_inputs, 2), float('nan'), dtype=torch.float64)
    output_constraints = []

    # Split by '(' to get individual constraints
    # This is much faster than regex
    parts = line.split('(')

    for part in parts:
        # Quick filter: must start with >= or <=
        if not part or not (part[0] == '>' or part[0] == '<'):
            continue

        # Split by whitespace to get tokens
        tokens = part.split()
        if len(tokens) < 3:
            continue

        op = tokens[0]  # >= or <=
        var = tokens[1]  # X_0 or Y_0

        # Quick check for variable type (first character)
        if var[0] == 'X':
            # Input variable
            # Extract index: X_123 -> 123
            idx_start = var.find('_')
            if idx_start == -1:
                continue
            idx_str = var[idx_start+1:]
            try:
                idx = int(idx_str)
            except ValueError:
                continue

            if idx >= n_inputs:
                continue

            # Extract value
            value_str = tokens[2].rstrip(')')
            try:
                value = float(value_str)
            except ValueError:
                continue

            # Set bound
            if op[0] == '>':  # >= or >
                input_bounds[idx, 0] = value
            else:  # <= or <
                input_bounds[idx, 1] = value

        elif var[0] == 'Y':
            # Output variable
            # Extract index
            idx_start = var.find('_')
            if idx_start == -1:
                continue
            idx_str = var[idx_start+1:]
            try:
                idx = int(idx_str)
            except ValueError:
                continue

            if idx >= n_outputs:
                continue

            # Extract value
            value_str = tokens[2].rstrip(')')
            try:
                value = float(value_str)
            except ValueError:
                continue

            # Create output constraint
            # Format: [b, a0, a1, ..., a_{n_outputs-1}] representing b + a0*Y_0 + a1*Y_1 + ... >= 0
            # Index 0 is bias (b), indices 1...n_outputs are coefficients
            constr_row = torch.zeros((1, n_outputs + 1), dtype=torch.float64)

            if op[0] == '>':  # >= or >
                # Y_i >= v
                # Convert to standard form: v - Y_i <= 0, then negate to get -v + Y_i >= 0
                # Bias (index 0): -v
                # Coefficient for Y_i (index i+1): +1
                constr_row[0, 0] = -value
                constr_row[0, idx + 1] = 1.0
            else:  # <= or <
                # Y_i <= v
                # Convert to standard form: Y_i - v <= 0, then negate to get v - Y_i >= 0
                # Bias (index 0): +v
                # Coefficient for Y_i (index i+1): -1
                constr_row[0, 0] = value
                constr_row[0, idx + 1] = -1.0

            output_constraints.append(constr_row)

    # If no output constraints, add zero constraint
    if not output_constraints:
        output_constraints = [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]

    return (input_bounds, output_constraints)
