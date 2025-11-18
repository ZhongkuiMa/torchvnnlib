"""Type5 VNN-LIB Processor: Top-level OR wrapping complete properties.

Type5: (or (and ...input_constraints... ...output_constraints...))

Top-level OR with complete (input+output) property sets.
Creates separate property for each OR clause.

This processor uses REGEX-BASED PARSING instead of AST for maximum performance.
TYPE5 files have a simple, fixed structure that can be parsed with regex.
"""

__docformat__ = "restructuredtext"
__all__ = ["process_type5"]

import time

import torch
from torch import Tensor

from ._utils import parse_and_block


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

        # Re-add "(and " prefix and parse using shared utility
        prop = parse_and_block('(and ' + part, n_inputs, n_outputs)
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
