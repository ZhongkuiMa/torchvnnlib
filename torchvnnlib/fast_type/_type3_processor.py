"""Type3 VNN-LIB Processor: Simple inputs + OR outputs.

Type3: (and ...input_constraints... (or ...output_constraints...))

Simple input bounds with OR structure for outputs (multiple output options).
Pattern: simple_in=True, simple_out=False, or_in=False, or_out=True
"""

__docformat__ = "restructuredtext"
__all__ = ["process_type3"]

import time

import torch
from torch import Tensor

from ._utils import convert_simple_input_bounds, parse_or_block


def process_type3(
    lines: list[str],
    n_inputs: int,
    n_outputs: int,
    verbose: bool = False,
    parsed_data: dict | None = None,
) -> list[list[tuple[Tensor, list[Tensor]]]]:
    """
    Fast processor for Type3 VNN-LIB files.

    Type3: Simple input bounds + OR(AND) output blocks
    Pattern: (and ...input_bounds... (or (and Y_...) (and Y_...) ...))

    Args:
        lines: Preprocessed assertion lines
        n_inputs: Number of input variables
        n_outputs: Number of output variables
        verbose: Print timing information
        parsed_data: Pre-parsed data from parse_simple_patterns() (optional)

    Returns:
        Standardized format: [[(input_bounds, [output_constr1, output_constr2, ...])]]
        Single AND group with one property containing multiple OR output options
    """
    if verbose:
        t_start = time.perf_counter()

    # Use pre-parsed data if available, otherwise parse now
    if parsed_data is None:
        from ._fast_type_detect import parse_simple_patterns

        t = time.perf_counter()
        parsed_data = parse_simple_patterns(lines, verbose=False)
        if verbose:
            print(f"  Type3 parsing: {time.perf_counter() - t:.4f}s")

    simple_input_bounds = parsed_data["simple_input_bounds"]
    or_block_lines = parsed_data["complex_lines"]  # OR blocks are complex

    if verbose:
        print(f"  Type3 processing:")
        print(f"    Simple input bounds: {len(simple_input_bounds)}")
        print(f"    OR block lines: {len(or_block_lines)}")

    # Convert simple input bounds to tensor
    t = time.perf_counter()
    input_bounds = convert_simple_input_bounds(simple_input_bounds, n_inputs)
    if verbose:
        print(f"    Input bounds conversion: {time.perf_counter() - t:.4f}s")

    # Parse OR block for output constraints
    t = time.perf_counter()
    output_constrs = parse_or_block(or_block_lines, n_inputs, n_outputs)
    if verbose:
        print(f"    OR block parsing: {time.perf_counter() - t:.4f}s")
        print(f"    Extracted {len(output_constrs)} output constraint options")

    # Package in expected format: [[(input_bounds, [output_constrs])]]
    and_properties = [[(input_bounds, output_constrs)]]

    if verbose:
        print(f"  Type3 total time: {time.perf_counter() - t_start:.4f}s")

    return and_properties
