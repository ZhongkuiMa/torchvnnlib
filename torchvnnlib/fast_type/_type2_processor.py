"""Type2 VNN-LIB Processor: Simple inputs + OR outputs.

Type2: (and ...input_constraints... (or ...output_constraints...))

Simple input bounds with one OR block containing multiple output constraint options.
Common in benchmarks like CIFAR100.
"""

__docformat__ = "restructuredtext"
__all__ = ["process_type2"]

import time

import torch
from torch import Tensor

from ._utils import convert_simple_input_bounds, parse_or_block


def process_type2(
    lines: list[str],
    n_inputs: int,
    n_outputs: int,
    verbose: bool = False,
    parsed_data: dict | None = None,
) -> list[list[tuple[Tensor, list[Tensor]]]]:
    """
    Fast processor for Type2 VNN-LIB files.

    Type2: Simple input bounds + OR(AND) output blocks
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
    t_start = time.perf_counter() if verbose else None

    # Use pre-parsed data if available, otherwise parse now
    if parsed_data is None:
        from ._fast_type_detect import parse_simple_patterns
        t = time.perf_counter() if verbose else None
        parsed_data = parse_simple_patterns(lines, verbose=False)
        if verbose and t is not None:
            print(f"  Type2 parsing: {time.perf_counter() - t:.4f}s")

    simple_input_bounds = parsed_data['simple_input_bounds']
    or_block_lines = parsed_data['complex_lines']  # OR blocks are complex

    if verbose:
        print(f"  Type2 processing:")
        print(f"    Simple input bounds: {len(simple_input_bounds)}")
        print(f"    OR block lines: {len(or_block_lines)}")

    # Convert simple input bounds to tensor
    t = time.perf_counter() if verbose else None
    input_bounds = convert_simple_input_bounds(simple_input_bounds, n_inputs)
    if verbose and t is not None:
        print(f"    Input bounds conversion: {time.perf_counter() - t:.4f}s")

    # Parse OR block for output constraints
    t = time.perf_counter() if verbose else None
    output_constrs = parse_or_block(or_block_lines, n_inputs, n_outputs)
    if verbose and t is not None:
        print(f"    OR block parsing: {time.perf_counter() - t:.4f}s")
        print(f"    Extracted {len(output_constrs)} output constraint options")

    # Package in expected format: [[(input_bounds, [output_constrs])]]
    and_properties = [[(input_bounds, output_constrs)]]

    if verbose and t_start is not None:
        print(f"  Type2 total time: {time.perf_counter() - t_start:.4f}s")

    return and_properties
