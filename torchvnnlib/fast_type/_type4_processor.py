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

from ._utils import parse_dual_or_blocks


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

    # Parse all lines to find two OR blocks using shared utility
    t = time.perf_counter()
    input_bounds_list, output_constrs_list = parse_dual_or_blocks(
        lines, n_inputs, n_outputs
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
