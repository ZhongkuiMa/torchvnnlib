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

from .._to_tensor import convert_input_bounds, convert_and_output_constrs
from ..ast import tokenize, parse, optimize, And, Or


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
    """Parse two OR blocks for inputs and outputs."""

    # Parse all lines together
    tokens_list = tokenize(lines, verbose=False, use_parallel=False)
    expr = parse(tokens_list, verbose=False, use_parallel=False)
    expr = optimize(expr, verbose=False, use_parallel=False)

    input_bounds_list = []
    output_constrs_list = []

    # Expect structure: And([Or(...inputs...), Or(...outputs...)])
    if isinstance(expr, And) and len(expr.args) >= 2:
        # Find which OR is for inputs and which for outputs
        or_blocks = [arg for arg in expr.args if isinstance(arg, Or)]

        if len(or_blocks) >= 2:
            # First OR block should be inputs, second outputs
            input_or = or_blocks[0]
            output_or = or_blocks[1]

            # Extract input bounds from first OR
            for or_arg in input_or.args:
                if isinstance(or_arg, And):
                    bounds = convert_input_bounds(or_arg, n_inputs)
                else:
                    bounds = convert_input_bounds(And([or_arg]), n_inputs)
                input_bounds_list.append(bounds)

            # Extract output constraints from second OR
            for or_arg in output_or.args:
                if isinstance(or_arg, And):
                    constr = convert_and_output_constrs(or_arg, n_outputs, n_inputs)
                else:
                    constr = convert_and_output_constrs(
                        And([or_arg]), n_outputs, n_inputs
                    )
                output_constrs_list.append(constr)

    # Fallback: if structure doesn't match, use default
    if not input_bounds_list:
        input_bounds_list = [
            torch.full((n_inputs, 2), float("nan"), dtype=torch.float64)
        ]
    if not output_constrs_list:
        output_constrs_list = [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]

    return input_bounds_list, output_constrs_list
