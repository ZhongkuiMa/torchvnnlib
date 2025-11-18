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

from ._utils import SIMPLE_INPUT_BOUND_PATTERN
from .._to_tensor import convert_and_output_constrs
from ..ast import tokenize, parse, optimize, And, Or


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
    if verbose:
        t_start = time.perf_counter()

    # Use pre-parsed data if available, otherwise parse now
    if parsed_data is None:
        from ._fast_type_detect import parse_simple_patterns
        t = time.perf_counter()
        parsed_data = parse_simple_patterns(lines, verbose=False)
        if verbose:
            print(f"  Type2 parsing: {time.perf_counter() - t:.4f}s")

    simple_input_bounds = parsed_data['simple_input_bounds']
    or_block_lines = parsed_data['complex_lines']  # OR blocks are complex

    if verbose:
        print(f"  Type2 processing:")
        print(f"    Simple input bounds: {len(simple_input_bounds)}")
        print(f"    OR block lines: {len(or_block_lines)}")

    # Convert simple input bounds to tensor
    t = time.perf_counter()
    input_bounds = _convert_simple_input_bounds(simple_input_bounds, n_inputs, verbose)
    if verbose:
        print(f"    Input bounds conversion: {time.perf_counter() - t:.4f}s")

    # Parse OR block for output constraints
    t = time.perf_counter()
    output_constrs = _parse_or_block(or_block_lines, n_inputs, n_outputs, verbose)
    if verbose:
        print(f"    OR block parsing: {time.perf_counter() - t:.4f}s")
        print(f"    Extracted {len(output_constrs)} output constraint options")

    # Package in expected format: [[(input_bounds, [output_constrs])]]
    and_properties = [[(input_bounds, output_constrs)]]

    if verbose:
        print(f"  Type2 total time: {time.perf_counter() - t_start:.4f}s")

    return and_properties


def _convert_simple_input_bounds(
    simple_bounds: list[tuple], n_inputs: int, verbose: bool = False
) -> Tensor:
    """Convert simple input bounds to tensor."""
    input_bounds = torch.full((n_inputs, 2), float("nan"), dtype=torch.float64)

    for op, var_type, idx, value in simple_bounds:
        if var_type != "X_":
            continue

        if op == "<=":
            input_bounds[idx, 1] = value
        elif op == ">=":
            input_bounds[idx, 0] = value
        elif op == "=":
            input_bounds[idx, 0] = value
            input_bounds[idx, 1] = value

    # Check for missing bounds
    if torch.isnan(input_bounds).any():
        nan_indices = torch.where(torch.isnan(input_bounds))
        raise ValueError(
            f"Missing input bounds at indices: {list(zip(nan_indices[0].tolist(), nan_indices[1].tolist()))}"
        )

    return input_bounds


def _parse_or_block(
    or_block_lines: list[str], n_inputs: int, n_outputs: int, verbose: bool = False
) -> list[Tensor]:
    """Parse OR block to extract output constraints."""

    if not or_block_lines:
        return [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]

    # Parse OR block using standard AST path
    tokens_list = tokenize(or_block_lines, verbose=False, use_parallel=False)
    expr = parse(tokens_list, verbose=False, use_parallel=False)
    expr = optimize(expr, verbose=False, use_parallel=False)

    output_constrs = []

    # Extract output constraints from OR structure
    if isinstance(expr, Or):
        # Direct OR expression
        for or_arg in expr.args:
            if isinstance(or_arg, And):
                constr = convert_and_output_constrs(or_arg, n_outputs, n_inputs)
            else:
                constr = convert_and_output_constrs(And([or_arg]), n_outputs, n_inputs)
            output_constrs.append(constr)
    elif isinstance(expr, And):
        # Might have OR nested inside
        for arg in expr.args:
            if isinstance(arg, Or):
                for or_arg in arg.args:
                    if isinstance(or_arg, And):
                        constr = convert_and_output_constrs(or_arg, n_outputs, n_inputs)
                    else:
                        constr = convert_and_output_constrs(
                            And([or_arg]), n_outputs, n_inputs
                        )
                    output_constrs.append(constr)
            else:
                constr = convert_and_output_constrs(And([arg]), n_outputs, n_inputs)
                output_constrs.append(constr)
    else:
        # Single constraint
        constr = convert_and_output_constrs(And([expr]), n_outputs, n_inputs)
        output_constrs.append(constr)

    return (
        output_constrs
        if output_constrs
        else [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]
    )
