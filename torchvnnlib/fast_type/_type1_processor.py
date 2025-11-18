"""Optimized Type1 processor using pre-parsed data.

This version uses data from fast_detect_and_parse to avoid re-scanning lines.
"""

__docformat__ = "restructuredtext"
__all__ = ["process_type1"]

import time

import torch
from torch import Tensor

from .._to_tensor import convert_and_output_constrs
from ..ast import tokenize, parse, optimize, And, Or


def process_type1(
    simple_input_bounds: list[tuple],
    simple_output_constrs: list[tuple],
    complex_lines: list[str],
    n_inputs: int,
    n_outputs: int,
    verbose: bool = False,
    simple_output_bounds: list[tuple] | None = None,
) -> list[list[tuple[Tensor, list[Tensor]]]]:
    """
    Fast Type1 processor using pre-parsed data.

    Type1 has simple input bounds and simple output constraints/bounds.
    All patterns should be parseable without AST.

    Args:
        simple_input_bounds: Pre-parsed simple input bounds (X_ constraints)
        simple_output_constrs: Pre-parsed simple output constraints (Y_ <op> Y_ constraints)
        complex_lines: Lines that don't match simple patterns (should be minimal for TYPE1)
        n_inputs: Number of input variables
        n_outputs: Number of output variables
        verbose: Print timing information
        simple_output_bounds: Pre-parsed simple output bounds (Y_ <op> value constraints)

    Returns:
        Standardized format: [[(input_bounds, [output_constrs])]]
    """
    if verbose:
        t_start = time.perf_counter()
        print(f"  Type1 fast processing:")
        print(f"    Simple input bounds: {len(simple_input_bounds)}")
        print(f"    Simple output constraints: {len(simple_output_constrs)}")
        if simple_output_bounds:
            print(f"    Simple output bounds: {len(simple_output_bounds)}")
        print(f"    Complex lines: {len(complex_lines)}")

    # Convert simple input bounds to tensor
    t = time.perf_counter()
    input_bounds = _convert_simple_input_bounds_vectorized(
        simple_input_bounds, n_inputs
    )
    if verbose:
        print(f"    Input bounds conversion: {time.perf_counter() - t:.4f}s")

    # Convert simple output constraints and bounds to tensors
    t = time.perf_counter()
    output_constrs_list = []

    # Handle Y <op> Y constraints
    if simple_output_constrs:
        output_constrs_tensor = _convert_simple_output_constraints(
            simple_output_constrs, n_outputs
        )
        if output_constrs_tensor is not None:
            output_constrs_list.append(output_constrs_tensor)

    # Handle Y <op> value constraints
    if simple_output_bounds:
        output_bounds_tensor = _convert_simple_output_bounds(
            simple_output_bounds, n_outputs
        )
        if output_bounds_tensor is not None:
            output_constrs_list.append(output_bounds_tensor)

    if verbose:
        print(f"    Output constraints conversion: {time.perf_counter() - t:.4f}s")

    # Handle complex expressions if any
    if complex_lines:
        if verbose:
            print(f"    Processing {len(complex_lines)} complex expressions...")

        t = time.perf_counter()
        tokens_list = tokenize(complex_lines, verbose=False, use_parallel=False)
        expr_complex = parse(tokens_list, verbose=False, use_parallel=False)
        expr_complex = optimize(expr_complex, verbose=False, use_parallel=False)
        if verbose:
            print(f"    Complex expression processing: {time.perf_counter() - t:.4f}s")

        # Extract output constraints
        t = time.perf_counter()
        complex_output_constrs = []

        if isinstance(expr_complex, Or):
            for or_arg in expr_complex.args:
                if isinstance(or_arg, And):
                    constr = convert_and_output_constrs(or_arg, n_outputs, n_inputs)
                else:
                    constr = convert_and_output_constrs(
                        And([or_arg]), n_outputs, n_inputs
                    )
                complex_output_constrs.append(constr)
        elif isinstance(expr_complex, And) and len(expr_complex.args) > 0:
            for arg in expr_complex.args:
                if isinstance(arg, Or):
                    for or_arg in arg.args:
                        if isinstance(or_arg, And):
                            constr = convert_and_output_constrs(
                                or_arg, n_outputs, n_inputs
                            )
                        else:
                            constr = convert_and_output_constrs(
                                And([or_arg]), n_outputs, n_inputs
                            )
                        complex_output_constrs.append(constr)
                else:
                    constr = convert_and_output_constrs(And([arg]), n_outputs, n_inputs)
                    complex_output_constrs.append(constr)
        elif isinstance(expr_complex, And):
            constr = convert_and_output_constrs(expr_complex, n_outputs, n_inputs)
            complex_output_constrs.append(constr)
        else:
            constr = convert_and_output_constrs(
                And([expr_complex]), n_outputs, n_inputs
            )
            complex_output_constrs.append(constr)

        if verbose:
            print(f"    Complex output extraction: {time.perf_counter() - t:.4f}s")
            print(
                f"    Extracted {len(complex_output_constrs)} complex output constraints"
            )

        # Merge simple and complex
        all_output_constrs = output_constrs_list + complex_output_constrs
        if not all_output_constrs:
            all_output_constrs = [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]
    else:
        # Pure Type1 - no complex lines
        all_output_constrs = output_constrs_list
        if not all_output_constrs:
            all_output_constrs = [torch.zeros((1, n_outputs + 1), dtype=torch.float64)]

    and_properties = [[(input_bounds, all_output_constrs)]]

    if verbose:
        print(f"  Type1 fast total: {time.perf_counter() - t_start:.4f}s")

    return and_properties


def _convert_simple_input_bounds_vectorized(
    simple_bounds: list[tuple], n_inputs: int
) -> Tensor:
    """Vectorized input bounds conversion (optimized version)."""
    input_bounds = torch.full((n_inputs, 2), float("nan"), dtype=torch.float64)

    # Group by operation type for batch processing
    leq_indices, leq_values = [], []
    geq_indices, geq_values = [], []
    eq_indices, eq_values = [], []

    for op, var_type, idx, value in simple_bounds:
        if var_type != "X_":
            continue

        if op == "<=":
            leq_indices.append(idx)
            leq_values.append(value)
        elif op == ">=":
            geq_indices.append(idx)
            geq_values.append(value)
        elif op == "=":
            eq_indices.append(idx)
            eq_values.append(value)

    # Batch assignments
    if leq_indices:
        input_bounds[leq_indices, 1] = torch.tensor(leq_values, dtype=torch.float64)
    if geq_indices:
        input_bounds[geq_indices, 0] = torch.tensor(geq_values, dtype=torch.float64)
    if eq_indices:
        eq_vals = torch.tensor(eq_values, dtype=torch.float64)
        eq_idx = torch.tensor(eq_indices, dtype=torch.long)
        input_bounds[eq_idx, 0] = eq_vals
        input_bounds[eq_idx, 1] = eq_vals

    # Validation
    if torch.isnan(input_bounds).any():
        nan_indices = torch.where(torch.isnan(input_bounds))
        raise ValueError(
            f"Missing input bounds at indices: {list(zip(nan_indices[0].tolist(), nan_indices[1].tolist()))}"
        )

    return input_bounds


def _convert_simple_output_constraints(
    simple_output_constrs: list[tuple], n_outputs: int
) -> Tensor:
    """Convert simple output constraints (Y <op> Y format)."""
    if not simple_output_constrs:
        return None

    constraints = []
    for op, var_prefix1, idx1, var_prefix2, idx2 in simple_output_constrs:
        constr = torch.zeros(n_outputs + 1, dtype=torch.float64)

        if op == "<=":
            # Y_idx1 <= Y_idx2  =>  Y_idx1 - Y_idx2 <= 0  =>  -Y_idx1 + Y_idx2 >= 0
            constr[idx1 + 1] = -1.0
            constr[idx2 + 1] = 1.0
        elif op == ">=":
            # Y_idx1 >= Y_idx2  =>  Y_idx1 - Y_idx2 >= 0
            constr[idx1 + 1] = 1.0
            constr[idx2 + 1] = -1.0

        constraints.append(constr)

    return torch.stack(constraints) if constraints else None


def _convert_simple_output_bounds(
    simple_output_bounds: list[tuple], n_outputs: int
) -> Tensor:
    """Convert simple output bounds (Y <op> value format)."""
    if not simple_output_bounds:
        return None

    constraints = []
    for op, var_prefix, idx, value in simple_output_bounds:
        constr = torch.zeros(n_outputs + 1, dtype=torch.float64)

        if op == "<=":
            # Y_idx <= value  =>  Y_idx - value <= 0  =>  Y_idx >= -value
            constr[0] = -float(value)  # constant term
            constr[idx + 1] = 1.0
        elif op == ">=":
            # Y_idx >= value  =>  Y_idx - value >= 0
            constr[0] = -float(value)  # constant term
            constr[idx + 1] = 1.0
        elif op == "=":
            # Y_idx = value  =>  Y_idx - value >= 0 AND value - Y_idx >= 0
            # First constraint: Y_idx >= value
            constr1 = torch.zeros(n_outputs + 1, dtype=torch.float64)
            constr1[0] = -float(value)
            constr1[idx + 1] = 1.0
            constraints.append(constr1)

            # Second constraint: Y_idx <= value
            constr2 = torch.zeros(n_outputs + 1, dtype=torch.float64)
            constr2[0] = float(value)
            constr2[idx + 1] = -1.0
            constraints.append(constr2)
            continue

        constraints.append(constr)

    return torch.stack(constraints) if constraints else None
