"""Optimized Type1 processor using pre-parsed data."""

__docformat__ = "restructuredtext"
__all__ = ["process_type1"]

import time

from torchvnnlib._backend import Backend, TensorLike
from torchvnnlib._to_tensor import convert_and_output_constrs
from torchvnnlib.ast import And, Or, optimize, parse, tokenize
from torchvnnlib.fast_type._utils import convert_simple_input_bounds


def _process_simple_output_constrs_and_bounds(
    simple_output_constrs: list[tuple],
    simple_output_bounds: list[tuple] | None,
    n_outputs: int,
    backend: Backend,
) -> list[TensorLike]:
    """Process simple output constraints and bounds into tensors.

    :param simple_output_constrs: List of simple output constraint tuples
    :param simple_output_bounds: List of simple output bound tuples
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
    :return: List of constraint tensors
    """
    output_constrs_list = []

    if simple_output_constrs:
        output_constrs_tensor = _convert_simple_output_constraints_batched(
            simple_output_constrs, n_outputs, backend
        )
        if output_constrs_tensor is not None:
            output_constrs_list.append(output_constrs_tensor)

    if simple_output_bounds:
        output_bounds_tensor = _convert_simple_output_bounds_batched(
            simple_output_bounds, n_outputs, backend
        )
        if output_bounds_tensor is not None:
            output_constrs_list.append(output_bounds_tensor)

    return output_constrs_list


def _ensure_non_empty_constraints(
    all_output_constrs: list[TensorLike], n_outputs: int, backend: Backend
) -> list[TensorLike]:
    """Ensure constraint list is non-empty by adding default zeros.

    :param all_output_constrs: List of constraint tensors
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
    :return: Non-empty constraint list
    """
    if not all_output_constrs:
        return [backend.zeros((1, n_outputs + 1), dtype="float64")]
    return all_output_constrs


def _extract_complex_output_constraints(
    expr_complex, n_outputs: int, n_inputs: int, backend: Backend
) -> list[TensorLike]:
    """Extract output constraints from complex expression tree."""
    complex_output_constrs = []

    if isinstance(expr_complex, Or):
        for or_arg in expr_complex.args:
            if isinstance(or_arg, And):
                constr = convert_and_output_constrs(or_arg, n_outputs, n_inputs, backend)
            else:
                constr = convert_and_output_constrs(And([or_arg]), n_outputs, n_inputs, backend)
            complex_output_constrs.append(constr)
    elif isinstance(expr_complex, And) and len(expr_complex.args) > 0:
        for arg in expr_complex.args:
            if isinstance(arg, Or):
                for or_arg in arg.args:
                    if isinstance(or_arg, And):
                        constr = convert_and_output_constrs(or_arg, n_outputs, n_inputs, backend)
                    else:
                        constr = convert_and_output_constrs(
                            And([or_arg]), n_outputs, n_inputs, backend
                        )
                    complex_output_constrs.append(constr)
            else:
                constr = convert_and_output_constrs(And([arg]), n_outputs, n_inputs, backend)
                complex_output_constrs.append(constr)
    elif isinstance(expr_complex, And):
        constr = convert_and_output_constrs(expr_complex, n_outputs, n_inputs, backend)
        complex_output_constrs.append(constr)
    else:
        constr = convert_and_output_constrs(And([expr_complex]), n_outputs, n_inputs, backend)
        complex_output_constrs.append(constr)

    return complex_output_constrs


def process_type1(
    simple_input_bounds: list[tuple],
    simple_output_constrs: list[tuple],
    complex_lines: list[str],
    n_inputs: int,
    n_outputs: int,
    backend: Backend,
    verbose: bool = False,
    simple_output_bounds: list[tuple] | None = None,
) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
    """Fast Type1 processor using pre-parsed data.

    :param simple_input_bounds: Pre-parsed simple input bounds
    :param simple_output_constrs: Pre-parsed simple output constraints
    :param complex_lines: Lines that don't match simple patterns
    :param n_inputs: Number of input variables
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
    :param verbose: Print timing information
    :param simple_output_bounds: Pre-parsed simple output bounds
    :return: Standardized format: [[(input_bounds, [output_constrs])]]
    """
    t_start = time.perf_counter() if verbose else None

    if verbose:
        print("  Type1 fast processing:")
        print(f"    Simple input bounds: {len(simple_input_bounds)}")
        print(f"    Simple output constraints: {len(simple_output_constrs)}")
        if simple_output_bounds:
            print(f"    Simple output bounds: {len(simple_output_bounds)}")
        print(f"    Complex lines: {len(complex_lines)}")

    t = time.perf_counter() if verbose else None
    input_bounds = convert_simple_input_bounds(simple_input_bounds, n_inputs, backend)
    if verbose and t is not None:
        print(f"    Input bounds conversion: {time.perf_counter() - t:.4f}s")

    t = time.perf_counter() if verbose else None
    output_constrs_list = _process_simple_output_constrs_and_bounds(
        simple_output_constrs, simple_output_bounds, n_outputs, backend
    )

    if verbose and t is not None:
        print(f"    Output constraints conversion: {time.perf_counter() - t:.4f}s")

    if complex_lines:
        if verbose:
            print(f"    Processing {len(complex_lines)} complex expressions...")

        t = time.perf_counter() if verbose else None
        tokens_list = tokenize(complex_lines, verbose=False, use_parallel=False)
        expr_complex = parse(tokens_list, verbose=False, use_parallel=False)
        expr_complex = optimize(expr_complex, verbose=False, use_parallel=False)
        if verbose and t is not None:
            print(f"    Complex expression processing: {time.perf_counter() - t:.4f}s")

        t = time.perf_counter() if verbose else None
        complex_output_constrs = _extract_complex_output_constraints(
            expr_complex, n_outputs, n_inputs, backend
        )

        if verbose and t is not None:
            print(f"    Complex output extraction: {time.perf_counter() - t:.4f}s")
            print(f"    Extracted {len(complex_output_constrs)} complex output constraints")

        all_output_constrs = output_constrs_list + complex_output_constrs
    else:
        all_output_constrs = output_constrs_list

    all_output_constrs = _ensure_non_empty_constraints(all_output_constrs, n_outputs, backend)
    and_properties = [[(input_bounds, all_output_constrs)]]

    if verbose and t_start is not None:
        print(f"  Type1 fast total: {time.perf_counter() - t_start:.4f}s")

    return and_properties


def _convert_simple_output_constraints_batched(
    simple_output_constrs: list[tuple], n_outputs: int, backend: Backend
) -> TensorLike | None:
    """Convert simple output constraints (Y <op> Y format) - batched version.

    :param simple_output_constrs: List of constraint tuples
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
    :return: Constraint tensor or None
    """
    if not simple_output_constrs:
        return None

    n_constrs = len(simple_output_constrs)
    constraints = backend.zeros((n_constrs, n_outputs + 1), dtype="float64")

    for i, (op, _var_prefix1, idx1, _var_prefix2, idx2) in enumerate(simple_output_constrs):
        if op == "<=":
            constraints[i, idx1 + 1] = -1.0
            constraints[i, idx2 + 1] = 1.0
        elif op == ">=":
            constraints[i, idx1 + 1] = 1.0
            constraints[i, idx2 + 1] = -1.0

    return constraints


def _convert_simple_output_bounds_batched(
    simple_output_bounds: list[tuple], n_outputs: int, backend: Backend
) -> TensorLike | None:
    """Convert simple output bounds (Y <op> value format) - batched version.

    :param simple_output_bounds: List of bound tuples
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
    :return: Constraint tensor or None
    """
    if not simple_output_bounds:
        return None

    n_constrs = sum(2 if op == "=" else 1 for op, _, _, _ in simple_output_bounds)
    constraints = backend.zeros((n_constrs, n_outputs + 1), dtype="float64")

    row_idx = 0
    for op, _var_prefix, idx, value in simple_output_bounds:
        if op == "<=":
            constraints[row_idx, 0] = float(value)
            constraints[row_idx, idx + 1] = -1.0
            row_idx += 1
        elif op == ">=":
            constraints[row_idx, 0] = -float(value)
            constraints[row_idx, idx + 1] = 1.0
            row_idx += 1
        elif op == "=":
            constraints[row_idx, 0] = -float(value)
            constraints[row_idx, idx + 1] = 1.0
            row_idx += 1

            constraints[row_idx, 0] = float(value)
            constraints[row_idx, idx + 1] = -1.0
            row_idx += 1

    return constraints
