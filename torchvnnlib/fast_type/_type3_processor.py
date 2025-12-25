"""Type3 VNN-LIB Processor: OR inputs + Simple outputs."""

__docformat__ = "restructuredtext"
__all__ = ["process_type3"]

import time

from torchvnnlib._backend import Backend, TensorLike
from torchvnnlib.fast_type._utils import parse_input_or_block


def _process_output_constraints_type3(
    simple_output_constrs: list[tuple],
    n_outputs: int,
    backend: Backend,
) -> list[TensorLike]:
    """Process simple output constraints into tensor list.

    :param simple_output_constrs: List of constraint tuples
    :param n_outputs: Number of output variables
    :param backend: Backend instance
    :return: List of constraint tensors
    """
    output_constrs_list = []

    if simple_output_constrs:
        output_constr = backend.zeros((len(simple_output_constrs), n_outputs + 1), dtype="float64")
        for i, (op, _, idx1, _, idx2) in enumerate(simple_output_constrs):
            if op == "<=":
                output_constr[i, idx1 + 1] = -1.0
                output_constr[i, idx2 + 1] = 1.0
            elif op == ">=":
                output_constr[i, idx1 + 1] = 1.0
                output_constr[i, idx2 + 1] = -1.0
        output_constrs_list.append(output_constr)

    return output_constrs_list


def _process_output_bounds_type3(
    simple_output_bounds: list[tuple],
    n_outputs: int,
    backend: Backend,
) -> list[TensorLike]:
    """Process simple output bounds into tensor list.

    :param simple_output_bounds: List of bound tuples
    :param n_outputs: Number of output variables
    :param backend: Backend instance
    :return: List of bound tensors
    """
    output_bounds_list = []

    if simple_output_bounds:
        n_bounds = sum(2 if op == "=" else 1 for op, _, _, _ in simple_output_bounds)
        output_bounds = backend.zeros((n_bounds, n_outputs + 1), dtype="float64")
        row_idx = 0
        for op, _, idx, value in simple_output_bounds:
            if op == "<=":
                output_bounds[row_idx, 0] = float(value)
                output_bounds[row_idx, idx + 1] = -1.0
                row_idx += 1
            elif op == ">=":
                output_bounds[row_idx, 0] = -float(value)
                output_bounds[row_idx, idx + 1] = 1.0
                row_idx += 1
            elif op == "=":
                output_bounds[row_idx, 0] = -float(value)
                output_bounds[row_idx, idx + 1] = 1.0
                row_idx += 1
                output_bounds[row_idx, 0] = float(value)
                output_bounds[row_idx, idx + 1] = -1.0
                row_idx += 1
        output_bounds_list.append(output_bounds)

    return output_bounds_list


def process_type3(
    lines: list[str],
    n_inputs: int,
    n_outputs: int,
    backend: Backend,
    verbose: bool = False,
    parsed_data: dict | None = None,
) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
    """Fast processor for Type3 VNN-LIB files.

    :param lines: Preprocessed assertion lines
    :param n_inputs: Number of input variables
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
    :param verbose: Print timing information
    :param parsed_data: Pre-parsed data from parse_simple_patterns()
    :return: Standardized format with OR input regions
    """
    t_start = time.perf_counter() if verbose else None

    if parsed_data is None:
        from torchvnnlib.fast_type._fast_type_detect import parse_simple_patterns

        t = time.perf_counter() if verbose else None
        parsed_data = parse_simple_patterns(lines, verbose=False)
        if verbose and t is not None:
            print(f"  Type3 parsing: {time.perf_counter() - t:.4f}s")

    simple_output_constrs = parsed_data["simple_output_constrs"]
    simple_output_bounds = parsed_data["simple_output_bounds"]
    or_block_lines = parsed_data["complex_lines"]

    if verbose:
        print("  Type3 processing:")
        print(f"    OR block lines: {len(or_block_lines)}")

    t = time.perf_counter() if verbose else None
    input_bounds_list = parse_input_or_block(or_block_lines, n_inputs, backend)
    if verbose and t is not None:
        print(f"    Input OR block parsing: {time.perf_counter() - t:.4f}s")
        print(f"    Extracted {len(input_bounds_list)} input regions")

    t = time.perf_counter() if verbose else None
    output_constrs_list = _process_output_constraints_type3(
        simple_output_constrs, n_outputs, backend
    )
    output_bounds = _process_output_bounds_type3(simple_output_bounds, n_outputs, backend)
    output_constrs_list.extend(output_bounds)

    if not output_constrs_list:
        output_constrs_list = [backend.zeros((1, n_outputs + 1), dtype="float64")]

    if verbose and t is not None:
        print(f"    Output constraints conversion: {time.perf_counter() - t:.4f}s")

    or_properties = [(input_bounds, output_constrs_list) for input_bounds in input_bounds_list]
    and_properties = [or_properties]

    if verbose and t_start is not None:
        print(f"  Type3 total time: {time.perf_counter() - t_start:.4f}s")

    return and_properties
