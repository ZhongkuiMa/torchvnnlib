"""Type3 VNN-LIB Processor: OR inputs + Simple outputs."""

__docformat__ = "restructuredtext"
__all__ = ["process_type3"]

import logging
import time

from torchvnnlib._backend import Backend, TensorLike
from torchvnnlib._constraint_row import (
    normalize_neg_zero,
    write_compare_row,
    write_value_bound_row,
    write_value_bound_rows_eq,
)
from torchvnnlib._logging import _enable_verbose
from torchvnnlib.fast_type._fast_type_detect import ParsedPatterns, parse_simple_patterns
from torchvnnlib.fast_type._utils import parse_input_or_block

_logger = logging.getLogger(__name__)


def _process_output_constraints_type3(
    simple_output_constrs: list[tuple], n_outputs: int, backend: Backend
) -> list[TensorLike]:
    """Stack ``Y_i <op> Y_j`` constraint tuples into one constraint block."""
    if not simple_output_constrs:
        return []

    output_constr = backend.zeros((len(simple_output_constrs), n_outputs + 1), dtype="float64")
    for i, (op, _, idx1, _, idx2) in enumerate(simple_output_constrs):
        write_compare_row(output_constr[i], idx1, idx2, op)
    return [normalize_neg_zero(output_constr)]


def _process_output_bounds_type3(
    simple_output_bounds: list[tuple], n_outputs: int, backend: Backend
) -> list[TensorLike]:
    """Stack ``Y_i <op> value`` bound tuples into one constraint block."""
    if not simple_output_bounds:
        return []

    n_bounds = sum(2 if op == "=" else 1 for op, _, _, _ in simple_output_bounds)
    output_bounds = backend.zeros((n_bounds, n_outputs + 1), dtype="float64")
    row_idx = 0
    for op, _, idx, value in simple_output_bounds:
        value_f = float(value)
        if op == "=":
            write_value_bound_rows_eq(
                output_bounds[row_idx], output_bounds[row_idx + 1], idx, value_f
            )
            row_idx += 2
        else:
            write_value_bound_row(output_bounds[row_idx], idx, op, value_f)
            row_idx += 1
    return [normalize_neg_zero(output_bounds)]


def process_type3(
    lines: list[str],
    n_inputs: int,
    n_outputs: int,
    backend: Backend,
    verbose: bool = False,
    parsed_data: ParsedPatterns | None = None,
) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
    """Fast processor for Type3 VNN-LIB files.

    :param lines: Preprocessed assertion lines.

    :param n_inputs: Number of input variables.

    :param n_outputs: Number of output variables.

    :param backend: Backend instance for tensor operations.

    :param verbose: Print timing information.

    :param parsed_data: Pre-parsed data from parse_simple_patterns().

    :return: Standardized format with OR input regions
    """
    t_start = time.perf_counter()

    if verbose:
        _enable_verbose()

    if parsed_data is None:
        t = time.perf_counter()
        parsed_data = parse_simple_patterns(lines, verbose=False)
        if verbose:
            _logger.info(f"  Type3 parsing: {time.perf_counter() - t:.4f}s")

    simple_output_constrs = parsed_data["simple_output_constrs"]
    simple_output_bounds = parsed_data["simple_output_bounds"]
    or_block_lines = parsed_data["complex_lines"]

    if verbose:
        _logger.info("  Type3 processing:")
        _logger.info(f"    OR block lines: {len(or_block_lines)}")

    t = time.perf_counter()
    input_bounds_list = parse_input_or_block(or_block_lines, n_inputs, backend)
    if verbose:
        _logger.info(f"    Input OR block parsing: {time.perf_counter() - t:.4f}s")
    if verbose:
        _logger.info(f"    Extracted {len(input_bounds_list)} input regions")

    t = time.perf_counter()
    output_constrs_list = _process_output_constraints_type3(
        simple_output_constrs, n_outputs, backend
    )
    output_bounds = _process_output_bounds_type3(simple_output_bounds, n_outputs, backend)
    output_constrs_list.extend(output_bounds)

    if not output_constrs_list:
        output_constrs_list = [backend.zeros((1, n_outputs + 1), dtype="float64")]

    if verbose:
        _logger.info(f"    Output constraints conversion: {time.perf_counter() - t:.4f}s")

    or_properties = [(input_bounds, output_constrs_list) for input_bounds in input_bounds_list]
    and_properties = [or_properties]

    if verbose:
        _logger.info(f"  Type3 total time: {time.perf_counter() - t_start:.4f}s")

    return and_properties
