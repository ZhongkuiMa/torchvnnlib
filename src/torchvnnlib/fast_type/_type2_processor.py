"""Type2 VNN-LIB Processor: Simple inputs + OR outputs."""

__docformat__ = "restructuredtext"
__all__ = ["process_type2"]

import logging
import time

from torchvnnlib._backend import Backend, TensorLike
from torchvnnlib.fast_type._utils import convert_simple_input_bounds, parse_or_block

_logger = logging.getLogger(__name__)


def process_type2(
    lines: list[str],
    n_inputs: int,
    n_outputs: int,
    backend: Backend,
    verbose: bool = False,
    parsed_data: dict | None = None,
) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
    """Fast processor for Type2 VNN-LIB files.

    :param lines: Preprocessed assertion lines.

    :param n_inputs: Number of input variables.

    :param n_outputs: Number of output variables.

    :param backend: Backend instance for tensor operations.

    :param verbose: Print timing information.

    :param parsed_data: Pre-parsed data from parse_simple_patterns().

    :return: Standardized format: [[(input_bounds, [output_constr1, output_constr2, ...])]]
    """
    t_start = time.perf_counter()

    if verbose:
        from torchvnnlib import _ensure_verbose_handler

        _ensure_verbose_handler()

    if parsed_data is None:
        from torchvnnlib.fast_type._fast_type_detect import parse_simple_patterns

        t = time.perf_counter()
        parsed_data = parse_simple_patterns(lines, verbose=False)
        if verbose:
            _logger.info(f"  Type2 parsing: {time.perf_counter() - t:.4f}s")

    simple_input_bounds = parsed_data["simple_input_bounds"]
    or_block_lines = parsed_data["complex_lines"]

    if verbose:
        _logger.info("  Type2 processing:")
    if verbose:
        _logger.info(f"    Simple input bounds: {len(simple_input_bounds)}")
    if verbose:
        _logger.info(f"    OR block lines: {len(or_block_lines)}")

    t = time.perf_counter()
    input_bounds = convert_simple_input_bounds(simple_input_bounds, n_inputs, backend)
    if verbose:
        _logger.info(f"    Input bounds conversion: {time.perf_counter() - t:.4f}s")

    t = time.perf_counter()
    output_constrs = parse_or_block(or_block_lines, n_inputs, n_outputs, backend)
    if verbose:
        _logger.info(f"    OR block parsing: {time.perf_counter() - t:.4f}s")
    if verbose:
        _logger.info(f"    Extracted {len(output_constrs)} output constraint options")

    and_properties = [[(input_bounds, output_constrs)]]

    if verbose:
        _logger.info(f"  Type2 total time: {time.perf_counter() - t_start:.4f}s")

    return and_properties
