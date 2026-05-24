"""Type4 VNN-LIB Processor: OR inputs + OR outputs."""

__docformat__ = "restructuredtext"
__all__ = ["process_type4"]

import logging
import time

from torchvnnlib._backend import Backend, TensorLike
from torchvnnlib.fast_type._utils import parse_dual_or_blocks

_logger = logging.getLogger(__name__)


def process_type4(
    lines: list[str],
    n_inputs: int,
    n_outputs: int,
    backend: Backend,
    verbose: bool = False,
) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
    """Fast processor for Type4 VNN-LIB files.

    :param lines: Preprocessed assertion lines.

    :param n_inputs: Number of input variables.

    :param n_outputs: Number of output variables.

    :param backend: Backend instance for tensor operations.

    :param verbose: Print timing information.

    :return: Standardized format with Cartesian product of input x output regions
    """
    t_start = time.perf_counter()

    if verbose:
        from torchvnnlib._logging import _enable_verbose

        _enable_verbose()

    t = time.perf_counter()
    input_bounds_list, output_constrs_list = parse_dual_or_blocks(
        lines, n_inputs, n_outputs, backend
    )
    if verbose:
        _logger.info("  Type4 detection:")
    if verbose:
        _logger.info(f"    Input OR groups: {len(input_bounds_list)}")
    if verbose:
        _logger.info(f"    Output OR options: {len(output_constrs_list)}")
    if verbose:
        _logger.info(f"    Parsing: {time.perf_counter() - t:.4f}s")

    or_properties = [(input_bounds, output_constrs_list) for input_bounds in input_bounds_list]
    and_properties = [or_properties]

    if verbose:
        _logger.info(f"  Type4 total time: {time.perf_counter() - t_start:.4f}s")

    return and_properties
