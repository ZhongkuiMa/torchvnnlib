"""Type5 VNN-LIB Processor: Top-level OR wrapping complete properties."""

__docformat__ = "restructuredtext"
__all__ = ["process_type5"]

import logging
import time

from torchvnnlib._backend import Backend, TensorLike
from torchvnnlib._logging import _enable_verbose
from torchvnnlib.fast_type._utils import iter_and_blocks, parse_and_block

_logger = logging.getLogger(__name__)


def process_type5(
    lines: list[str],
    n_inputs: int,
    n_outputs: int,
    backend: Backend,
    verbose: bool = False,
) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
    """Fast processor for Type5 VNN-LIB files.

    :param lines: Preprocessed assertion lines.

    :param n_inputs: Number of input variables.

    :param n_outputs: Number of output variables.

    :param backend: Backend instance for tensor operations.

    :param verbose: Print timing information.

    :return: Standardized format with complete properties
    """
    t_start = time.perf_counter()

    if verbose:
        _enable_verbose()

    t = time.perf_counter()
    properties = _parse_top_level_or(lines, n_inputs, n_outputs, backend)
    if verbose:
        _logger.info("  Type5 detection:")
        _logger.info(f"    OR clauses (properties): {len(properties)}")
        _logger.info(f"    Parsing: {time.perf_counter() - t:.4f}s")

    and_properties = [properties]

    if verbose:
        _logger.info(f"  Type5 total time: {time.perf_counter() - t_start:.4f}s")

    return and_properties


def _parse_top_level_or(
    lines: list[str],
    n_inputs: int,
    n_outputs: int,
    backend: Backend,
) -> list[tuple[TensorLike, list[TensorLike]]]:
    """Parse top-level OR to extract complete properties.

    :param lines: Preprocessed assertion lines.
    :param n_inputs: Number of input variables.
    :param n_outputs: Number of output variables.
    :param backend: Backend instance for tensor operations.
    :return: List of ``(input_bounds, output_constraints)`` tuples.
    """
    properties = [
        parse_and_block(block, n_inputs, n_outputs, backend)
        for block in iter_and_blocks(lines, with_prefix=True)
    ]

    return properties or [
        (
            backend.full((n_inputs, 2), float("nan"), dtype="float64"),
            [backend.zeros((1, n_outputs + 1), dtype="float64")],
        )
    ]
