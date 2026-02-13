"""Type5 VNN-LIB Processor: Top-level OR wrapping complete properties."""

__docformat__ = "restructuredtext"
__all__ = ["process_type5"]

import time

from torchvnnlib._backend import Backend, TensorLike
from torchvnnlib.fast_type._utils import parse_and_block


def process_type5(
    lines: list[str],
    n_inputs: int,
    n_outputs: int,
    backend: Backend,
    verbose: bool = False,
) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
    """Fast processor for Type5 VNN-LIB files.

    :param lines: Preprocessed assertion lines
    :param n_inputs: Number of input variables
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
    :param verbose: Print timing information
    :return: Standardized format with complete properties
    """
    t_start = time.perf_counter() if verbose else None

    t = time.perf_counter() if verbose else None
    properties = _parse_top_level_or(lines, n_inputs, n_outputs, backend, verbose)
    if verbose and t is not None:
        print("  Type5 detection:")
        print(f"    OR clauses (properties): {len(properties)}")
        print(f"    Parsing: {time.perf_counter() - t:.4f}s")

    and_properties = [properties]

    if verbose and t_start is not None:
        print(f"  Type5 total time: {time.perf_counter() - t_start:.4f}s")

    return and_properties


def _parse_top_level_or(
    lines: list[str],
    n_inputs: int,
    n_outputs: int,
    backend: Backend,
    verbose: bool = False,
) -> list[tuple[TensorLike, list[TensorLike]]]:
    """Parse top-level OR to extract complete properties.

    :param lines: Preprocessed assertion lines
    :param n_inputs: Number of input variables
    :param n_outputs: Number of output variables
    :param backend: Backend instance for tensor operations
    :param verbose: Print timing information
    :return: List of (input_bounds, output_constraints) tuples
    """
    properties = []

    content = " ".join(lines)
    parts = content.split("(and ")

    for part in parts[1:]:
        if not part:
            continue

        prop = parse_and_block("(and " + part, n_inputs, n_outputs, backend)
        if prop:
            properties.append(prop)

    return properties or [
        (
            backend.full((n_inputs, 2), float("nan"), dtype="float64"),
            [backend.zeros((1, n_outputs + 1), dtype="float64")],
        )
    ]
