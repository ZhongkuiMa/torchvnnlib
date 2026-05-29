"""Fast type detection and simple-pattern parsing for VNN-LIB files."""

__docformat__ = "restructuredtext"
__all__ = [
    "ParsedPatterns",
    "VNNLIBType",
    "fast_detect_type",
    "parse_simple_patterns",
]

import logging
import time
from typing import TypedDict

from torchvnnlib._logging import _enable_verbose
from torchvnnlib.fast_type._enums import VNNLIBType
from torchvnnlib.fast_type._utils import (
    SIMPLE_INPUT_BOUND_PATTERN,
    SIMPLE_OUTPUT_BOUND_PATTERN,
    SIMPLE_OUTPUT_CONSTRAINT_PATTERN,
)


# Public structural type for :func:`parse_simple_patterns` results. Replaces
# the prior bare ``dict[str, Any]`` so static type-checkers catch key typos.
# The runtime value is still a plain ``dict`` -- existing callers that
# subscript by string key keep working unchanged.
class ParsedPatterns(TypedDict):
    """Buckets returned by :func:`parse_simple_patterns`.

    The three ``simple_*`` lists carry already-classified constraint tuples;
    ``complex_lines`` carries the residual lines that did not match any simple
    regex and must fall through to the AST path.
    """

    simple_input_bounds: list[tuple[str, str, int, float]]
    simple_output_constrs: list[tuple[str, str, int, str, int]]
    simple_output_bounds: list[tuple[str, str, int, float]]
    complex_lines: list[str]


_logger = logging.getLogger(__name__)


def _classify_var_presence(stripped: str) -> tuple[bool, bool, bool]:
    """Check stripped line for X_ / Y_ and return ``(is_input, is_output, is_mixed)``."""
    has_x = "X_" in stripped
    has_y = "Y_" in stripped
    return has_x and not has_y, has_y and not has_x, has_x and has_y


def _handle_or_and_pattern(
    stripped: str,
    pattern_flags: dict[str, bool],
) -> bool:
    """Update flags for an ``(assert (or (and ...))`` line.

    :param stripped: Stripped line content.
    :param pattern_flags: Flag dict updated in place.
    :return: ``True`` only on mixed X+Y signal (caller may early-exit).
    """
    is_input, is_output, is_mixed = _classify_var_presence(stripped)
    if is_mixed:
        pattern_flags["has_or_and_mixed"] = True
        return True
    if is_input:
        pattern_flags["has_or_and_input"] = True
    elif is_output:
        pattern_flags["has_or_and_output"] = True
        pattern_flags["has_output_info"] = True
    return False


def _handle_simple_pattern(
    stripped: str,
    pattern_flags: dict[str, bool],
) -> bool:
    """Update flags for a flat ``(assert ...)`` line.

    :param stripped: Stripped line content.
    :param pattern_flags: Flag dict updated in place.
    :return: Always ``False`` (no early-exit signal from simple lines).
    """
    is_input, is_output, _is_mixed = _classify_var_presence(stripped)
    if is_input:
        pattern_flags["has_simple_input"] = True
    elif is_output:
        pattern_flags["has_simple_output"] = True
        pattern_flags["has_output_info"] = True
    return False


def _update_type_patterns_from_line(
    stripped: str,
    pattern_flags: dict[str, bool],
) -> bool:
    """Update pattern flags from a single line.

    :param stripped: Stripped line content.

    :param pattern_flags: Dict of flags to update.

    :return: True if should break from loop, False otherwise
    """
    if stripped.startswith("(assert (or (and"):
        return _handle_or_and_pattern(stripped, pattern_flags)
    if stripped.startswith("(assert (or"):
        raise ValueError("Unsupported VNN-LIB pattern: Top-level OR without AND blocks.")
    if stripped.startswith("(assert"):
        return _handle_simple_pattern(stripped, pattern_flags)
    return False


def fast_detect_type(lines: list[str], verbose: bool = False) -> VNNLIBType:
    """Classify VNN-LIB structural shape without building an AST.

    :param lines: Preprocessed assertion lines.
    :param verbose: Emit timing logs.
    :return: :class:`VNNLIBType` enum value.
    """
    start_time = time.perf_counter()

    if verbose:
        _enable_verbose()

    pattern_flags = {
        "has_simple_input": False,
        "has_simple_output": False,
        "has_or_and_input": False,
        "has_or_and_output": False,
        "has_or_and_mixed": False,
        "has_output_info": False,
    }

    # Full scan to avoid false negatives when inputs/outputs are interleaved.
    # Early-exit on mixed OR/AND (signals TYPE5 / COMPLEX boundary already crossed).
    for line in lines:
        if (
            _update_type_patterns_from_line(line.strip(), pattern_flags)
            and pattern_flags["has_or_and_mixed"]
        ):
            break

    vnnlib_type = _classify_type_by_patterns(
        pattern_flags["has_simple_input"],
        pattern_flags["has_simple_output"],
        pattern_flags["has_or_and_input"],
        pattern_flags["has_or_and_output"],
        pattern_flags["has_or_and_mixed"],
    )

    if verbose:
        elapsed = time.perf_counter() - start_time
        _logger.info(f"  Fast type detection: {elapsed:.4f}s")
        _logger.info(f"    Detected: {vnnlib_type.name}")
        _logger.info(
            f"    Patterns: simple_in={pattern_flags['has_simple_input']}, "
            f"simple_out={pattern_flags['has_simple_output']}, "
            f"or_in={pattern_flags['has_or_and_input']}, "
            f"or_out={pattern_flags['has_or_and_output']}, "
            f"or_mixed={pattern_flags['has_or_and_mixed']}"
        )

    return vnnlib_type


def parse_simple_patterns(lines: list[str], verbose: bool = False) -> ParsedPatterns:
    """Parse simple input/output bound patterns without building an AST.

    Recognises three flat regex shapes:

    - Simple input bound: ``(assert (<=/>=|= X_5 0.123))``
    - Simple output constraint: ``(assert (<=/>= Y_0 Y_1))``
    - Simple output bound: ``(assert (<=/>=|= Y_5 0.123))``

    Anything else lands in ``complex_lines`` for the AST fallback. Call this
    once the file's :class:`VNNLIBType` is known and you only need the simple
    buckets.

    :param lines: Preprocessed assertion lines.
    :param verbose: Print timing information.
    :return: :class:`ParsedPatterns` mapping bucket names to constraint
        tuples (input/output bounds, output constraints) and residual lines.
    """
    start_time = time.perf_counter()

    if verbose:
        _enable_verbose()

    simple_input_bounds = []
    simple_output_constrs = []
    simple_output_bounds = []
    complex_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip OR/AND structures - these are complex
        if stripped.startswith("(assert (or"):
            complex_lines.append(line)
            continue

        # Try to parse simple assertions
        if stripped.startswith("(assert"):
            has_x = "X_" in stripped
            has_y = "Y_" in stripped

            # Simple input bound: (assert (<=/>=|= X_5 0.123))
            if has_x and not has_y:
                match = SIMPLE_INPUT_BOUND_PATTERN.match(line)
                if match:
                    op, var_prefix, idx, value = match.groups()
                    simple_input_bounds.append((op, var_prefix, int(idx), float(value)))
                else:
                    complex_lines.append(line)
                continue

            # Simple output patterns
            if has_y and not has_x:
                # Try simple output constraint: (assert (<=/>= Y_0 Y_1))
                match = SIMPLE_OUTPUT_CONSTRAINT_PATTERN.match(line)
                if match:
                    op, var_prefix1, idx1, var_prefix2, idx2 = match.groups()
                    simple_output_constrs.append(
                        (op, var_prefix1, int(idx1), var_prefix2, int(idx2))
                    )
                    continue
                # Try simple output bound: (assert (<=/>=|= Y_5 0.123))
                match = SIMPLE_OUTPUT_BOUND_PATTERN.match(line)
                if match:
                    op, var_prefix, idx, value = match.groups()
                    simple_output_bounds.append((op, var_prefix, int(idx), float(value)))
                    continue

            # No match - complex
            complex_lines.append(line)
            continue

        # Other lines - complex
        complex_lines.append(line)

    data: ParsedPatterns = {
        "simple_input_bounds": simple_input_bounds,
        "simple_output_constrs": simple_output_constrs,
        "simple_output_bounds": simple_output_bounds,
        "complex_lines": complex_lines,
    }

    if verbose:
        elapsed = time.perf_counter() - start_time
        _logger.info(f"  Parse simple patterns: {elapsed:.4f}s")
        _logger.info(f"    Input bounds: {len(simple_input_bounds)}")
        _logger.info(f"    Output constraints: {len(simple_output_constrs)}")
        _logger.info(f"    Output bounds: {len(simple_output_bounds)}")
        _logger.info(f"    Complex lines: {len(complex_lines)}")

    return data


def _classify_type_by_patterns(
    has_simple_input: bool,
    has_simple_output: bool,
    has_or_and_input: bool,
    has_or_and_output: bool,
    has_or_and_mixed: bool,
) -> VNNLIBType:
    """Classify VNN-LIB type using pattern matching lookup table.

    Type1: Simple inputs + Simple outputs.
    Type2: OR(AND) inputs + Simple outputs.
    Type3: Simple inputs + OR(AND) outputs.
    Type4: OR(AND) inputs + OR(AND) outputs.
    Type5: Mixed OR(AND) with both X and Y.
    COMPLEX: Everything else.

    :param has_simple_input: Pattern (1) -- ``(assert X_``.
    :param has_simple_output: Pattern (2) -- ``(assert Y_``.
    :param has_or_and_input: Pattern (3) -- ``(assert (or (and X_``.
    :param has_or_and_output: Pattern (4) -- ``(assert (or (and Y_``.
    :param has_or_and_mixed: Pattern (5) -- ``(assert (or (and`` with both X and Y.
    :return: VNNLIBType enum value.
    """
    # Pattern lookup: (simple_in, simple_out, or_in, or_out, mixed) -> Type
    pattern_key = (
        has_simple_input,
        has_simple_output,
        has_or_and_input,
        has_or_and_output,
        has_or_and_mixed,
    )

    type_patterns = {
        (True, True, False, False, False): VNNLIBType.TYPE1,
        (True, False, False, True, False): VNNLIBType.TYPE2,
        (False, True, True, False, False): VNNLIBType.TYPE3,
        (False, False, True, True, False): VNNLIBType.TYPE4,
        (False, False, False, False, True): VNNLIBType.TYPE5,
    }

    return type_patterns.get(pattern_key, VNNLIBType.COMPLEX)
