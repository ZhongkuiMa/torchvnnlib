"""Combined fast type detection and parsing.

This module combines type detection with data extraction to avoid duplicate regex scanning.
Instead of scanning all lines twice (once for detection, once for parsing), we do it in one pass.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "VNNLIBType",
    "fast_detect_type",
    "fast_detect_and_parse",
    "parse_simple_patterns",
]

import re
import time

from ._utils import (
    VNNLIBType,
    SIMPLE_INPUT_BOUND_PATTERN,
    SIMPLE_OUTPUT_CONSTRAINT_PATTERN,
    SIMPLE_OUTPUT_BOUND_PATTERN,
)

# Top-level OR pattern
TOP_LEVEL_OR_PATTERN = re.compile(r"^\s*\(\s*assert\s+\(\s*or\s+", re.IGNORECASE)


def fast_detect_type(lines: list[str], verbose: bool = False) -> VNNLIBType:
    """
    Fast type detection without parsing - just classifies the VNN-LIB structure.

    This is a lightweight version that only detects the type without parsing data.
    Use this when you only need classification and will handle parsing separately.

    Optimization: Scans from end to beginning since output constraints are at the end.
    Early termination conditions:
    (a) If has_or_and_mixed=True, immediately terminate (TYPE5 cannot be other types)
    (b) When has_simple_input=True and we already have output info, skip remaining inputs

    Args:
        lines: Preprocessed assertion lines
        n_inputs: Number of input variables
        n_outputs: Number of output variables
        verbose: Print timing information

    Returns:
        VNNLIBType enum value
    """
    start_time = time.perf_counter() if verbose else None

    # Pattern counters
    has_simple_input = False  # Pattern (1): (assert X_
    has_simple_output = False  # Pattern (2): (assert Y_
    has_or_and_input = False  # Pattern (3): (assert (or (and X_
    has_or_and_output = False  # Pattern (4): (assert (or (and Y_
    has_or_and_mixed = False  # Pattern (5): (assert (or (and with both X and Y

    # Track if we have any output pattern info for early termination
    has_output_info = False

    # Reverse scan - outputs at end, inputs at beginning
    for line in reversed(lines):
        stripped = line.strip()

        # Check for (assert (or (and ...)) pattern
        if stripped.startswith("(assert (or (and"):
            has_x = "X_" in stripped
            has_y = "Y_" in stripped

            if has_x and has_y:
                # Pattern (5): OR(AND) with both X and Y variables
                # Early termination: TYPE5 cannot be other types
                has_or_and_mixed = True
                break
            elif has_x:
                # Pattern (3): OR(AND) for inputs only
                has_or_and_input = True
                # Early termination: if we already have output info and now have input info, we're done
                if has_output_info:
                    break
            elif has_y:
                # Pattern (4): OR(AND) for outputs only
                has_or_and_output = True
                has_output_info = True
            continue
        elif stripped.startswith("(assert (or"):
            raise ValueError(
                "Unsupported VNN-LIB pattern: Top-level OR without AND blocks."
            )
        # Check for simple assertions
        elif stripped.startswith("(assert"):
            has_x = "X_" in stripped
            has_y = "Y_" in stripped

            if has_x and not has_y:
                # Pattern (1): Simple input bound
                has_simple_input = True
                # Early termination: if we already have output info and now have input info, we're done
                if has_output_info:
                    break
            elif has_y and not has_x:
                # Pattern (2): Simple output constraint
                has_simple_output = True
                has_output_info = True

    # Classify type based on patterns
    vnnlib_type = _classify_type_by_patterns(
        has_simple_input,
        has_simple_output,
        has_or_and_input,
        has_or_and_output,
        has_or_and_mixed,
    )

    if verbose and start_time is not None:
        elapsed = time.perf_counter() - start_time
        print(f"  Fast type detection: {elapsed:.4f}s")
        print(f"    Detected: {vnnlib_type.name}")
        print(
            f"    Patterns: simple_in={has_simple_input}, simple_out={has_simple_output}, "
            f"or_in={has_or_and_input}, or_out={has_or_and_output}, or_mixed={has_or_and_mixed}"
        )

    return vnnlib_type


def parse_simple_patterns(lines: list[str], verbose: bool = False) -> dict:
    """
    Parse simple input bounds and output constraints without type detection.

    This function only parses simple patterns (no OR/AND structures):
    - Simple input bounds: (assert (<=/>=|= X_5 0.123))
    - Simple output constraints: (assert (<=/>= Y_0 Y_1))
    - Simple output bounds: (assert (<=/>=|= Y_5 0.123))

    Use this when you already know the type and only need to extract simple patterns.
    For combined detection and parsing, use fast_detect_and_parse() instead.

    Args:
        lines: Preprocessed assertion lines
        verbose: Print timing information

    Returns:
        Dictionary with:
        - simple_input_bounds: List of (op, var_prefix, idx, value)
        - simple_output_constrs: List of (op, var_prefix1, idx1, var_prefix2, idx2)
        - simple_output_bounds: List of (op, var_prefix, idx, value)
        - complex_lines: Lines that don't match simple patterns
        - complex_indices: Line indices for complex lines
    """
    start_time = time.perf_counter() if verbose else None

    simple_input_bounds = []
    simple_output_constrs = []
    simple_output_bounds = []
    complex_lines = []
    complex_indices = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip OR/AND structures - these are complex
        if stripped.startswith("(assert (or"):
            complex_lines.append(line)
            complex_indices.append(i)
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
                    complex_indices.append(i)
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
                    simple_output_bounds.append(
                        (op, var_prefix, int(idx), float(value))
                    )
                    continue

                # No match - complex
                complex_lines.append(line)
                complex_indices.append(i)
                continue

        # Other lines - complex
        complex_lines.append(line)
        complex_indices.append(i)

    data = {
        "simple_input_bounds": simple_input_bounds,
        "simple_output_constrs": simple_output_constrs,
        "simple_output_bounds": simple_output_bounds,
        "complex_lines": complex_lines,
        "complex_indices": complex_indices,
    }

    if verbose and start_time is not None:
        elapsed = time.perf_counter() - start_time
        print(f"  Parse simple patterns: {elapsed:.4f}s")
        print(f"    Input bounds: {len(simple_input_bounds)}")
        print(f"    Output constraints: {len(simple_output_constrs)}")
        print(f"    Output bounds: {len(simple_output_bounds)}")
        print(f"    Complex lines: {len(complex_lines)}")

    return data


def fast_detect_and_parse(
    lines: list[str], n_inputs: int, n_outputs: int, verbose: bool = False
) -> tuple[VNNLIBType, dict]:
    """
    Combined type detection and data parsing in a single pass.

    Type detection is based on checking characteristic patterns:
    (1) "(assert X_" - simple input bound (only X, no Y)
    (2) "(assert Y_" - simple output constraint (only Y, no X)
    (3) "(assert (or (and X_" - OR of AND blocks for inputs (only X, no Y)
    (4) "(assert (or (and Y_" - OR of AND blocks for outputs (only Y, no X)
    (5) "(assert (or (and" with both X and Y - TYPE5 pattern

    Type1: has only (1) and (2)
    Type2: has only (3) and (2)
    Type3: has only (1) and (4)
    Type4: has only (3) and (4)
    Type5: has only (5)

    Args:
        lines: Preprocessed assertion lines
        n_inputs: Number of input variables
        n_outputs: Number of output variables
        verbose: Print timing information

    Returns:
        - type: VNNLIBType enum value
        - data: Dictionary with parsed data
    """
    start_time = time.perf_counter() if verbose else None

    n_lines = len(lines)
    simple_input_bounds = []
    simple_output_constrs = []
    complex_lines = []
    complex_indices = []

    # Pattern counters
    has_simple_input = False  # Pattern (1): (assert X_
    has_simple_output = False  # Pattern (2): (assert Y_
    has_or_and_input = False  # Pattern (3): (assert (or (and X_
    has_or_and_output = False  # Pattern (4): (assert (or (and Y_
    has_or_and_mixed = False  # Pattern (5): (assert (or (and with both X and Y

    # Single pass through all lines
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for (assert (or (and ...)) pattern
        if stripped.startswith("(assert (or") and "(and" in stripped:
            # For TYPE5 detection, check if BOTH X_ and Y_ appear in the line
            # TYPE5 pattern: (assert (or (and X_... Y_...) ...)) - X and Y mixed in AND clauses
            # TYPE3/TYPE4 pattern: (assert (or (and Y_...) ...)) or (assert (or (and X_...) ...))
            #
            # Since preprocessing ensures each line is a complete sentence, we can simply
            # check for presence of X_ and Y_ in the whole line
            has_x = "X_" in stripped
            has_y = "Y_" in stripped

            if has_x and has_y:
                # Pattern (5): OR(AND) with both X and Y variables
                has_or_and_mixed = True
            elif has_x and not has_y:
                # Pattern (3): OR(AND) for inputs only
                has_or_and_input = True
            elif has_y and not has_x:
                # Pattern (4): OR(AND) for outputs only
                has_or_and_output = True

            complex_lines.append(line)
            complex_indices.append(i)
            continue

        # Check for simple assertions (no nested or/and)
        # Pattern (1): Simple input - has X_ but not Y_, and no (or (and structure
        # Pattern (2): Simple output - has Y_ but not X_, and no (or (and structure
        if stripped.startswith("(assert") and not stripped.startswith("(assert (or"):
            has_x = "X_" in stripped
            has_y = "Y_" in stripped

            if has_x and not has_y:
                # Pattern (1): Simple input bound
                has_simple_input = True
                # Try to parse for fast processing
                match = SIMPLE_INPUT_BOUND_PATTERN.match(line)
                if match:
                    op, var_prefix, idx, value = match.groups()
                    simple_input_bounds.append((op, var_prefix, int(idx), float(value)))
                else:
                    complex_lines.append(line)
                    complex_indices.append(i)
                continue

            if has_y and not has_x:
                # Pattern (2): Simple output constraint
                has_simple_output = True
                # Try to parse for fast processing
                match = SIMPLE_OUTPUT_CONSTRAINT_PATTERN.match(line)
                if match:
                    op, var_prefix1, idx1, var_prefix2, idx2 = match.groups()
                    simple_output_constrs.append(
                        (op, var_prefix1, int(idx1), var_prefix2, int(idx2))
                    )
                else:
                    complex_lines.append(line)
                    complex_indices.append(i)
                continue

        # Other complex patterns
        complex_lines.append(line)
        complex_indices.append(i)

    # Classify type based on patterns
    vnnlib_type = _classify_type_by_patterns(
        has_simple_input,
        has_simple_output,
        has_or_and_input,
        has_or_and_output,
        has_or_and_mixed,
    )

    metadata = {
        "n_lines": n_lines,
        "has_simple_input": has_simple_input,
        "has_simple_output": has_simple_output,
        "has_or_and_input": has_or_and_input,
        "has_or_and_output": has_or_and_output,
        "has_or_and_mixed": has_or_and_mixed,
    }

    # Note: For pure type detection without parsing, use fast_detect_type_only() instead.
    # This function includes optional parsing for TYPE1 simple patterns to avoid re-scanning.
    data = {
        "simple_input_bounds": simple_input_bounds,
        "simple_output_constrs": simple_output_constrs,
        "complex_lines": complex_lines,
        "complex_indices": complex_indices,
        "metadata": metadata,
    }

    if verbose and start_time is not None:
        elapsed = time.perf_counter() - start_time
        print(f"  Fast detect+parse: {elapsed:.4f}s")
        print(f"    Detected: {vnnlib_type.name}")
        print(
            f"    Patterns: simple_in={has_simple_input}, simple_out={has_simple_output}, "
            f"or_in={has_or_and_input}, or_out={has_or_and_output}, or_mixed={has_or_and_mixed}"
        )

    return vnnlib_type, data


def _classify_type_by_patterns(
    has_simple_input: bool,
    has_simple_output: bool,
    has_or_and_input: bool,
    has_or_and_output: bool,
    has_or_and_mixed: bool,
) -> VNNLIBType:
    """Classify VNN-LIB type using pattern matching lookup table.

    Type1: Simple inputs + Simple outputs
    Type2: OR(AND) inputs + Simple outputs
    Type3: Simple inputs + OR(AND) outputs
    Type4: OR(AND) inputs + OR(AND) outputs
    Type5: Mixed OR(AND) with both X and Y
    COMPLEX: Everything else

    Args:
        has_simple_input: Pattern (1) - (assert X_
        has_simple_output: Pattern (2) - (assert Y_
        has_or_and_input: Pattern (3) - (assert (or (and X_
        has_or_and_output: Pattern (4) - (assert (or (and Y_
        has_or_and_mixed: Pattern (5) - (assert (or (and with both X and Y

    Returns:
        VNNLIBType enum value
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
        (False, True, True, False, False): VNNLIBType.TYPE2,
        (True, False, False, True, False): VNNLIBType.TYPE3,
        (False, False, True, True, False): VNNLIBType.TYPE4,
        (False, False, False, False, True): VNNLIBType.TYPE5,
    }

    return type_patterns.get(pattern_key, VNNLIBType.COMPLEX)
