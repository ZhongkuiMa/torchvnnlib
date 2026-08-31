"""Preprocessing of raw VNN-LIB file content before parsing."""

__docformat__ = "restructuredtext"
__all__ = ["preprocess_vnnlib"]

import re
from typing import Final

COMMENT_PATTERN = re.compile(r";.*")
ASSERT_PATTERN = re.compile(r"^\(assert\s+.*\)$")
DECL_INPUT_PATTERN = re.compile(r"declare-const\s+X_\d+")
DECL_OUTPUT_PATTERN = re.compile(r"declare-const\s+Y_\d+")

# Structural guard against pathological/adversarial input. A merged expression
# nested deeper than this is almost certainly malformed. Expression breadth is
# deliberately unrestricted: official VNN-COMP properties contain shallow
# top-level disjunctions larger than 50 MiB, and the source is already resident
# in ``lines`` before this pass begins.
_MAX_PAREN_DEPTH: Final[int] = 10_000


def _remove_comments(lines: list[str]) -> list[str]:
    """Strip ``;`` comments and blank lines.

    :param lines: Raw VNN-LIB lines.
    :return: Lines with comments removed and blanks dropped.
    """
    new_lines = []
    for raw in lines:
        stripped = COMMENT_PATTERN.sub("", raw).strip()
        if stripped:
            new_lines.append(stripped)
    return new_lines


def _remove_declare_clauses(lines: list[str]) -> tuple[list[str], int, int]:
    """Drop ``declare-const`` lines while counting input/output variables.

    :param lines: Lines after comment removal.
    :return: Tuple of (lines without declarations, n_inputs, n_outputs).
    """
    new_lines = []
    n_inputs = 0
    n_outputs = 0
    for line in lines:
        if "declare-const" not in line:
            new_lines.append(line)
        elif DECL_INPUT_PATTERN.search(line):
            n_inputs += 1
        elif DECL_OUTPUT_PATTERN.search(line):
            n_outputs += 1
    return new_lines, n_inputs, n_outputs


def _merge_multi_line_expr(lines: list[str]) -> list[str]:
    """Glue multi-line S-expressions into single lines via paren balance.

    :param lines: Lines after declaration removal.
    :return: One complete S-expression per output line.
    :raises ValueError: On unbalanced parentheses or nesting deeper than
        :data:`_MAX_PAREN_DEPTH`.
    """
    new_lines = []
    current_parts: list[str] = []
    paren_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue

        current_parts.append(stripped)
        paren_count += stripped.count("(")
        paren_count -= stripped.count(")")

        if paren_count > _MAX_PAREN_DEPTH:
            raise ValueError(f"Paren nesting exceeds {_MAX_PAREN_DEPTH}; input likely malformed.")
        if paren_count < 0:
            raise ValueError("Unbalanced parentheses in VNN-LIB file (extra ')').")

        if paren_count == 0 and current_parts:
            new_lines.append(" ".join(current_parts))
            current_parts = []

    if current_parts:
        raise ValueError("Unbalanced parentheses in VNN-LIB file (missing ')').")

    return new_lines


def _check_illegal_lines(lines: list[str]) -> None:
    """Raise if any line is not a single top-level ``(assert ...)`` form.

    :param lines: Merged expressions, one per element.
    :raises ValueError: On any non-``assert`` form.
    """
    for line in lines:
        if not ASSERT_PATTERN.match(line):
            raise ValueError(f"Illegal line in VNN-LIB file: {line}")


def preprocess_vnnlib(lines: list[str]) -> tuple[list[str], int, int]:
    """Run full preprocess pipeline on raw VNN-LIB lines.

    :param lines: Raw VNN-LIB file lines.
    :return: Tuple of (one-assert-per-line list, n_inputs, n_outputs).
    """
    lines = _remove_comments(lines)
    lines, n_inputs, n_outputs = _remove_declare_clauses(lines)
    lines = _merge_multi_line_expr(lines)
    _check_illegal_lines(lines)
    return lines, n_inputs, n_outputs
