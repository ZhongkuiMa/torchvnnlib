__docformat__ = "restructuredtext"
__all__ = ["preprocess_vnnlib"]

import re

COMMENT_PATTERN = re.compile(r";.*")
ASSERT_PATTERN = re.compile(r"^\(assert\s+.*\)$")


def _remove_comments(lines: list[str]) -> list[str]:
    """Remove comment lines from VNN-LIB file.

    :param lines: Lines from VNN-LIB file
    :return: Lines without comments
    """
    new_lines = []

    for line in lines:
        line = COMMENT_PATTERN.sub("", line)
        line = line.strip()
        if line:
            new_lines.append(line)

    return new_lines


def _remove_declare_clauses(lines: list[str]) -> tuple[list[str], int, int]:
    """Remove variable declaration lines and count variables.

    :param lines: Lines from VNN-LIB file
    :return: Tuple of (lines without declarations, input count, output count)
    """
    new_lines = []
    n_inputs = 0
    n_outputs = 0

    for line in lines:
        if "declare-const" not in line:
            new_lines.append(line)
        elif "X" in line:
            n_inputs += 1
        elif "Y" in line:
            n_outputs += 1

    return new_lines, n_inputs, n_outputs


def _merge_multi_line_expr(lines: list[str]) -> list[str]:
    """Merge multi-line VNN-LIB expressions into single lines.

    Uses parenthesis counting to determine expression boundaries.

    :param lines: Lines from VNN-LIB file
    :return: Lines where each line is a complete expression
    """
    new_lines = []
    current_parts = []
    paren_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue

        current_parts.append(stripped)
        paren_count += stripped.count("(")
        paren_count -= stripped.count(")")

        if paren_count == 0 and current_parts:
            new_lines.append(" ".join(current_parts))
            current_parts = []

    if current_parts:
        raise ValueError("Unbalanced parentheses in VNN-LIB file")

    return new_lines


def _check_illegal_lines(lines: list[str]) -> None:
    """Check for illegal lines in VNN-LIB file.

    :param lines: Lines from VNN-LIB file
    :raises ValueError: If illegal line format is found
    """
    for line in lines:
        if not ASSERT_PATTERN.match(line):
            raise ValueError(f"Illegal line in VNN-LIB file: {line}")


def preprocess_vnnlib(lines: list[str]) -> tuple[list[str], int, int]:
    """Preprocess VNN-LIB file lines.

    Removes comments, declarations, and merges multi-line expressions.

    :param lines: Lines from VNN-LIB file
    :return: Tuple of (preprocessed lines, input count, output count)
    """
    lines = _remove_comments(lines)
    lines, n_inputs, n_outputs = _remove_declare_clauses(lines)
    lines = _merge_multi_line_expr(lines)
    _check_illegal_lines(lines)

    return lines, n_inputs, n_outputs
