__docformat__ = ["restructuredtext"]
__all__ = ["preprocess_vnnlib"]

import re

# Pre-compile regex patterns for performance
COMMENT_PATTERN = re.compile(r";.*")
ASSERT_PATTERN = re.compile(r"^\(assert\s+.*\)$")


def _remove_comments(lines: list[str]) -> list[str]:
    new_lines = []

    for line in lines:
        # Remove comments starting with `;` (using pre-compiled pattern)
        line = COMMENT_PATTERN.sub("", line)
        # Remove leading and trailing whitespace
        line = line.strip()
        # Only add non-empty lines
        if line:
            new_lines.append(line)

    return new_lines


def _remove_declare_clauses(lines: list[str]) -> tuple[list[str], int, int]:
    """
    Removes all lines that declare variables in the VNNLIB file.
    These lines start with `declare-const` and are not needed for parsing expressions.

    :param lines: A list of lines from a VNNLIB file.
    :return: A list of lines without the declare clauses.
    """
    new_lines = []
    n_inputs = 0
    n_outputs = 0

    for line in lines:
        if "declare-const" not in line:
            new_lines.append(line)
        elif "X" in line:
            # Count the number of inputs
            n_inputs += 1
        elif "Y" in line:
            # Count the number of outputs
            n_outputs += 1

    return new_lines, n_inputs, n_outputs


def _merge_multi_line_expr(lines: list[str]) -> list[str]:
    """
    Merges multiple lines of a VNNLIB expression into a single line.
    If a line ends with a left parenthesis `(`, it indicates that the expression
    continues on the next line. This function will merge all such lines into a single
    expression.
    The number of parentheses will be counted to determine if the expression is
    complete or not.

    :param lines: A list of lines from a VNNLIB file.
    :return: A list of lines where each line is a complete expression.
    """
    new_lines = []
    current_parts = []  # Use list accumulation for O(n) instead of O(n²)
    paren_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):  # skip empty or comment lines
            continue

        current_parts.append(stripped)
        paren_count += stripped.count("(")
        paren_count -= stripped.count(")")

        if paren_count == 0 and current_parts:
            new_lines.append(" ".join(current_parts))  # Join once at the end
            current_parts = []

    if current_parts:
        raise ValueError("Unbalanced parentheses in VNNLIB file.")

    return new_lines


def _check_illegal_lines(lines: list[str]) -> None:
    """
    Checks for illegal lines in the VNNLIB file.
    Illegal lines are those that do not match the expected format of VNNLIB expressions.

    :param lines: A list of lines from a VNNLIB file.
    :raises ValueError: If an illegal line is found.
    """
    for line in lines:
        # Check if the line has a form as "(assert <expr>)" (using pre-compiled pattern)
        if not ASSERT_PATTERN.match(line):
            raise ValueError(f"Illegal line found in VNNLIB file: {line}")


def preprocess_vnnlib(lines: list[str]) -> tuple[list[str], int, int]:
    """
    Pre-processes the lines of a VNNLIB file by removing comments, declare clauses,
    and merging multi-line expressions.

    :param lines: A list of lines from a VNNLIB file.
    :return: A list of pre-processed lines.
    """
    lines = _remove_comments(lines)
    lines, n_inputs, n_outputs = _remove_declare_clauses(lines)
    lines = _merge_multi_line_expr(lines)
    _check_illegal_lines(lines)

    return lines, n_inputs, n_outputs
