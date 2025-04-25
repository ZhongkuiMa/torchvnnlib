__docformat__ = ["restructuredtext"]
__all__ = ["pre_process_vnnlib"]

import re


def _remove_comments(lines: list[str]) -> list[str]:
    new_lines = []
    for line in lines:
        # Remove comments starting with `;`
        line = re.sub(r";.*", "", line)
        # Remove leading and trailing whitespace
        line = line.strip()
        # Only add non-empty lines
        if line:
            new_lines.append(line)

    return new_lines


def _remove_declare_clauses(lines: list[str]) -> list[str]:
    """
    Removes all lines that declare variables in the VNNLIB file.
    These lines start with `declare-const` and are not needed for parsing expressions.

    :param lines: A list of lines from a VNNLIB file.
    :return: A list of lines without the declare clauses.
    """
    new_lines = []
    for line in lines:
        if "declare-const" not in line:
            new_lines.append(line)
    return new_lines


def _merge_multi_line_expr(lines: list[str]) -> list[str]:
    """
    Merges multiple lines of a VNNLIB expression into a single line.
    If a line ends with a left parenthesis `(`, it indicates that the expression
    continues on the next line. This function will merge all such lines into a single
    expression.
    The number of parantheses will be counted to determine if the expression is
    complete or not.

    :param lines: A list of lines from a VNNLIB file.
    :return: A list of lines where each line is a complete expression.
    """
    new_lines = []
    current_expr = ""
    paren_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):  # skip empty or comment lines
            continue

        current_expr += " " + stripped
        paren_count += stripped.count("(")
        paren_count -= stripped.count(")")

        if paren_count == 0 and current_expr.strip():
            new_lines.append(current_expr.strip())
            current_expr = ""

    if current_expr.strip():
        raise ValueError("Unbalanced parentheses in VNNLIB file.")

    return new_lines


def pre_process_vnnlib(lines: list[str]) -> list[str]:
    """
    Pre-processes the lines of a VNNLIB file by removing comments, declare clauses,
    and merging multi-line expressions.

    :param lines: A list of lines from a VNNLIB file.
    :return: A list of pre-processed lines.
    """
    lines = _remove_comments(lines)
    lines = _remove_declare_clauses(lines)
    lines = _merge_multi_line_expr(lines)

    return lines
