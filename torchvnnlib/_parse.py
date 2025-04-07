__docformat__ = "restructuredtext"
__all__ = ["load_vnnlib"]

import re

import torch
from torch import Tensor

# Extract the content contained in "(" and ")"
pattern_parentheses = re.compile(r"^\((.+?)\)$")
# Extract the content contained in "(declare-const " and ")"
pattern_declare_const = re.compile(r"^declare-const\s*(.+?)\s*Real$")
# Extract the content contained in "(assert " and ")"
pattern_assert = re.compile(r"^assert\s*(.+?)\s*$")


class InvalidStatement(ValueError):
    def __init__(self, info: str = ""):
        super().__init__(f"Invalid statement: {info}")


def _remove_comments(line: str) -> str:
    # Remove the string after ";"
    return line.split(";")[0]


def _remove_special_characters(line: str) -> str:
    # Remove the "\n" at the end of the line
    line = line.replace("\n", "")
    # Remove the space at the beginning and end of the line
    line = line.strip()
    # Remove the repeated spaces
    line = re.sub(r"\s+", " ", line)
    # Remove the space on the either side of the parentheses
    line = (
        line.replace(" (", "(")
        .replace("( ", "(")
        .replace(") ", ")")
        .replace(" " ")", ")")
    )
    return line


def _replace_ineq_signs(line: str) -> str:
    line = line.replace("> ", ">=").replace("< ", "<=")
    return line


def _extract_code_lines(lines: list[str]) -> list[str]:
    new_lines = []
    for line in lines:
        line = _remove_comments(line)
        line = _remove_special_characters(line)
        if not line:
            continue
        new_lines.append(line)
    return new_lines


def _count_paired_parenthese(line: str) -> int:
    return line.count("(") - line.count(")")


def _parse_one_statement(line: str, lines: iter) -> str:
    counter_parentheses = 0
    counter_parentheses += _count_paired_parenthese(line)
    if counter_parentheses < 0:
        raise ValueError("Parentheses are not paired correctly.")

    if counter_parentheses == 0:
        statement = line
        statement = pattern_parentheses.search(statement).group(1)

        return statement

    statement = line
    while counter_parentheses > 0:
        line = next(lines)
        line = _remove_comments(line)
        line = _remove_special_characters(line)
        counter_parentheses += _count_paired_parenthese(line)
        statement += line

    statement = _replace_ineq_signs(statement)
    statement = pattern_parentheses.search(statement).group(1)

    return statement


def _parse_declare_const(statement: str, variable_counter: dict[str, int]) -> bool:
    statement_group = pattern_declare_const.findall(statement)
    if len(statement_group) == 0:
        return False  # Not a declare-const statement
    if len(statement_group) > 1:
        raise InvalidStatement(statement)

    var_name, var_idx = statement_group[0].split("_")
    if var_name not in variable_counter:
        raise InvalidStatement(statement)

    variable_counter[var_name] = max(variable_counter[var_name], int(var_idx))

    return True


def _parse_input_bounds(
    op: str,
    var1: str,
    var2: str,
    input_bounds: list[list[float]],
    variable_counter: dict[str, int],
):
    n = variable_counter["X"]
    if not input_bounds:
        # We must append rather than assign the list to keep the reference.
        for _ in range(n):
            input_bounds.append([-float("inf"), float("inf")])

    value = float(var2)
    idx = int(var1.split("_")[1])
    if op == ">=":
        input_bounds[idx][0] = value
    elif op == "<=":
        input_bounds[idx][1] = value
    else:
        raise InvalidStatement(f"Invalid operator {op}")


def _parse_output_constrs(
    op: str,
    var1: str,
    var2: str,
    output_constrs: list[list[float]],
    variable_counter: dict[str, int],
):
    n = variable_counter["Y"]
    constr = [0.0] * (n + 1)
    # NOTE: We use Ax + b >= 0.
    if var1.startswith("Y") and var2.startswith("Y"):
        idx1, idx2 = int(var1.split("_")[1]), int(var2.split("_")[1])
        if op == ">=":
            constr[idx1], constr[idx2] = 1.0, -1.0
        elif op == "<=":
            constr[idx1], constr[idx2] = -1.0, 1.0
        else:
            raise InvalidStatement(f"Invalid operator {op}")
    elif var1.startswith("Y"):
        idx = int(var1.split("_")[1])
        if op == ">=":
            constr[idx], constr[-1] = 1.0, float(var2)
        elif op == "<=":
            constr[idx], constr[n] = -1.0, -float(var2)
        else:
            raise InvalidStatement(f"Invalid operator {op}")
    else:
        raise InvalidStatement(f"Invalid variable {var1} and {var2}")

    output_constrs.append(constr)


def _parse_simple_assert(
    statement: str,
    input_bounds: list[list[float]],
    output_constrs: list[list[float]],
    variable_counter: dict[str, int],
):
    # Without "and" or "or"
    op, var1, var2 = statement.split(" ")
    if var1.startswith("X"):
        _parse_input_bounds(op, var1, var2, input_bounds, variable_counter)
    elif var1.startswith("Y") or var2.startswith("Y"):
        _parse_output_constrs(op, var1, var2, output_constrs, variable_counter)
    else:
        raise InvalidStatement(statement)


def _parse_complex_assert(
    statement: str,
    input_bounds_groups: list[list[list[float]]],
    output_constrs_groups: list[list[list[float]]],
    variable_counter: dict[str, int],
):
    # With "and" or "or"
    n_or = statement.count("or")
    if n_or != 1:
        raise InvalidStatement(statement)

    new_or_groups = []
    or_groups = statement.split("))(and")

    for or_group in or_groups:
        or_group = or_group.replace("or", "").replace("and", "")
        and_groups = or_group.split(")(")
        new_and_groups = []

        for and_group in and_groups:
            and_group = and_group.replace("(", "").replace(")", "")
            new_and_groups.append(and_group)
        new_or_groups.append(new_and_groups)

    for or_groups in new_or_groups:
        input_bounds = []
        constrs = []
        for and_group in or_groups:
            _parse_simple_assert(and_group, input_bounds, constrs, variable_counter)
        if input_bounds:
            input_bounds_groups.append(input_bounds)
        if constrs:
            output_constrs_groups.append(constrs)


def _parse_assert(
    statement: str,
    input_bounds_groups: list[list[list[float]]],
    output_constrs_groups: list[list[list[float]]],
    variable_counter: dict[str, int],
):
    statement_group = pattern_assert.findall(statement)
    if len(statement_group) == 0:
        return False
    if len(statement_group) > 1:
        raise InvalidStatement(statement)

    statement = statement_group[0]

    if "or" not in statement and "and" not in statement:
        statement = pattern_parentheses.search(statement).group(1)
        # Here, we set the first element of the list as an empty list to store the
        # simple assert statements.
        input_bounds = input_bounds_groups[0]
        constrs = output_constrs_groups[0]
        _parse_simple_assert(statement, input_bounds, constrs, variable_counter)
        return

    _parse_complex_assert(
        statement, input_bounds_groups, output_constrs_groups, variable_counter
    )


def _check_input_bounds(input_bounds_groups: list[list[float]]):
    for input_bounds in input_bounds_groups:
        for lower, upper in input_bounds:
            if lower > upper or lower == -float("inf") or upper == float("inf"):
                raise ValueError(f"Invalid bound: [{lower}, {upper}]")


def load_vnnlib(file_path: str) -> dict[str, list[Tensor]]:
    """
    Parse the .vnnlib file and return the input bounds and output constraints.

    1. input_bounds: A (n, 2) array where n is the number of input variables. Each row
    contains the lower and upper bounds of the corresponding input variable.

    2. output_constrs: A (p, q, n+1) array where p is the number of "or" output
    constraints, q is the number of "and" output constraints, and n is the number
    of input variables. Each row contains the coefficients of the corresponding
    output constraint. The last element of each row is the constant term, i.e., we
    accept the form of Ax + b >= 0.

    :param file_path: The path to the .vnnlib file.

    :return: A tuple of two numpy arrays. The first array contains the input bounds
        and the second array contains the output constraints.
    """

    if not file_path.endswith(".vnnlib"):
        raise ValueError(f"The file must be a .vnnlib file but got {file_path}")

    with open(file_path, "r") as f:
        lines = f.readlines()

    lines = _extract_code_lines(lines)
    lines = iter(lines)

    variable_counter = {"X": 0, "Y": 0}

    # Suppose all declare-const statements are before all assert statements.

    statement = None
    while line := next(lines, None):
        statement = _parse_one_statement(line, lines)
        if _parse_declare_const(statement, variable_counter):
            continue
        break

    for key in variable_counter:
        variable_counter[key] += 1

    # Here, we set the first element of the list as an empty list to store the simple
    # assert statements.
    input_bounds_groups = [[]]
    output_constrs_groups = [[]]

    _parse_assert(
        statement, input_bounds_groups, output_constrs_groups, variable_counter
    )

    while line := next(lines, None):
        statement = _parse_one_statement(line, lines)
        _parse_assert(
            statement, input_bounds_groups, output_constrs_groups, variable_counter
        )

    _check_input_bounds(input_bounds_groups)

    # Because we set the first element is a default empty list, we need to remove it if
    # it is empty.
    if not input_bounds_groups[0]:
        input_bounds_groups.pop(0)
    if not output_constrs_groups[0]:
        output_constrs_groups.pop(0)

    input_bounds_groups = [torch.tensor(t) for t in input_bounds_groups]
    output_constrs_groups = [torch.tensor(t) for t in output_constrs_groups]

    result = {"inputs": input_bounds_groups, "outputs": output_constrs_groups}

    return result
