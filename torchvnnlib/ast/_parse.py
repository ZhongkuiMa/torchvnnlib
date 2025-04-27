__docformat__ = "restructuredtext"
__all__ = ["parse"]

import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from ._expr import *

number_pattern = re.compile(r"^-?\d+(\.\d+)?$")

NARY_OPS_MAP = {
    "and": And,
    "or": Or,
    "+": Add,
}
BINARY_OPS_MAP = {
    "<=": Leq,
    ">=": Geq,
    "-": Sub,
    "*": Mul,
    "/": Div,
}


def parse_expr(tokens: deque[str]) -> Expr:
    # NOTE: The deque is really quicker than list by popleft(), thousands of times

    tok = tokens.popleft()

    if tok == "(":
        op = tokens.popleft()

        if op == "assert":
            expr = parse_expr(tokens)
            assert tokens.popleft() == ")"
            return expr

        elif op in NARY_OPS_MAP.keys():
            args = []
            while tokens[0] != ")":
                args.append(parse_expr(tokens))
            tokens.popleft()  # pop ')'

            return NARY_OPS_MAP[op](args)

        elif op in BINARY_OPS_MAP.keys():
            if tokens[0] == "(":
                tokens.popleft()  # pop nested '('
                a = parse_expr(tokens)
                b = parse_expr(tokens)
                tokens.popleft()  # pop nested ')'
            else:
                a = parse_expr(tokens)
                b = parse_expr(tokens)

            tokens.popleft()  # pop ')'

            return BINARY_OPS_MAP[op](a, b)

        else:
            raise ValueError(f"Unknown operator: {op}")

    else:
        try:
            # Handle numbers faster than regex
            return Cst(float(tok))
        except ValueError:
            return Var(tok)


def _parse_single_expr(tokens: list[str]) -> Expr:
    if tokens[0] == "(" and tokens[1] == "assert":
        tokens = tokens[2:-1]  # Remove surrounding (assert ...)

    return parse_expr(deque(tokens))


def _parse_expr_list(tokens_list: list[list[str]]) -> list[Expr]:
    with ThreadPoolExecutor() as executor:
        exprs = list(executor.map(_parse_single_expr, tokens_list))
    return exprs


def _merge_all_exprs_as_and(exprs_list: list[Expr]) -> Expr:
    """
    Merges all expressions in the list into a single expression using logical AND.

    :param exprs_list: A list of Expr objects.
    :return: A single Expr object representing the conjunction of all expressions.
    """
    if len(exprs_list) == 1:
        return exprs_list[0]
    return And(exprs_list)


def parse(tokens_list: list[list[str]]) -> Expr:
    """
    Parses a VNNLIB file and returns a list of tokens.

    :param tokens_list: A list of lists of tokens, where each inner list represents a
        line of tokens.
    :return: An expression object representing the parsed VNNLIB file.
    """
    exprs_list = _parse_expr_list(tokens_list)
    expr = _merge_all_exprs_as_and(exprs_list)

    return expr
