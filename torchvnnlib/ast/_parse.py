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


def parse_tokens_list(tokens_list: list[deque[str]]) -> list[Expr]:
    with ThreadPoolExecutor() as executor:
        exprs = list(executor.map(parse_tokens, tokens_list))

    return exprs


def parse_tokens(tokens: deque[str]) -> Expr:
    # NOTE: The deque is really quicker than list by popleft(), thousands of times

    tok = tokens.popleft()

    if tok == "(":
        op = tokens.popleft()

        if op == "assert":
            expr = parse_tokens(tokens)
            tokens.popleft()  # Expecting ')'
            return expr

        elif op in NARY_OPS_MAP:
            args = []
            while tokens[0] != ")":
                args.append(parse_tokens(tokens))
            tokens.popleft()  # Expecting ')'
            return NARY_OPS_MAP[op](args)

        elif op in BINARY_OPS_MAP:
            if tokens[0] == "(":
                tokens.popleft()  # Expecting '('
                a = parse_tokens(tokens)
                b = parse_tokens(tokens)
                tokens.popleft()  # Expecting ')'
            else:
                a = parse_tokens(tokens)
                b = parse_tokens(tokens)
            tokens.popleft()  # Expecting ')'
            return BINARY_OPS_MAP[op](a, b)

        else:
            raise ValueError(f"Unknown operator: {op}")

    else:
        try:
            # Handle numbers faster than regex
            return Cst(float(tok))
        except ValueError:
            return Var(tok)


def _merge_all_exprs_as_and(exprs_list: list[Expr]) -> Expr:
    """
    Merges all expressions in the list into a single expression using logical AND.

    :param exprs_list: A list of Expr objects.
    :return: A single Expr object representing the conjunction of all expressions.
    """
    if len(exprs_list) == 1:
        return exprs_list[0]
    return And(exprs_list)


def parse(tokens_list: list[deque[str]]) -> Expr:
    """
    Parses a VNNLIB file and returns a list of tokens.

    :param tokens_list: A list of lists of tokens, where each inner list represents a
        line of tokens.
    :return: An expression object representing the parsed VNNLIB file.
    """
    exprs_list = parse_tokens_list(tokens_list)
    # Merge all expressions into a single And expression
    expr = _merge_all_exprs_as_and(exprs_list)

    return expr
