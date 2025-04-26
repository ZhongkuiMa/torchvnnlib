__docformat__ = "restructuredtext"
__all__ = ["parse"]

import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from ._expr import *

number_pattern = re.compile(r"^-?\d+(\.\d+)?$")


class Parser:
    __slots__ = ("tokens",)  # to save memory

    def __init__(self, tokens: deque[str]):
        self.tokens = tokens

    def parse_expr(self) -> Expr:
        tokens = self.tokens

        tok = tokens.popleft()

        if tok == "(":
            op = tokens.popleft()

            if op == "assert":
                expr = self.parse_expr()
                assert tokens.popleft() == ")"
                return expr

            elif op in {"and", "or", "+"}:
                args = []
                while tokens[0] != ")":
                    args.append(self.parse_expr())
                tokens.popleft()  # pop ')'
                if op == "and":
                    return And(args)
                elif op == "or":
                    return Or(args)
                else:  # "+"
                    return Add(args)

            elif op in {"<=", ">=", "-", "*", "/"}:
                if tokens[0] == "(":
                    tokens.popleft()  # pop nested '('
                    a = self.parse_expr()
                    b = self.parse_expr()
                    tokens.popleft()  # pop nested ')'
                else:
                    a = self.parse_expr()
                    b = self.parse_expr()

                tokens.popleft()  # pop ')'

                if op == "<=":
                    return Leq(a, b)
                elif op == ">=":
                    return Geq(a, b)
                elif op == "-":
                    return Sub(a, b)
                elif op == "*":
                    return Mul(a, b)
                else:  # "/"
                    return Div(a, b)

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
    # NOTE: The deque is really quicker than list, thousands of times
    parser = Parser(deque(tokens))
    return parser.parse_expr()


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
