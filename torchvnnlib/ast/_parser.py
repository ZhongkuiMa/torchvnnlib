__docformat__ = "restructuredtext"
__all__ = ["parse"]

import re

from ._expr import *


def _parse_expr_list(tokens_list: list[list[str]]) -> list[Expr]:
    """
    Parse all tokens (a complete expression str) in each line and transform them into
    Expr objects.

    :param tokens_list: A list of lists of tokens, where each inner list represents a
        line of tokens.
    :return: A list of Expr objects.
    """

    def _parse_expr(tokens: list[str]) -> Expr | list[str]:
        """
        Parse a single expression from a list of tokens.

        :param tokens: A list of tokens.
        :return: An Expr object.
        """

        # We have removed the first two tokens which are `(` and `assert`
        # and the last token `)`.
        tok = tokens.pop(0)
        if tok == "(":
            op = tokens.pop(0)
            if op == "assert":
                inner = _parse_expr(tokens)
                assert tokens.pop(0) == ")"
                return inner
            elif op in {"and", "or", "+"}:
                args = []
                while tokens[0] != ")":
                    args.append(_parse_expr(tokens))
                tokens.pop(0)
                if op == "and":
                    return And(args)
                elif op == "or":
                    return Or(args)
                elif op == "+":
                    return Add(args)
                raise ValueError(f"Unknown logical operator: {op}")
            elif op in ["<=", ">=", "-", "*", "/"]:
                has_nested_expr = False
                if tokens[0] == "(":
                    # This is a nested expression
                    tokens.pop(0)
                    has_nested_expr = True
                a = _parse_expr(tokens)
                b = _parse_expr(tokens)
                if has_nested_expr:
                    tokens.pop(0)
                tokens.pop(0)
                if op == "<=":
                    return Leq(a, b)
                elif op == ">=":
                    return Geq(a, b)
                elif op == "-":
                    return Sub(a, b)
                elif op == "*":
                    return Mul(a, b)
                elif op == "/":
                    return Div(a, b)
                raise ValueError(f"Unknown arithmetic operator: {op}")
            else:
                # There maybe some redundant parentheses in the expression
                raise ValueError(f"Unknown operator: {op}")
        elif re.match(r"^-?\d+(\.\d+)?$", tok):
            # Handle numbers (both integers and floats)
            return Cst(float(tok))
        else:
            return Var(tok)

    exprs = []
    for tokens in tokens_list:
        # The first operator is assert and we need extract the inner expression
        # Note that we have make all expressions in a single line
        if tokens[0] == "(" and tokens[1] == "assert":
            tokens = tokens[2:-1]

        expr = _parse_expr(tokens)
        exprs.append(expr)

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

    :param file_path: The path to the VNNLIB file.
    :return: An expression object representing the parsed VNNLIB file.
    """

    exprs_list = _parse_expr_list(tokens_list)
    expr = _merge_all_exprs_as_and(exprs_list)

    return expr
