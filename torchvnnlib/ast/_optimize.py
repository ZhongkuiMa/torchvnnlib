__docformat__ = ["restructuredtext"]
__all__ = ["optimize"]

import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ._expr import *

# Pre-compile regex patterns for performance
VAR_LETTER_PATTERN = re.compile(r"([XY])")
VAR_NUMBER_PATTERN = re.compile(r"(\d+)")


def _else_recursion(expr: Expr, func: Callable) -> Expr:
    if isinstance(expr, UnaryOp):
        expr.arg = func(expr.arg)
    elif isinstance(expr, BinaryOp):
        expr.left = func(expr.left)
        expr.right = func(expr.right)
    elif isinstance(expr, NaryOp):
        for i in range(len(expr.args)):
            expr.args[i] = func(expr.args[i])
    return expr


def _remove_single_and_or(expr: Expr) -> Expr:
    """
    Remove single and from the expression tree.
    """

    if isinstance(expr, And) or isinstance(expr, Or):
        if len(expr.args) == 1 and not expr.args[0].has_output_vars:
            return expr.args[0]
        else:
            for i in range(len(expr.args)):
                expr.args[i] = _remove_single_and_or(expr.args[i])

    return _else_recursion(expr, _remove_single_and_or)


def _simplify_leqgeq(expr: Expr) -> Expr:
    """
    If there is "a <= b" and "b <= a" in the same level of an and operation, remove one
    of them, then combine them into a single expression as a == b.
    """
    if not isinstance(expr, And):
        return _else_recursion(expr, _simplify_leqgeq)

    simplified_args = []
    leq_pairs = {}
    geq_pairs = {}

    for arg in expr.args:
        if isinstance(arg, Leq):
            # Use expression objects directly as key instead of converting to string
            key = (arg.left, arg.right)
            leq_pairs[key] = arg
        elif isinstance(arg, Geq):
            # Use expression objects directly as key instead of converting to string
            key = (arg.left, arg.right)
            geq_pairs[key] = arg
        else:
            simplified_args.append(_simplify_leqgeq(arg))

    # Combine matching Le and Ge into Eq
    used = set()
    for key, le_expr in leq_pairs.items():
        if key in geq_pairs:
            simplified_args.append(Eq(le_expr.left, le_expr.right))
            used.add(key)

    # Add remaining Le/Ge that weren't combined
    for k, v in leq_pairs.items():
        if k not in used:
            simplified_args.append(v)
    for k, v in geq_pairs.items():
        if k not in used:
            simplified_args.append(v)

    return And(simplified_args)


def _get_priority(var: Var) -> int | float:
    """Extract the number from the variable name using cached values."""
    # Use cached var_type and index for O(1) performance
    if not isinstance(var, Var):
        # This is a constant
        return float("inf")

    priority = 0
    if var.var_type == "Y":
        priority += 1000000

    if var.index > 100000000:
        warnings.warn(
            "The number in the variable name is greater than 10000. This will result "
            "in incorrect sorting when printing.",
        )
    priority += var.index

    return priority


def _sort_vars_in_expr(expr: Expr) -> Expr:
    """
    Sort the variables in the expression based on the number in Var's name.
    If nested structures are found, use the first encountered Var.
    """
    if isinstance(expr, NaryOp):
        # Recursively sort all arguments
        expr.args = [_sort_vars_in_expr(arg) for arg in expr.args]
        expr.args = sorted(expr.args, key=lambda x: _get_priority(x))
        return expr

    return _else_recursion(expr, _sort_vars_in_expr)


# def flatten_and_or(expr: Expr) -> Expr:
#     """
#     This function only apply to max 3 deep and/or.
#     This aims to transform
#     (and A B C (or D E (and F G)))
#     into
#     (and
#         (or A B C D)
#         (or A B C E)
#         (or A B C (and F G))
#     )
#     """
#     if not isinstance(expr, And):
#         raise ValueError("This function only applies to And expressions. ")
#
#     if all(not isinstance(arg, Or) for arg in expr.args):
#         return expr
#
#     if any(isinstance(arg, And) for arg in expr.args):
#         raise ValueError("Nested And expressions are not supported.")
#
#     or_expr_list = []
#     other_expr_list = []
#     for arg in expr.args:
#         if isinstance(arg, Or):
#             or_expr_list.append(arg)
#         else:
#             other_expr_list.append(arg)
#
#     new_or_expr = []
#     for or_exprs_comb in itertools.product(*or_expr_list):
#         new_and_expr = deepcopy(other_expr_list)
#         or_exprs_comb = list(or_exprs_comb)
#         new_and_expr.extend(or_exprs_comb)
#         new_or_expr.append(And(new_and_expr))
#
#     # Create a new And expression with the flattened Or expressions
#     new_or_expr = Or(new_or_expr)
#     new_and_expr = And([new_or_expr])
#
#     return new_and_expr


# def fuse_and_and(expr: Expr) -> Expr:
#     """
#     Fuse two And expressions into one.
#     """
#     if isinstance(expr, And):
#         for arg in expr.args:
#             if isinstance(arg, And):
#                 expr.args.remove(arg)
#                 expr.args.extend(arg.args)
#
#     return _else_recursion(expr, fuse_and_and)


def fuse_and_and(expr: Expr) -> Expr:
    """
    Fuse two And expressions into one.
    """
    if isinstance(expr, And):
        # Build new list instead of mutating during iteration (bug fix)
        new_args = []
        for arg in expr.args:
            if isinstance(arg, And):
                new_args.extend(arg.args)  # Flatten nested And
            else:
                new_args.append(arg)
        expr.args = new_args

    return _else_recursion(expr, fuse_and_and)


def optimize(expr: Expr) -> Expr:
    """Optimize expressions with simple parallel processing."""
    is_and = isinstance(expr, And)
    is_or = isinstance(expr, Or)

    if not (is_and or is_or):
        raise ValueError("The expression must be either an And or an Or.")

    def simplify(epr: Expr) -> Expr:
        epr = _remove_single_and_or(epr)
        epr = _simplify_leqgeq(epr)
        return epr

    with ThreadPoolExecutor() as executor:
        expr_list = list(executor.map(simplify, expr.args))

    if is_and:
        expr = And(expr_list)
    elif is_or:
        expr = Or(expr_list)
    else:
        raise ValueError("The expression must be either an And or an Or.")

    expr = _sort_vars_in_expr(expr)

    return expr
