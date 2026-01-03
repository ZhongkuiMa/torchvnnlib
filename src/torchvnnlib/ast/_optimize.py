__docformat__ = ["restructuredtext"]
__all__ = ["optimize"]

import re
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from torchvnnlib.ast._expr import (
    And,
    BinaryOp,
    Cst,
    Eq,
    Expr,
    Geq,
    Leq,
    NaryOp,
    Or,
    UnaryOp,
    Var,
)

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
    """Remove single and from the expression tree."""
    if isinstance(expr, And | Or):
        if len(expr.args) == 1 and not expr.args[0].has_output_vars:
            return expr.args[0]
        for i in range(len(expr.args)):
            expr.args[i] = _remove_single_and_or(expr.args[i])

    return _else_recursion(expr, _remove_single_and_or)


def _combine_leq_geq_pairs(leq_pairs: dict, geq_pairs: dict) -> tuple[list[Expr], set]:
    """Combine matching Leq and Geq pairs into Eq expressions."""
    combined: list[Expr] = []
    used = set()
    for key, le_expr in leq_pairs.items():
        if key in geq_pairs:
            combined.append(Eq(le_expr.left, le_expr.right))
            used.add(key)
    return combined, used


def _simplify_leqgeq(expr: Expr) -> Expr:
    """Simplify <= and >= pairs into == when both are present.

    If there is "a <= b" and "b <= a" in the same level of an and operation,
    remove one and combine them into a single expression as a == b.
    """
    if not isinstance(expr, And):
        return _else_recursion(expr, _simplify_leqgeq)

    simplified_args = []
    leq_pairs = {}
    geq_pairs = {}

    for arg in expr.args:
        if isinstance(arg, Leq):
            key = (arg.left, arg.right)
            leq_pairs[key] = arg
        elif isinstance(arg, Geq):
            key = (arg.left, arg.right)
            geq_pairs[key] = arg
        else:
            simplified_args.append(_simplify_leqgeq(arg))

    # Combine matching Le and Ge into Eq
    combined, used = _combine_leq_geq_pairs(leq_pairs, geq_pairs)
    simplified_args.extend(combined)

    # Add remaining Le/Ge that weren't combined
    simplified_args.extend(v for k, v in leq_pairs.items() if k not in used)
    simplified_args.extend(v for k, v in geq_pairs.items() if k not in used)

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
            stacklevel=2,
        )
    priority += var.index

    return priority


def _sort_vars_in_expr(expr: Expr) -> Expr:
    """Sort the variables in the expression based on the number in Var's name.

    If nested structures are found, use the first encountered Var.
    """
    if isinstance(expr, NaryOp):
        # Recursively sort all arguments
        expr.args = [_sort_vars_in_expr(arg) for arg in expr.args]
        expr.args = sorted(expr.args, key=lambda x: _get_priority(cast(Var, x)))
        return expr

    return _else_recursion(expr, _sort_vars_in_expr)


def optimize(expr: Expr, verbose: bool = False, use_parallel: bool = True) -> Expr:
    """Optimize expressions with optional parallel processing."""
    import time

    is_and = isinstance(expr, And)
    is_or = isinstance(expr, Or)

    if not (is_and or is_or):
        raise ValueError("The expression must be either an And or an Or.")

    def simplify(epr: Expr) -> Expr:
        # Quick check: skip expensive operations for simple leaf expressions
        if isinstance(epr, Leq | Geq | Eq | Var | Cst):
            return epr

        epr = _remove_single_and_or(epr)
        epr = _simplify_leqgeq(epr)
        return epr

    t = time.perf_counter()
    nary_expr = cast(NaryOp, expr)  # Type narrowed by isinstance checks above
    if use_parallel:
        with ThreadPoolExecutor() as executor:
            expr_list = list(executor.map(simplify, nary_expr.args))
        if verbose:
            print(f"    - Simplify (parallel): {time.perf_counter() - t:.4f}s")
    else:
        expr_list = [simplify(arg) for arg in nary_expr.args]
        if verbose:
            print(f"    - Simplify (sequential): {time.perf_counter() - t:.4f}s")

    if is_and:
        expr = And(expr_list)
    elif is_or:
        expr = Or(expr_list)
    else:
        raise ValueError("The expression must be either an And or an Or.")

    t = time.perf_counter()
    expr = _sort_vars_in_expr(expr)
    if verbose:
        print(f"    - Sort vars: {time.perf_counter() - t:.4f}s")

    return expr
