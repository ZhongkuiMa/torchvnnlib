"""Optimization and simplification of VNN-LIB AST expressions."""

__docformat__ = "restructuredtext"
__all__ = ["optimize"]

import logging
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Final, cast

from torchvnnlib._logging import _enable_verbose
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
    Var,
)

_logger = logging.getLogger(__name__)

# Sort priority: output vars (Y_*) always sort after input vars (X_*).
_Y_PRIORITY_OFFSET: Final[int] = 1_000_000
# Var indices above this break sort priority because Y_OFFSET is exceeded.
_MAX_VAR_INDEX: Final[int] = 100_000_000


def _else_recursion(expr: Expr, func: Callable[[Expr], Expr]) -> Expr:
    """Recurse ``func`` into operand slots of ``expr`` in place."""
    if isinstance(expr, BinaryOp):
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


def _combine_leq_geq_pairs(
    leq_pairs: dict[tuple[Expr, Expr], Leq],
    geq_pairs: dict[tuple[Expr, Expr], Geq],
) -> tuple[list[Expr], set[tuple[Expr, Expr]]]:
    """Combine matching ``Leq`` and ``Geq`` pairs into ``Eq``."""
    combined: list[Expr] = []
    used: set[tuple[Expr, Expr]] = set()
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

    simplified_args: list[Expr] = []
    leq_pairs: dict[tuple[Expr, Expr], Leq] = {}
    geq_pairs: dict[tuple[Expr, Expr], Geq] = {}

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


def _get_priority(var: Expr) -> int | float:
    """Return sort priority for ``var``; constants sort last (``inf``)."""
    if not isinstance(var, Var):
        return float("inf")

    priority = 0
    if var.var_type == "Y":
        priority += _Y_PRIORITY_OFFSET

    if var.index > _MAX_VAR_INDEX:
        warnings.warn(
            f"Variable index exceeds {_MAX_VAR_INDEX}; sort order may be wrong.",
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
        expr.args = sorted(expr.args, key=_get_priority)
        return expr

    return _else_recursion(expr, _sort_vars_in_expr)


def optimize(expr: Expr, verbose: bool = False, use_parallel: bool = True) -> Expr:
    """Optimize expressions with optional parallel processing.

    :param expr: Root ``And`` or ``Or`` expression.
    :param verbose: Emit timing logs.
    :param use_parallel: Run per-arg simplification in a thread pool.
    :return: Optimized expression tree.
    :raises ValueError: If ``expr`` is not ``And`` or ``Or``.
    """
    if verbose:
        _enable_verbose()

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
            _logger.info(f"    - Simplify (parallel): {time.perf_counter() - t:.4f}s")
    else:
        expr_list = [simplify(arg) for arg in nary_expr.args]
        if verbose:
            _logger.info(f"    - Simplify (sequential): {time.perf_counter() - t:.4f}s")

    # ``is_and`` / ``is_or`` are exhaustive thanks to the early guard above.
    expr = And(expr_list) if is_and else Or(expr_list)

    t = time.perf_counter()
    expr = _sort_vars_in_expr(expr)
    if verbose:
        _logger.info(f"    - Sort vars: {time.perf_counter() - t:.4f}s")

    return expr
