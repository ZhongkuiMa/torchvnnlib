__docformat__ = "restructuredtext"
__all__ = ["flatten"]

import itertools
from typing import cast

from torchvnnlib.ast._expr import And, Cst, Eq, Expr, Geq, Leq, Or, Var


def _check_input_bound_expr(expr: Expr):
    if not (isinstance(expr, Eq | Leq | Geq)):
        raise ValueError(f"Not an Eq, Leq or Geq expression for input bound: {expr}")
    if not (isinstance(expr.left, Var) or isinstance(expr.right, Cst)):
        raise ValueError(f"Invalid binary expression for input bound: {expr}")


def _check_output_constrs_expr(expr: Expr):
    pass


def _check_bound_or_constr_expr(expr: Expr):
    if expr.has_output_vars:
        _check_output_constrs_expr(expr)
    else:
        _check_input_bound_expr(expr)


def _check_or_expr(or_expr: Or, parent_expr: Expr) -> None:
    """Check Or expression for valid structure."""
    for sub_sub_expr in or_expr.args:
        if isinstance(sub_sub_expr, Or):
            raise ValueError(f"Nested Or expression: {parent_expr}")
        if isinstance(sub_sub_expr, And):
            for sub_sub_sub_expr in sub_sub_expr.args:
                _check_bound_or_constr_expr(sub_sub_sub_expr)
        else:
            _check_bound_or_constr_expr(sub_sub_expr)


def _validate_or_counts(
    sub_expr: Or, num_or_input_bounds: int, num_or_output_constrs: int, parent_expr: Expr
) -> tuple[int, int]:
    """Validate Or expression counts and return updated counts."""
    if sub_expr.has_output_vars:
        num_or_output_constrs += 1
        if num_or_output_constrs > 1:
            raise ValueError(f"More than one Or expression for output constraints: {parent_expr}")
    else:
        num_or_input_bounds += 1
        if num_or_input_bounds > 1:
            raise ValueError(f"More than one Or expression for input bounds: {parent_expr}")
    return num_or_input_bounds, num_or_output_constrs


def _check_and_expr(expr: Expr):
    if not isinstance(expr, And):
        raise ValueError(f"Not an And expression: {expr}")

    has_x = False
    has_y = False
    num_or_input_bounds = 0
    num_or_output_constrs = 0

    for sub_expr in expr.args:
        if sub_expr.has_output_vars:
            has_y = True
        elif sub_expr.has_input_vars:
            has_x = True

        if isinstance(sub_expr, Or):
            num_or_input_bounds, num_or_output_constrs = _validate_or_counts(
                sub_expr, num_or_input_bounds, num_or_output_constrs, expr
            )
            _check_or_expr(sub_expr, expr)
        elif isinstance(sub_expr, And):
            raise ValueError(f"Nested And expression: {expr}\n{sub_expr}")
        else:
            _check_bound_or_constr_expr(sub_expr)

    if not has_x:
        raise ValueError(f"Missing input bound expression: {expr}")
    if not has_y:
        raise ValueError(f"Missing output constraints expression: {expr}")


def _build_input_expr_list(direct_input_exprs: list[Expr], or_input_exprs: list[Expr]) -> list[And]:
    """Build list of AND expressions for input bounds."""
    if direct_input_exprs:
        return [And(direct_input_exprs)]

    input_expr_list = []
    for sub_expr in or_input_exprs:
        if not isinstance(sub_expr, And):
            sub_expr = And([sub_expr])
        input_expr_list.append(sub_expr)
    return input_expr_list


def _build_output_expr_list(
    direct_output_exprs: list[Expr], or_output_exprs: Or | None
) -> list[Or]:
    """Build list of OR expressions for output constraints."""
    if or_output_exprs is None:
        # Only direct output expressions
        if not isinstance(direct_output_exprs[0], And):
            output_expr = And(direct_output_exprs)
        else:
            output_expr = direct_output_exprs[0]
        return [Or([output_expr])]

    if len(direct_output_exprs) == 0:
        # Only Or output expressions
        new_or_output_exprs: list[And] = []
        for expr_item in or_output_exprs.args:
            and_expr = expr_item if isinstance(expr_item, And) else And([expr_item])
            new_or_output_exprs.append(and_expr)
        return [Or(cast(list[Expr], new_or_output_exprs))]

    # Both direct AND Or output expressions exist
    new_or_output_exprs = []
    for expr_item in or_output_exprs.args:
        combined_args = list(direct_output_exprs)
        if isinstance(expr_item, And):
            combined_args.extend(expr_item.args)
        else:
            combined_args.append(expr_item)
        new_or_output_exprs.append(And(combined_args))
    return [Or(cast(list[Expr], new_or_output_exprs))]


def _flatten_and_expr(expr: And) -> Or:
    _check_and_expr(expr)

    # Collect direct expressions
    direct_input_exprs = [
        sub_expr
        for sub_expr in expr.args
        if not isinstance(sub_expr, Or) and not sub_expr.has_output_vars
    ]
    direct_output_exprs = [
        sub_expr
        for sub_expr in expr.args
        if not isinstance(sub_expr, Or) and sub_expr.has_output_vars
    ]

    # Collect Or expressions
    or_input_exprs: list[Expr] = []
    or_output_exprs = None
    for sub_expr in expr.args:
        if isinstance(sub_expr, Or):
            if not sub_expr.has_output_vars and not or_input_exprs:
                or_input_exprs.extend(sub_expr.args)
            elif sub_expr.has_output_vars and or_output_exprs is None:
                or_output_exprs = sub_expr

    # Build expression lists
    input_expr_list = _build_input_expr_list(direct_input_exprs, or_input_exprs)
    output_expr_list = _build_output_expr_list(direct_output_exprs, or_output_exprs)

    # Combine all expressions
    all_or_comb: list[And] = []
    for input_expr, output_expr_or in itertools.product(input_expr_list, output_expr_list):
        all_or_comb.append(And([input_expr, output_expr_or]))

    return Or(cast(list[Expr], all_or_comb))


def flatten(expr: Expr) -> And:
    """Flatten expression tree into normalized and-or form.

    We only support the following types of expressions:
    (1) (and ...input constraints... ...output constraints...)
    (2) (and ...input constraints... (or ...output constraints...))
    (3) (and (or ...input constraints...) ...output constraints...)
    (4) (and (or ...input constraints...) (or ...output constraints...))
    (5) (or
            (and ...input constraints... ...output constraints...)
            (and ...input constraints... ...output constraints...)
            ...
        ).

    We aim to convert the above expressions to a list of expressions,
    Each such expression has the form of:
    (and (and ...input constraints...) (or (and ...output constraints...) ...))
    """
    if isinstance(expr, And):
        or_expr = _flatten_and_expr(expr)
        return And([or_expr])

    # If not And, must be Or based on the constraint checking
    assert isinstance(expr, Or), f"Expression must be And or Or, got {type(expr)}"
    or_expr_list = []
    for and_expr in expr.args:
        or_expr = _flatten_and_expr(cast(And, and_expr))
        or_expr_list.append(or_expr.args[0])
    or_expr = Or(cast(list[Expr], or_expr_list))
    return And([or_expr])
