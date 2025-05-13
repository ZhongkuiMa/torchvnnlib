__docformat__ = "restructuredtext"
__all__ = ["flatten"]

import itertools

from ._expr import *


def _check_input_bound_expr(expr: Expr):
    if not (isinstance(expr, Eq) or isinstance(expr, Leq) or isinstance(expr, Geq)):
        raise ValueError(f"Not an Eq, Leq or Geq expression for input bound: {expr}")
    if not (isinstance(expr.left, Var) or isinstance(expr.right, Cst)):
        raise ValueError(f"Invalid binary expression for input bound: {expr}")


def _check_output_constrs_expr(expr: Expr):
    pass


def _check_bound_or_constr_expr(expr: Expr):
    if "Y" in expr.__repr__():
        _check_output_constrs_expr(expr)
    else:
        _check_input_bound_expr(expr)


def _check_and_expr(expr: Expr):
    if not isinstance(expr, And):
        raise ValueError(f"Not an And expression: {expr}")

    has_X = False
    has_Y = False
    num_or_input_bounds = 0
    num_or_output_constrs = 0
    for sub_expr in expr.args:
        if "Y" in sub_expr.__repr__():
            has_Y = True
        elif "X" in sub_expr.__repr__():
            has_X = True

        if isinstance(sub_expr, Or):
            if "Y" in sub_expr.__repr__():
                num_or_output_constrs += 1
                if num_or_output_constrs > 1:
                    raise ValueError(
                        f"More than one Or expression for output constraints: {expr}"
                    )
            else:
                num_or_input_bounds += 1
                if num_or_input_bounds > 1:
                    raise ValueError(
                        f"More than one Or expression for input bounds: {expr}"
                    )
            for sub_sub_expr in sub_expr.args:
                if isinstance(sub_sub_expr, Or):
                    raise ValueError(f"Nested Or expression: {expr}")
                if isinstance(sub_sub_expr, And):
                    for sub_sub_sub_expr in sub_sub_expr.args:
                        _check_bound_or_constr_expr(sub_sub_sub_expr)
                else:
                    _check_bound_or_constr_expr(sub_sub_expr)
        elif isinstance(sub_expr, And):
            raise ValueError(f"Nested And expression: {expr}")
        else:
            _check_bound_or_constr_expr(sub_expr)

    if not has_X:
        raise ValueError(f"Missing input bound expression: {expr}")
    if not has_Y:
        raise ValueError(f"Missing output constraints expression: {expr}")


def _flatten_and_expr(expr: And) -> Or:
    _check_and_expr(expr)

    # Collect direct input bound expressions
    direct_input_exprs = []
    for sub_expr in expr.args:
        if not isinstance(sub_expr, Or) and "Y" not in sub_expr.__repr__():
            direct_input_exprs.append(sub_expr)

    # Collect direct output constraints expressions
    direct_output_exprs = []
    for sub_expr in expr.args:
        if not isinstance(sub_expr, Or) and "Y" in sub_expr.__repr__():
            direct_output_exprs.append(sub_expr)

    # Collect input bound expressions in the Or expression
    or_input_exprs = []
    if len(direct_input_exprs) == 0:
        for sub_expr in expr.args:
            if isinstance(sub_expr, Or) and "Y" not in sub_expr.__repr__():
                or_input_exprs.extend(sub_expr.args)
                break

    # Collect output constraints expressions in the Or expression
    or_output_exprs = None
    if len(direct_output_exprs) == 0:
        for sub_expr in expr.args:
            if isinstance(sub_expr, Or) and "Y" in sub_expr.__repr__():
                or_output_exprs = sub_expr
                break

    if direct_input_exprs:
        input_expr_list = [And(direct_input_exprs)]
    else:
        input_expr_list = []
        for sub_expr in or_input_exprs:
            if not (isinstance(sub_expr, And)):
                sub_expr = And(sub_expr)
            input_expr_list.append(sub_expr)

    if direct_output_exprs:
        if not isinstance(direct_output_exprs[0], And):
            output_expr = And(direct_output_exprs)
        else:
            output_expr = direct_output_exprs[0]
        output_expr_list = [Or([output_expr])]
    else:
        new_or_output_exprs = []
        for output_expr in or_output_exprs.args:
            if not isinstance(output_expr, And):
                output_expr = And([output_expr])
            new_or_output_exprs.append(output_expr)
        or_output_exprs = new_or_output_exprs
        output_expr_list = [Or(or_output_exprs)]

    all_or_comb = []
    for input_expr, output_expr in itertools.product(input_expr_list, output_expr_list):
        all_or_comb.append(And([input_expr, output_expr]))

    all_or_properties = Or(all_or_comb)

    return all_or_properties


def flatten(expr: Expr) -> And:
    """
    We only support three types of expressions:
    (1) (and ...input constraints... ...output constraints...)
    (2) (and ...input constraints... (or ...output constraints...))
    (3) (and (or ...input constraints...) ...output constraints...)
    (4) (and (or ...input constraints...) (or ...output constraints...))
    (5) (or
            (and ...input constraints... ...output constraints...)
            (and ...input constraints... ...output constraints...)
            ...
        )

    We aim to convert the above expressions to a list of expressions,
    Each such expression has the form of:
    (and (and ...input constraints...) (or (and ...output constraints...) ...))
    """

    if isinstance(expr, And):
        or_expr = _flatten_and_expr(expr)
        return And([or_expr])

    or_expr_list = []
    expr: Or
    and_expr: And
    for and_expr in expr.args:
        or_expr = _flatten_and_expr(and_expr)
        or_expr_list.append(or_expr.args[0])
    or_expr = Or(or_expr_list)
    return And([or_expr])
