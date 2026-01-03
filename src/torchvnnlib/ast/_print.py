__docformat__ = ["restructuredtext"]
__all__ = ["print_expr"]

from torchvnnlib.ast._expr import (
    Add,
    And,
    Cst,
    Div,
    Eq,
    Expr,
    Geq,
    Leq,
    Mul,
    Or,
    Sub,
    Var,
)


def _print_binary_op(
    op_symbol: str, left: Expr, right: Expr, indent: int, prefix: str, with_newline: bool = False
) -> None:
    """Print binary operation."""
    print(f"{prefix}({op_symbol} ", end="")
    print_expr(left, indent + 1)
    print(" ", end="")
    print_expr(right, indent + 1)
    print(")", end="")
    if with_newline:
        print()


def _print_nary_op(op_symbol: str, args: list[Expr], indent: int) -> None:
    """Print n-ary operation."""
    print(f"({op_symbol} ", end="")
    for arg in args:
        print_expr(arg, indent + 1)
        print(" ", end="")
    print(")", end="")


def _print_logical_op(op_symbol: str, args: list[Expr], indent: int, prefix: str) -> None:
    """Print logical operation."""
    print(f"{prefix}({op_symbol} ")
    for arg in args:
        print_expr(arg, indent + 1)
    print(f"{prefix})")


def print_expr(expr: Expr, indent: int = 0) -> None:
    prefix = "  " * indent

    if isinstance(expr, Cst):
        print(f"{expr.value}", end="")
    elif isinstance(expr, Var):
        print(f"{expr.name}", end="")
    elif isinstance(expr, Eq | Leq | Geq):
        ops = {Eq: "=", Leq: "<=", Geq: ">="}
        _print_binary_op(ops[type(expr)], expr.left, expr.right, indent, prefix, with_newline=True)
    elif isinstance(expr, Add):
        _print_nary_op("+", expr.args, indent)
    elif isinstance(expr, Sub):
        _print_binary_op("-", expr.left, expr.right, indent, prefix)
    elif isinstance(expr, Mul):
        _print_binary_op("*", expr.left, expr.right, indent, prefix)
    elif isinstance(expr, Div):
        _print_binary_op("/", expr.left, expr.right, indent, prefix)
    elif isinstance(expr, And | Or):
        op_sym = "and" if isinstance(expr, And) else "or"
        _print_logical_op(op_sym, expr.args, indent, prefix)
    else:
        raise RuntimeError(f"Unknown expr: {expr}")
