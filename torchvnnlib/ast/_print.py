__docformat__ = ["restructuredtext"]
__all__ = ["print_ast"]

from ._expr import *


def print_ast(expr: Expr, indent: int = 0):
    prefix = "  " * indent

    def print_binary_op(op_symbol, left, right):
        print(f"{prefix}({op_symbol} ", end="")
        print_ast(left, indent + 1)
        print(" ", end="")
        print_ast(right, indent + 1)
        print(")", end="")

    def print_nary_op(op_symbol, args):
        print(f"({op_symbol} ", end="")
        for arg in args:
            print_ast(arg, indent + 1)
            print(" ", end="")
        print(")", end="")

    def print_logical_op(op_symbol, args):
        print(f"{prefix}({op_symbol} ")
        for arg in args:
            print_ast(arg, indent + 1)
        print(f"{prefix})")

    if isinstance(expr, Cst):
        print(f"{expr.value}", end="")
    elif isinstance(expr, Var):
        print(f"{expr.name}", end="")
    elif isinstance(expr, Eq):
        print_binary_op("=", expr.left, expr.right)
        print()
    elif isinstance(expr, Leq):
        print_binary_op("<=", expr.left, expr.right)
        print()
    elif isinstance(expr, Geq):
        print_binary_op(">=", expr.left, expr.right)
        print()
    elif isinstance(expr, Add):
        print_nary_op("+", expr.args)
    elif isinstance(expr, Sub):
        print_binary_op("-", expr.left, expr.right)
    elif isinstance(expr, Mul):
        print_binary_op("*", expr.left, expr.right)
    elif isinstance(expr, Div):
        print_binary_op("/", expr.left, expr.right)
    elif isinstance(expr, And):
        print_logical_op("and", expr.args)
    elif isinstance(expr, Or):
        print_logical_op("or", expr.args)
    else:
        raise RuntimeError(f"Unknown expr: {expr}")
