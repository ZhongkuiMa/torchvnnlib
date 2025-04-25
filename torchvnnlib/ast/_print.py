__docformat__ = ["restructuredtext"]
from ._expr import *

__all__ = ["print_ast"]


def print_ast(expr: Expr, indent: int = 0):
    prefix = "  " * indent
    if isinstance(expr, Cst):
        # If there is a \n in front of print, remove it
        print(f"{expr.value}", end="")
    elif isinstance(expr, Var):
        print(f"{expr.name}", end="")
    elif isinstance(expr, Eq):
        print(f"{prefix}(= ", end="")
        print_ast(expr.left, indent + 1)
        print(" ", end="")
        print_ast(expr.right, indent + 1)
        print(f")")
    elif isinstance(expr, Leq):
        print(f"{prefix}(<= ", end="")
        print_ast(expr.left, indent + 1)
        print(" ", end="")
        print_ast(expr.right, indent + 1)
        print(f")")
    elif isinstance(expr, Geq):
        print(f"{prefix}(>= ", end="")
        print_ast(expr.left, indent + 1)
        print(" ", end="")
        print_ast(expr.right, indent + 1)
        print(f")")
    elif isinstance(expr, Add):
        print(f"(+ ", end="")
        for arg in expr.args:
            print_ast(arg, indent + 1)
            print(" ", end="")
        print(f")", end="")
    elif isinstance(expr, Sub):
        print(f"(- ", end="")
        print_ast(expr.left, indent + 1)
        print(" ", end="")
        print_ast(expr.right, indent + 1)
        print(f")", end="")
    elif isinstance(expr, Mul):
        print(f"(* ", end="")
        print_ast(expr.left, indent + 1)
        print(" ", end="")
        print_ast(expr.right, indent + 1)
        print(f")", end="")
    elif isinstance(expr, Div):
        print(f"(/ ", end="")
        print_ast(expr.left, indent + 1)
        print(" ", end="")
        print_ast(expr.right, indent + 1)
        print(f")", end="")
    elif isinstance(expr, And):
        print(f"{prefix}(and ")
        for arg in expr.args:
            print_ast(arg, indent + 1)
        print(f"{prefix})")
    elif isinstance(expr, Or):
        print(f"{prefix}(or ")
        for arg in expr.args:
            print_ast(arg, indent + 1)
        print(f"{prefix})")
    else:
        raise RuntimeError(f"Unknown expr: {expr}")
