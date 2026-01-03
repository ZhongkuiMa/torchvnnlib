__docformat__ = "restructuredtext"
__all__ = ["parse"]

import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from torchvnnlib.ast._expr import Add, And, Cst, Div, Expr, Geq, Leq, Mul, Or, Sub, Var

# Pre-compiled pattern (defined but not used - kept for compatibility)
number_pattern = re.compile(r"^-?\d+(\.\d+)?$")

# Combine operator maps for faster lookup (single dictionary access)
OPS_MAP = {
    "and": (And, True),  # (class, is_nary)
    "or": (Or, True),
    "+": (Add, True),
    "<=": (Leq, False),
    ">=": (Geq, False),
    "-": (Sub, False),
    "*": (Mul, False),
    "/": (Div, False),
}


def _parse_tokens_list(
    tokens_list: list[deque[str]], verbose: bool = False, use_parallel: bool = True
) -> list[Expr]:
    """Parse tokens with optional parallel processing."""
    import time

    if not use_parallel:
        t = time.perf_counter()
        exprs = [_parse_tokens(tokens) for tokens in tokens_list]
        if verbose:
            print(f"    - Parse tokens (sequential): {time.perf_counter() - t:.4f}s")
        return exprs

    t = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        exprs = list(executor.map(_parse_tokens, tokens_list))
    if verbose:
        print(f"    - Parse tokens (parallel): {time.perf_counter() - t:.4f}s")
    return exprs


def _parse_tokens(tokens: deque[str]) -> Expr:
    """Parse tokens with optimized operator lookup."""
    # NOTE: The deque is really quicker than list by popleft(), thousands of times

    tok = tokens.popleft()

    if tok == "(":
        op = tokens.popleft()

        if op == "assert":
            expr = _parse_tokens(tokens)
            tokens.popleft()  # Expecting ')'
            return expr

        # Optimized: Single dictionary lookup instead of two checks
        op_info = OPS_MAP.get(op)
        if op_info is not None:
            op_class, is_nary = op_info

            if is_nary:
                # N-ary operator (and, or, +)
                args = []
                while tokens[0] != ")":
                    args.append(_parse_tokens(tokens))
                tokens.popleft()  # Expecting ')'
                return op_class(args)  # type: ignore[no-any-return]
            # Binary operator (<=, >=, -, *, /)
            # Check if arguments are wrapped in extra parentheses
            if tokens[0] == "(":
                tokens.popleft()  # Expecting '('
                a = _parse_tokens(tokens)
                b = _parse_tokens(tokens)
                tokens.popleft()  # Expecting ')'
            else:
                a = _parse_tokens(tokens)
                b = _parse_tokens(tokens)
            tokens.popleft()  # Expecting ')'
            return op_class(a, b)  # type: ignore[no-any-return]
        raise ValueError(f"Unknown operator: {op}")

    # Optimized: Try float conversion directly (faster than regex)
    try:
        return Cst(float(tok))
    except ValueError:
        return Var(tok)


def _merge_all_exprs_as_and(exprs_list: list[Expr]) -> Expr:
    """Merge all expressions in the list into a single expression using logical AND.

    :param exprs_list: A list of Expr objects.
    :return: A single Expr object representing the conjunction of all expressions.
    """
    if len(exprs_list) == 1:
        return exprs_list[0]
    return And(exprs_list)


def parse(tokens_list: list[deque[str]], verbose: bool = False, use_parallel: bool = True) -> Expr:
    """Parse a VNNLIB file and return an expression object.

    :param tokens_list: A list of lists of tokens, where each inner list represents a
        line of tokens.
    :param verbose: Print timing information.
    :param use_parallel: Use parallel processing.
    :return: An expression object representing the parsed VNNLIB file.
    """
    import time

    exprs_list = _parse_tokens_list(tokens_list, verbose=verbose, use_parallel=use_parallel)

    t = time.perf_counter()
    # Merge all expressions into a single And expression
    expr = _merge_all_exprs_as_and(exprs_list)
    if verbose:
        print(f"    - Merge exprs: {time.perf_counter() - t:.4f}s")

    return expr
