"""Recursive descent parser for tokenized VNN-LIB S-expressions."""

__docformat__ = "restructuredtext"
__all__ = ["parse"]

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Final

from torchvnnlib._logging import _enable_verbose
from torchvnnlib.ast._expr import Add, And, Cst, Div, Expr, Geq, Leq, Mul, Or, Sub, Var

_logger = logging.getLogger(__name__)

OPS_MAP: Final[dict[str, tuple[type[Expr], bool]]] = {
    "and": (And, True),
    "or": (Or, True),
    "+": (Add, True),
    "<=": (Leq, False),
    ">=": (Geq, False),
    "-": (Sub, False),
    "*": (Mul, False),
    "/": (Div, False),
}

# Soft recursion guard: VNN-LIB nesting beyond this is almost certainly hostile
# input. Raised before Python's own RecursionError so users get a useful message
# instead of a stack-frame dump.
_MAX_PARSE_DEPTH: Final[int] = 500


def _parse_tokens_list(
    tokens_list: list[deque[str]], verbose: bool = False, use_parallel: bool = True
) -> list[Expr]:
    """Parse tokens with optional parallel processing.

    :param tokens_list: One deque of tokens per S-expression line.
    :param verbose: Emit timing logs.
    :param use_parallel: Run parses in a thread pool.
    :return: One ``Expr`` per input deque.
    """
    if verbose:
        _enable_verbose()

    if not use_parallel:
        t = time.perf_counter()
        exprs = [_parse_tokens(tokens) for tokens in tokens_list]
        if verbose:
            _logger.info(f"    - Parse tokens (sequential): {time.perf_counter() - t:.4f}s")
        return exprs

    t = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        exprs = list(executor.map(_parse_tokens, tokens_list))
        if verbose:
            _logger.info(f"    - Parse tokens (parallel): {time.perf_counter() - t:.4f}s")
    return exprs


def _parse_tokens(tokens: deque[str], depth: int = 0) -> Expr:
    """Parse one S-expression token stream.

    Uses ``deque.popleft`` (O(1)) for token consumption -- list-based pop is
    thousands of times slower at this scale.

    :param tokens: Tokens for one S-expression.
    :param depth: Current recursion depth (internal).
    :return: Root ``Expr`` of the parsed subtree.
    :raises ValueError: On unknown operator or excessive nesting.
    """
    if depth > _MAX_PARSE_DEPTH:
        raise ValueError(
            f"VNN-LIB expression exceeds max nesting depth ({_MAX_PARSE_DEPTH}); "
            "input is likely malformed or adversarial."
        )

    tok = tokens.popleft()

    if tok == "(":
        op = tokens.popleft()

        if op == "assert":
            expr = _parse_tokens(tokens, depth + 1)
            tokens.popleft()  # Expecting ')'
            return expr

        op_info = OPS_MAP.get(op)
        if op_info is not None:
            op_class, is_nary = op_info

            if is_nary:
                args = []
                while tokens[0] != ")":
                    args.append(_parse_tokens(tokens, depth + 1))
                tokens.popleft()  # Expecting ')'
                return op_class(args)  # type: ignore[call-arg]
            # Binary operator. Optional extra parens around the operand pair.
            if tokens[0] == "(":
                tokens.popleft()
                a = _parse_tokens(tokens, depth + 1)
                b = _parse_tokens(tokens, depth + 1)
                tokens.popleft()  # Expecting ')'
            else:
                a = _parse_tokens(tokens, depth + 1)
                b = _parse_tokens(tokens, depth + 1)
            tokens.popleft()  # Expecting ')'
            return op_class(a, b)  # type: ignore[call-arg]
        raise ValueError(f"Unknown operator: {op}")

    # Numeric literal first, then identifier.
    try:
        return Cst(float(tok))
    except ValueError:
        return Var(tok)


def _merge_all_exprs_as_and(exprs_list: list[Expr]) -> Expr:
    """Merge a list of expressions under a single ``And`` (or pass through if singleton).

    :param exprs_list: Top-level expressions parsed from the file.
    :return: Single ``Expr`` representing the conjunction.
    """
    if len(exprs_list) == 1:
        return exprs_list[0]
    return And(exprs_list)


def parse(tokens_list: list[deque[str]], verbose: bool = False, use_parallel: bool = True) -> Expr:
    """Parse a VNN-LIB token stream into an ``Expr`` tree.

    :param tokens_list: One deque of tokens per assertion line.
    :param verbose: Emit timing logs.
    :param use_parallel: Run per-line parses in a thread pool.
    :return: Root ``Expr`` of the parsed file (an ``And`` for multi-assert files).
    """
    if verbose:
        _enable_verbose()

    # No global ``sys.setrecursionlimit`` adjustment: mutating it from a worker
    # racing with other threads (parallel parse, host application) corrupts a
    # process-wide setting. The :data:`_MAX_PARSE_DEPTH` soft cap rejects
    # adversarial nesting long before CPython's own limit is hit.
    exprs_list = _parse_tokens_list(tokens_list, verbose=verbose, use_parallel=use_parallel)

    t = time.perf_counter()
    expr = _merge_all_exprs_as_and(exprs_list)
    if verbose:
        _logger.info(f"    - Merge exprs: {time.perf_counter() - t:.4f}s")

    return expr
