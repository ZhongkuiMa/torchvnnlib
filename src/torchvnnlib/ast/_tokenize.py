__docformat__ = ["restructuredtext"]
__all__ = ["tokenize"]

import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Pre-compile regex pattern at module level for performance
# The ordering of the regex is important - match the longest possible token first
TOKEN_PATTERN = re.compile(
    r"""
        \s*                                            # optional whitespace
        (
            -?(?:\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)? |  # float/scientific notation
            -?\d+                                   |  # integer
            [a-zA-Z_][a-zA-Z0-9_-]*                 |  # identifier
            [()]                                    |  # parentheses
            <=|>=|==|!=|=|<|>                       |  # comparison operators
            [-+*/]                                  |  # arithmetic operators
        )
        """,
    re.VERBOSE,
)


def _tokenize_line(args: tuple[int, str]) -> deque[str] | None:
    """Tokenize a single line (helper function for parallel processing).

    :param args: Tuple of (line_index, line_string)
    :return: Deque of tokens, or None if line is empty
    """
    i, line = args
    pos = 0
    tokens = []

    while pos < len(line):
        match = TOKEN_PATTERN.match(line, pos)
        if not match:
            raise ValueError(
                f"Invalid token at line {i + 1}, position {pos}: '{line[pos:].strip()}'"
            )
        token = match.group(1)
        tokens.append(token)
        pos = match.end()

    return deque(tokens) if tokens else None


def tokenize(
    lines: list[str], verbose: bool = False, use_parallel: bool = True
) -> list[deque[str]]:
    """Tokenize lines from a VNNLIB file in parallel.

    The input is a list of lines from a VNNLIB file and this function tokenizes each
    line in parallel for improved performance.

    Tokenizes a string into a list of tokens. The tokens can be:
    - Parentheses: `(`, `)`
    - Identifiers: a word including "declare-const", "Real", "assert", "and", "or".
    - Numbers: a sequence of digits, optionally containing a decimal point
    - Operators: `+`, `-`, `*`, `/`, `<=`, `>=`, `=`, `<`, `>`, etc.
    For other tokens, it will be ignored.

    :param lines: The input string to tokenize.
    :param verbose: Print timing information.
    :param use_parallel: Use parallel processing for large files.
    :return: A list of tokens.
    """
    import time

    # Sequential processing for small files or when parallel is disabled
    if len(lines) < 100 or not use_parallel:
        t = time.perf_counter()
        tokens_list = []
        for i, line in enumerate(lines):
            result = _tokenize_line((i, line))
            if result is not None:
                tokens_list.append(result)
        if verbose:
            print(f"    - Tokenize (sequential): {time.perf_counter() - t:.4f}s")
        return tokens_list

    # For larger files, use parallel processing
    t = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        # Create enumerated args for line tracking
        indexed_lines = list(enumerate(lines))
        results = executor.map(_tokenize_line, indexed_lines)
        # Filter out None results (empty lines)
        tokens_list = [r for r in results if r is not None]
    if verbose:
        print(f"    - Tokenize (parallel): {time.perf_counter() - t:.4f}s")

    return tokens_list
