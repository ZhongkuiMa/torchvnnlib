__docformat__ = ["restructuredtext"]
__all__ = ["tokenize"]

import re


def tokenize(lines: list[str]) -> list[list[str]]:
    """
    The input is a list of lines from a VNNLIB file and this funciton tokenizes each
    line.
    Tokenizes a string into a list of tokens. The tokens can be:
    - Parentheses: `(`, `)`
    - Identifiers: a word including "declare-const", "Real", "assert", "and", "or".
    - Numbers: a sequence of digits, optionally containing a decimal point
    - Operators: `+`, `-`, `*`, `/`, `<=`, `>=`, `=`, `<`, `>`, etc.
    For other tokens, it will be ignored.

    :param lines: The input string to tokenize.
    :return:  A list of tokens.
    """
    # Regex pattern to match valid tokens
    # The ordering of the regex is important.
    # We need to match the longest possible token first.
    token_pattern = re.compile(
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

    tokens_list = []
    for i, line in enumerate(lines):
        pos = 0
        tokens = []
        while pos < len(line):
            match = token_pattern.match(line, pos)
            if not match:
                raise ValueError(
                    f"Invalid token at line {i+1}, position {pos}: '{line[pos:].strip()}'"
                )
            token = match.group(1)
            tokens.append(token)
            pos = match.end()
        if tokens:
            tokens_list.append(tokens)

    return tokens_list
