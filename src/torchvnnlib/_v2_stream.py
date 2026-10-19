"""Bounded streaming parser for one canonical VNN-LIB 2.0 grammar."""

from __future__ import annotations

__docformat__ = "restructuredtext"
__all__ = ["STREAMING_ABI", "try_parse_canonical_query"]

import gzip
import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from torchvnnlib._query import (
    VNNLibCapabilities,
    VNNLibComparator,
    VNNLibConstant,
    VNNLibConstraint,
    VNNLibExpression,
    VNNLibInputDomain,
    VNNLibNetwork,
    VNNLibParseBackend,
    VNNLibQuery,
    VNNLibTensor,
    VNNLibTensorKind,
    VNNLibVariable,
    VNNLibVersion,
    _pack_input_domain,
)

STREAMING_ABI = 1

_MAX_LINE_CHARS = 1024 * 1024
_TOKEN_RE = re.compile(r"\s+|;[^\r\n]*|<=|>=|!=|==|[()\[\],<>]|[^\s()\[\],;<>!=]+")
# The official 1.0.2 grammar gives Float32 and Real comparisons a decimal
# literal with digits on both sides of the point. Keeping this recognizer
# narrower than Python's float syntax prevents the fast path from accepting a
# query that the language authority rejects.
_NUMBER_RE = re.compile(r"-?\d+\.\d+\Z")
_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_COMPARATORS = {
    "<": VNNLibComparator.LT,
    "<=": VNNLibComparator.LE,
    "==": VNNLibComparator.EQ,
    "!=": VNNLibComparator.NE,
    ">=": VNNLibComparator.GE,
    ">": VNNLibComparator.GT,
}
_REVERSED_COMPARATOR = {
    VNNLibComparator.LT: VNNLibComparator.GT,
    VNNLibComparator.LE: VNNLibComparator.GE,
    VNNLibComparator.EQ: VNNLibComparator.EQ,
    VNNLibComparator.NE: VNNLibComparator.NE,
    VNNLibComparator.GE: VNNLibComparator.LE,
    VNNLibComparator.GT: VNNLibComparator.LT,
}
_DTYPES = {"float32": "F32", "real": "Real"}


class _NotCanonicalError(Exception):
    """Signal that the official parser must own this query."""


def _tokens(stream: TextIO) -> Iterator[str]:
    """Yield tokens while retaining at most one bounded source line."""
    while True:
        line = stream.readline(_MAX_LINE_CHARS + 1)
        if not line:
            return
        if len(line) > _MAX_LINE_CHARS:
            raise _NotCanonicalError("source line exceeds streaming grammar limit")
        position = 0
        while position < len(line):
            match = _TOKEN_RE.match(line, position)
            if match is None:
                raise _NotCanonicalError("unsupported lexical token")
            token = match.group(0)
            position = match.end()
            if token.isspace() or token.startswith(";"):
                continue
            yield token


class _TokenStream:
    """Single-token lookahead over the bounded lexer."""

    def __init__(self, tokens: Iterator[str]) -> None:
        self._tokens = tokens
        self._lookahead: str | None = None

    def peek(self) -> str | None:
        """Return the next token without consuming it."""
        if self._lookahead is None:
            self._lookahead = next(self._tokens, None)
        return self._lookahead

    def take(self) -> str:
        """Consume and return one token."""
        token = self.peek()
        if token is None:
            raise _NotCanonicalError("unexpected end of file")
        self._lookahead = None
        return token

    def expect(self, expected: str) -> None:
        """Consume one exact grammar token."""
        if self.take() != expected:
            raise _NotCanonicalError(f"expected {expected!r}")


def _symbol(tokens: _TokenStream) -> str:
    value = tokens.take()
    if _SYMBOL_RE.fullmatch(value) is None:
        raise _NotCanonicalError("unsupported symbol")
    return value


def _shape(tokens: _TokenStream) -> tuple[int, ...]:
    tokens.expect("[")
    if tokens.peek() == "]":
        tokens.take()
        return ()
    dimensions: list[int] = []
    while True:
        token = tokens.take()
        if not token.isdecimal() or int(token) <= 0:
            raise _NotCanonicalError("invalid tensor dimension")
        dimensions.append(int(token))
        delimiter = tokens.take()
        if delimiter == "]":
            return tuple(dimensions)
        if delimiter != ",":
            raise _NotCanonicalError("invalid tensor shape")


def _tensor_declaration(
    tokens: _TokenStream,
    *,
    network_name: str,
    keyword: str,
) -> VNNLibTensor:
    kind = {
        "declare-input": VNNLibTensorKind.INPUT,
        "declare-output": VNNLibTensorKind.OUTPUT,
    }.get(keyword)
    if kind is None:
        raise _NotCanonicalError("unsupported network declaration")
    name = _symbol(tokens)
    try:
        dtype = _DTYPES[tokens.take()]
    except KeyError as error:
        raise _NotCanonicalError("unsupported tensor dtype") from error
    shape = _shape(tokens)
    tokens.expect(")")
    return VNNLibTensor(
        name=name,
        network_name=network_name,
        kind=kind,
        dtype=dtype,
        shape=shape,
    )


def _network(tokens: _TokenStream) -> VNNLibNetwork:
    tokens.expect("(")
    tokens.expect("declare-network")
    name = _symbol(tokens)
    tensors: list[VNNLibTensor] = []
    while tokens.peek() == "(":
        tokens.take()
        tensors.append(
            _tensor_declaration(
                tokens,
                network_name=name,
                keyword=tokens.take(),
            )
        )
    tokens.expect(")")
    inputs = tuple(tensor for tensor in tensors if tensor.kind is VNNLibTensorKind.INPUT)
    outputs = tuple(tensor for tensor in tensors if tensor.kind is VNNLibTensorKind.OUTPUT)
    if len(inputs) != 1 or len(outputs) != 1 or len(tensors) != 2:
        raise _NotCanonicalError("streaming grammar requires one input and one output")
    if inputs[0].name == outputs[0].name:
        raise _NotCanonicalError("tensor names must be unique")
    return VNNLibNetwork(name=name, inputs=inputs, hidden=(), outputs=outputs)


def _number(token: str) -> VNNLibConstant:
    if _NUMBER_RE.fullmatch(token) is None:
        raise _NotCanonicalError("unsupported numeric literal")
    value = float(token)
    if not math.isfinite(value):
        raise _NotCanonicalError("non-finite numeric literal")
    return VNNLibConstant(value=value, lexeme=token)


def _operand(
    tokens: _TokenStream,
    tensors: dict[str, VNNLibTensor],
) -> VNNLibExpression:
    token = tokens.take()
    if _NUMBER_RE.fullmatch(token) is not None:
        return _number(token)
    try:
        tensor = tensors[token]
    except KeyError as error:
        raise _NotCanonicalError("unknown or unsupported variable") from error
    tokens.expect("[")
    indices: list[int] = []
    if tokens.peek() != "]":
        while True:
            index = tokens.take()
            if not index.isdecimal():
                raise _NotCanonicalError("invalid tensor index")
            indices.append(int(index))
            delimiter = tokens.take()
            if delimiter == "]":
                break
            if delimiter != ",":
                raise _NotCanonicalError("invalid tensor index list")
    else:
        tokens.take()
    if len(indices) != len(tensor.shape) or any(
        index >= dimension for index, dimension in zip(indices, tensor.shape, strict=True)
    ):
        raise _NotCanonicalError("tensor index is outside its declaration")
    return VNNLibVariable(tensor=tensor, indices=tuple(indices))


def _comparison(
    tokens: _TokenStream,
    tensors: dict[str, VNNLibTensor],
    input_name: str,
) -> tuple[
    tuple[str, tuple[int, ...], VNNLibComparator, str] | None,
    VNNLibConstraint | None,
    bool,
]:
    tokens.expect("(")
    try:
        comparator = _COMPARATORS[tokens.take()]
    except KeyError as error:
        raise _NotCanonicalError("unsupported comparison") from error
    lhs = _operand(tokens, tensors)
    rhs = _operand(tokens, tensors)
    tokens.expect(")")
    if isinstance(lhs, VNNLibVariable) and isinstance(rhs, VNNLibConstant):
        variable, constant, normalized = lhs, rhs, comparator
    elif isinstance(rhs, VNNLibVariable) and isinstance(lhs, VNNLibConstant):
        variable, constant = rhs, lhs
        normalized = _REVERSED_COMPARATOR[comparator]
    else:
        raise _NotCanonicalError("comparison must relate one tensor scalar and one number")
    strict = comparator in {VNNLibComparator.LT, VNNLibComparator.GT}
    if variable.tensor.name == input_name:
        if normalized is VNNLibComparator.NE:
            raise _NotCanonicalError("not-equal input bounds require the official parser")
        return (
            (variable.tensor.name, variable.indices, normalized, constant.lexeme),
            None,
            strict,
        )
    return None, VNNLibConstraint(comparator=comparator, lhs=lhs, rhs=rhs), strict


def _query(tokens: _TokenStream) -> VNNLibQuery:
    tokens.expect("(")
    tokens.expect("vnnlib-version")
    tokens.expect("<")
    tokens.expect("2.0")
    tokens.expect(">")
    tokens.expect(")")
    network = _network(tokens)
    input_tensor = network.inputs[0]
    tensors = {input_tensor.name: input_tensor, network.outputs[0].name: network.outputs[0]}

    tokens.expect("(")
    tokens.expect("assert")
    tokens.expect("(")
    tokens.expect("or")
    branches: list[tuple[VNNLibConstraint, ...]] = []
    domains: list[VNNLibInputDomain] = []
    has_strict = False
    while tokens.peek() == "(":
        tokens.take()
        tokens.expect("and")
        bounds: list[tuple[str, tuple[int, ...], VNNLibComparator, str]] = []
        constraints: list[VNNLibConstraint] = []
        while tokens.peek() == "(":
            bound, constraint, strict = _comparison(tokens, tensors, input_tensor.name)
            has_strict |= strict
            if bound is not None:
                bounds.append(bound)
            if constraint is not None:
                constraints.append(constraint)
        tokens.expect(")")
        if not bounds or not constraints:
            raise _NotCanonicalError("each canonical branch requires input and output constraints")
        branches.append(tuple(constraints))
        domains.append(_pack_input_domain((input_tensor,), bounds))
    if not branches:
        raise _NotCanonicalError("canonical disjunction must contain at least one branch")
    tokens.expect(")")
    tokens.expect(")")
    if tokens.peek() is not None:
        raise _NotCanonicalError("canonical query has trailing statements")
    return VNNLibQuery(
        version=VNNLibVersion(2, 0),
        networks=(network,),
        branches=tuple(branches),
        capabilities=VNNLibCapabilities(
            has_hidden=False,
            has_multiple_networks=False,
            has_multiple_inputs=False,
            has_multiple_outputs=False,
            has_nonlinear=False,
            has_strict_comparisons=has_strict,
        ),
        global_input_domain=_pack_input_domain((input_tensor,), []),
        branch_input_domains=tuple(domains),
        parse_backend=VNNLibParseBackend.CANONICAL_STREAM,
    )


@contextmanager
def _open_text(source: Path) -> Iterator[TextIO]:
    """Open plain or gzip-compressed UTF-8 text with one ownership shape."""
    if source.suffix == ".gz":
        with gzip.open(source, mode="rt", encoding="utf-8") as stream:
            yield stream
    else:
        with source.open(mode="rt", encoding="utf-8") as stream:
            yield stream


def try_parse_canonical_query(path: str | Path) -> VNNLibQuery | None:
    """Return a streamed query, or ``None`` when the official parser must run."""
    try:
        with _open_text(Path(path)) as stream:
            return _query(_TokenStream(_tokens(stream)))
    except _NotCanonicalError:
        return None
