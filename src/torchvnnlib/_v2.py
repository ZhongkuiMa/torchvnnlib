"""VNN-LIB 2.0 parsing adapter backed by the official parser."""

__docformat__ = "restructuredtext"
__all__ = ["detect_version", "parse_query"]

import gzip
import itertools
import json
import re
from pathlib import Path
from typing import Any

import vnnlib

from torchvnnlib._query import (
    VNNLibAdd,
    VNNLibCapabilities,
    VNNLibComparator,
    VNNLibConstant,
    VNNLibConstraint,
    VNNLibExpression,
    VNNLibMultiply,
    VNNLibNegate,
    VNNLibNetwork,
    VNNLibQuery,
    VNNLibSubtract,
    VNNLibTensor,
    VNNLibTensorKind,
    VNNLibVariable,
    VNNLibVersion,
)

_VERSION_RE = re.compile(
    r"\(\s*vnnlib-version\s+<\s*(?P<major>\d+)\.(?P<minor>\d+)\s*>\s*\)",
    re.IGNORECASE,
)
_VERSION_TOKEN_RE = re.compile(r"\(\s*vnnlib-version\b", re.IGNORECASE)


def _read_text(path: str | Path) -> str:
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, mode="rt", encoding="utf-8") as stream:
            return stream.read()
    return source.read_text(encoding="utf-8")


def _source_text(path: str | Path, lines: list[str] | tuple[str, ...] | None) -> str:
    if lines is not None:
        separator = "" if any(line.endswith(("\n", "\r")) for line in lines) else "\n"
        return separator.join(lines)
    if not path:
        raise ValueError("A VNN-LIB path or in-memory lines are required")
    return _read_text(path)


def detect_version(
    path: str | Path = "", *, lines: list[str] | tuple[str, ...] | None = None
) -> VNNLibVersion:
    """Detect the declared language version, defaulting headerless files to 1.0."""
    text = "\n".join(
        line.split(";", maxsplit=1)[0] for line in _source_text(path, lines).splitlines()
    ).lstrip()
    token = _VERSION_TOKEN_RE.search(text)
    if token is None:
        return VNNLibVersion(1, 0)
    if token.start() != 0:
        raise ValueError("VNN-LIB version declaration must be the first statement")
    match = _VERSION_RE.match(text)
    if match is None:
        raise ValueError("Malformed VNN-LIB version declaration")
    return VNNLibVersion(int(match.group("major")), int(match.group("minor")))


def _tensor_kind(kind: Any) -> VNNLibTensorKind:
    name = kind.name.lower()
    try:
        return VNNLibTensorKind(name)
    except ValueError as error:
        raise ValueError(f"Unsupported VNN-LIB tensor kind: {kind}") from error


def _convert_tensor(definition: Any) -> VNNLibTensor:
    onnx_name = definition.onnx_name
    if isinstance(onnx_name, str) and onnx_name.startswith('"'):
        onnx_name = json.loads(onnx_name)
    return VNNLibTensor(
        name=definition.name,
        network_name=definition.network_name,
        kind=_tensor_kind(definition.kind),
        dtype=definition.dtype.name,
        shape=tuple(definition.shape),
        onnx_name=onnx_name,
    )


def _optional_relation(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _convert_network(network: Any) -> VNNLibNetwork:
    return VNNLibNetwork(
        name=network.name,
        inputs=tuple(_convert_tensor(item) for item in network.inputs),
        hidden=tuple(_convert_tensor(item) for item in network.hidden),
        outputs=tuple(_convert_tensor(item) for item in network.outputs),
        equal_to=_optional_relation(network.equal_to),
        # The official 1.0.2 Python binding exposes the standard's
        # ``isomorphic-to`` field under the historical name ``isometric_to``.
        isomorphic_to=_optional_relation(network.isometric_to),
    )


def _convert_expression(expression: Any, tensors: dict[str, VNNLibTensor]) -> VNNLibExpression:
    if isinstance(expression, vnnlib.Var):
        try:
            tensor = tensors[expression.name]
        except KeyError as error:
            raise ValueError(f"Unknown VNN-LIB variable: {expression.name}") from error
        return VNNLibVariable(tensor=tensor, indices=tuple(expression.indices))
    if isinstance(expression, vnnlib.Literal):
        lexeme = expression.lexeme
        value: int | float = (
            float(lexeme) if any(c in lexeme.lower() for c in ".e") else int(lexeme)
        )
        return VNNLibConstant(value=value, lexeme=lexeme)
    if isinstance(expression, vnnlib.Float):
        return VNNLibConstant(value=expression.value, lexeme=repr(expression.value))
    if isinstance(expression, vnnlib.Int):
        return VNNLibConstant(value=expression.value, lexeme=str(expression.value))
    if isinstance(expression, vnnlib.Negate):
        return VNNLibNegate(_convert_expression(expression.expr, tensors))
    if isinstance(expression, vnnlib.Plus):
        return VNNLibAdd(tuple(_convert_expression(item, tensors) for item in expression.args))
    if isinstance(expression, vnnlib.Minus):
        return VNNLibSubtract(
            head=_convert_expression(expression.head, tensors),
            rest=tuple(_convert_expression(item, tensors) for item in expression.rest),
        )
    if isinstance(expression, vnnlib.Multiply):
        return VNNLibMultiply(tuple(_convert_expression(item, tensors) for item in expression.args))
    raise TypeError(f"Unsupported VNN-LIB arithmetic expression: {type(expression).__name__}")


_COMPARATORS: tuple[tuple[type[Any], VNNLibComparator], ...] = (
    (vnnlib.LessThan, VNNLibComparator.LT),
    (vnnlib.LessEqual, VNNLibComparator.LE),
    (vnnlib.Equal, VNNLibComparator.EQ),
    (vnnlib.NotEqual, VNNLibComparator.NE),
    (vnnlib.GreaterEqual, VNNLibComparator.GE),
    (vnnlib.GreaterThan, VNNLibComparator.GT),
)


def _convert_comparison(comparison: Any, tensors: dict[str, VNNLibTensor]) -> VNNLibConstraint:
    for comparison_type, comparator in _COMPARATORS:
        if isinstance(comparison, comparison_type):
            return VNNLibConstraint(
                comparator=comparator,
                lhs=_convert_expression(comparison.lhs, tensors),
                rhs=_convert_expression(comparison.rhs, tensors),
            )
    raise TypeError(f"Unsupported VNN-LIB comparison: {type(comparison).__name__}")


def _contains_nonlinear(expression: VNNLibExpression) -> bool:
    if isinstance(expression, VNNLibMultiply):
        nonconstant = sum(not isinstance(item, VNNLibConstant) for item in expression.expressions)
        return nonconstant > 1 or any(_contains_nonlinear(item) for item in expression.expressions)
    if isinstance(expression, VNNLibNegate):
        return _contains_nonlinear(expression.expression)
    if isinstance(expression, VNNLibAdd):
        return any(_contains_nonlinear(item) for item in expression.expressions)
    if isinstance(expression, VNNLibSubtract):
        return _contains_nonlinear(expression.head) or any(
            _contains_nonlinear(item) for item in expression.rest
        )
    return False


def parse_query(
    path: str | Path = "", *, lines: list[str] | tuple[str, ...] | None = None
) -> VNNLibQuery:
    """Parse a VNN-LIB 2.0 query into TorchVNNLIB's stable owned IR."""
    text = _source_text(path, lines)
    version = detect_version(lines=[text])
    if version == VNNLibVersion(1, 0):
        raise ValueError(
            f"parse_query() requires VNN-LIB 2.0, got {version}; use parse() for VNN-LIB 1.0"
        )
    if version != VNNLibVersion(2, 0):
        raise ValueError(f"parse_query() supports VNN-LIB 2.0, got {version}")

    parsed = vnnlib.parse_query_string(text)
    networks = tuple(_convert_network(network) for network in parsed.networks)
    tensors = {
        tensor.name: tensor
        for network in networks
        for tensor in (*network.inputs, *network.hidden, *network.outputs)
    }

    # Assertions form a top-level conjunction.  The upstream parser provides
    # DNF per assertion, so compute the conjunction by a Cartesian product.
    assertion_dnfs = [assertion.expr.to_dnf() for assertion in parsed.assertions]
    products = itertools.product(*assertion_dnfs) if assertion_dnfs else [()]
    branches = tuple(
        tuple(_convert_comparison(item, tensors) for clause in product for item in clause)
        for product in products
    )
    constraints = tuple(constraint for branch in branches for constraint in branch)
    capabilities = VNNLibCapabilities(
        has_hidden=any(network.hidden for network in networks),
        has_multiple_networks=len(networks) > 1,
        has_multiple_inputs=any(len(network.inputs) > 1 for network in networks),
        has_multiple_outputs=any(len(network.outputs) > 1 for network in networks),
        has_nonlinear=any(
            _contains_nonlinear(constraint.lhs) or _contains_nonlinear(constraint.rhs)
            for constraint in constraints
        ),
        has_strict_comparisons=any(
            constraint.comparator in {VNNLibComparator.LT, VNNLibComparator.GT}
            for constraint in constraints
        ),
    )
    return VNNLibQuery(
        version=version,
        networks=networks,
        branches=branches,
        capabilities=capabilities,
    )
