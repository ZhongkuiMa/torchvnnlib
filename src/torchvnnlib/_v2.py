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
    VNNLibInputDomain,
    VNNLibMultiply,
    VNNLibNegate,
    VNNLibNetwork,
    VNNLibQuery,
    VNNLibSubtract,
    VNNLibTensor,
    VNNLibTensorKind,
    VNNLibVariable,
    VNNLibVersion,
    _pack_input_domain,
)
from torchvnnlib._v2_stream import try_parse_canonical_query

_VERSION_RE = re.compile(
    r"\(\s*vnnlib-version\s+<\s*(?P<major>\d+)\.(?P<minor>\d+)\s*>\s*\)",
    re.IGNORECASE,
)
_VERSION_TOKEN_RE = re.compile(r"\(\s*vnnlib-version\b", re.IGNORECASE)
_VERSION_PREFIX_LIMIT = 64 * 1024


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


def _source_prefix(path: str | Path) -> str:
    """Read a bounded prefix for the mandatory first-statement version check."""
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, mode="rt", encoding="utf-8") as stream:
            return stream.read(_VERSION_PREFIX_LIMIT)
    with source.open(encoding="utf-8") as stream:
        return stream.read(_VERSION_PREFIX_LIMIT)


def detect_version(
    path: str | Path = "", *, lines: list[str] | tuple[str, ...] | None = None
) -> VNNLibVersion:
    """Detect the declared language version, defaulting headerless files to 1.0."""
    source = _source_text(path, lines) if lines is not None else _source_prefix(path)
    text = "\n".join(line.split(";", maxsplit=1)[0] for line in source.splitlines()).lstrip()
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
        lexeme = _numeric_lexeme(expression)
        value: int | float = (
            float(lexeme) if any(c in lexeme.lower() for c in ".e") else int(lexeme)
        )
        return VNNLibConstant(value=value, lexeme=lexeme)
    if isinstance(expression, vnnlib.Float):
        return VNNLibConstant(value=expression.value, lexeme=_numeric_lexeme(expression))
    if isinstance(expression, vnnlib.Int):
        return VNNLibConstant(value=expression.value, lexeme=_numeric_lexeme(expression))
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


def _numeric_lexeme(expression: Any) -> str:
    """Return the official parser's exact numeric source spelling."""
    lexeme = getattr(expression, "lexeme", None)
    if not isinstance(lexeme, str) or not lexeme:
        raise RuntimeError(
            "Official vnnlib numeric nodes must expose a non-empty 'lexeme'; "
            "the installed parser API is incompatible with TorchVNNLIB"
        )
    return lexeme


_COMPARATORS: tuple[tuple[type[Any], VNNLibComparator], ...] = (
    (vnnlib.LessThan, VNNLibComparator.LT),
    (vnnlib.LessEqual, VNNLibComparator.LE),
    (vnnlib.Equal, VNNLibComparator.EQ),
    (vnnlib.NotEqual, VNNLibComparator.NE),
    (vnnlib.GreaterEqual, VNNLibComparator.GE),
    (vnnlib.GreaterThan, VNNLibComparator.GT),
)

_REVERSED_COMPARATOR = {
    VNNLibComparator.LT: VNNLibComparator.GT,
    VNNLibComparator.LE: VNNLibComparator.GE,
    VNNLibComparator.EQ: VNNLibComparator.EQ,
    VNNLibComparator.GE: VNNLibComparator.LE,
    VNNLibComparator.GT: VNNLibComparator.LT,
}


def _convert_comparison(comparison: Any, tensors: dict[str, VNNLibTensor]) -> VNNLibConstraint:
    for comparison_type, comparator in _COMPARATORS:
        if isinstance(comparison, comparison_type):
            return VNNLibConstraint(
                comparator=comparator,
                lhs=_convert_expression(comparison.lhs, tensors),
                rhs=_convert_expression(comparison.rhs, tensors),
            )
    raise TypeError(f"Unsupported VNN-LIB comparison: {type(comparison).__name__}")


def _comparison_operator(comparison: Any) -> VNNLibComparator | None:
    """Return the owned comparator for an official comparison node."""
    for comparison_type, comparator in _COMPARATORS:
        if isinstance(comparison, comparison_type):
            return comparator
    return None


def _simple_input_bound(
    comparison: Any, input_names: set[str]
) -> tuple[str, tuple[int, ...], VNNLibComparator, str] | None:
    """Normalize one scalar input-to-constant comparison."""
    comparator = _comparison_operator(comparison)
    if comparator is None or comparator is VNNLibComparator.NE:
        return None
    lhs = comparison.lhs
    rhs = comparison.rhs
    numeric_types = (vnnlib.Literal, vnnlib.Float, vnnlib.Int)
    if isinstance(lhs, vnnlib.Var) and isinstance(rhs, numeric_types):
        variable, numeric = lhs, rhs
    elif isinstance(rhs, vnnlib.Var) and isinstance(lhs, numeric_types):
        variable, numeric = rhs, lhs
        comparator = _REVERSED_COMPARATOR[comparator]
    else:
        return None
    if variable.name not in input_names:
        return None
    return variable.name, tuple(variable.indices), comparator, _numeric_lexeme(numeric)


def _pure_input_bounds(
    expression: Any, input_names: set[str]
) -> list[tuple[str, tuple[int, ...], VNNLibComparator, str]] | None:
    """Extract a complete top-level conjunction of simple input bounds."""
    if isinstance(expression, vnnlib.And):
        bounds: list[tuple[str, tuple[int, ...], VNNLibComparator, str]] = []
        for child in expression.args:
            child_bounds = _pure_input_bounds(child, input_names)
            if child_bounds is None:
                return None
            bounds.extend(child_bounds)
        return bounds
    bound = _simple_input_bound(expression, input_names)
    return None if bound is None else [bound]


def _convert_residual_branches(
    assertion_dnfs: list[Any],
    tensors: dict[str, VNNLibTensor],
    inputs: tuple[VNNLibTensor, ...],
    input_names: set[str],
) -> tuple[tuple[tuple[VNNLibConstraint, ...], ...], tuple[VNNLibInputDomain, ...], bool]:
    """Convert residual DNF while packing branch-local scalar input bounds."""
    products = itertools.product(*assertion_dnfs) if assertion_dnfs else [()]
    branches: list[tuple[VNNLibConstraint, ...]] = []
    domains: list[VNNLibInputDomain] = []
    has_strict_bound = False
    for product in products:
        constraints: list[VNNLibConstraint] = []
        bounds: list[tuple[str, tuple[int, ...], VNNLibComparator, str]] = []
        for clause in product:
            for item in clause:
                bound = _simple_input_bound(item, input_names)
                if bound is None:
                    constraints.append(_convert_comparison(item, tensors))
                    continue
                bounds.append(bound)
                has_strict_bound |= bound[2] in {
                    VNNLibComparator.LT,
                    VNNLibComparator.GT,
                }
        branches.append(tuple(constraints))
        domains.append(_pack_input_domain(inputs, bounds))
    return tuple(branches), tuple(domains), has_strict_bound


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
    text = _source_text(path, lines) if lines is not None else None
    version = detect_version(path, lines=[text] if text is not None else None)
    if version == VNNLibVersion(1, 0):
        raise ValueError(
            f"parse_query() requires VNN-LIB 2.0, got {version}; use parse() for VNN-LIB 1.0"
        )
    if version != VNNLibVersion(2, 0):
        raise ValueError(f"parse_query() supports VNN-LIB 2.0, got {version}")

    if text is None:
        streamed = try_parse_canonical_query(path)
        if streamed is not None:
            return streamed
        if Path(path).suffix == ".gz":
            text = _read_text(path)

    parsed = (
        vnnlib.parse_query_string(text)
        if text is not None
        else vnnlib.parse_query_file(str(Path(path)))
    )
    networks = tuple(_convert_network(network) for network in parsed.networks)
    tensors = {
        tensor.name: tensor
        for network in networks
        for tensor in (*network.inputs, *network.hidden, *network.outputs)
    }

    inputs = tuple(tensor for network in networks for tensor in network.inputs)
    input_names = {tensor.name for tensor in inputs}
    global_bounds: list[tuple[str, tuple[int, ...], VNNLibComparator, str]] = []
    residual_assertions = []
    for assertion in parsed.assertions:
        bounds = _pure_input_bounds(assertion.expr, input_names)
        if bounds is None:
            residual_assertions.append(assertion)
        else:
            global_bounds.extend(bounds)

    # Assertions form a top-level conjunction. The upstream parser provides
    # DNF per assertion, so compute only the residual conjunction here. Simple
    # input bounds inside mixed branches remain branch-local but use the same
    # packed representation as the global domain.
    assertion_dnfs = [assertion.expr.to_dnf() for assertion in residual_assertions]
    branches, branch_input_domains, has_strict_branch_bound = _convert_residual_branches(
        assertion_dnfs,
        tensors,
        inputs,
        input_names,
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
        )
        or any(
            comparator in {VNNLibComparator.LT, VNNLibComparator.GT}
            for _, _, comparator, _ in global_bounds
        )
        or has_strict_branch_bound,
    )
    return VNNLibQuery(
        version=version,
        networks=networks,
        global_input_domain=_pack_input_domain(inputs, global_bounds),
        branch_input_domains=branch_input_domains,
        branches=branches,
        capabilities=capabilities,
    )
