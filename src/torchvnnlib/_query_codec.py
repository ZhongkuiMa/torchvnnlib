"""Safe versioned serialization for owned VNN-LIB 2.0 queries."""

from __future__ import annotations

__docformat__ = "restructuredtext"
__all__ = ["VNNLibQueryCodecError", "dump_query", "load_query"]

import base64
import hashlib
import json
import os
import tempfile
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

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
    VNNLibParseBackend,
    VNNLibQuery,
    VNNLibSubtract,
    VNNLibTensor,
    VNNLibTensorKind,
    VNNLibVariable,
    VNNLibVersion,
)
from torchvnnlib._v2_stream import STREAMING_ABI

_SCHEMA = "torchvnnlib-query"
_SCHEMA_VERSION = 3


class VNNLibQueryCodecError(ValueError):
    """Raised when a serialized query is invalid or incompatible."""


def _encode_tensor(tensor: VNNLibTensor) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "network": tensor.network_name,
        "kind": tensor.kind.value,
        "dtype": tensor.dtype,
        "shape": list(tensor.shape),
        "onnx_name": tensor.onnx_name,
    }


def _decode_tensor(data: dict[str, Any]) -> VNNLibTensor:
    return VNNLibTensor(
        name=str(data["name"]),
        network_name=str(data["network"]),
        kind=VNNLibTensorKind(data["kind"]),
        dtype=str(data["dtype"]),
        shape=tuple(int(item) for item in data["shape"]),
        onnx_name=data["onnx_name"],
    )


def _encode_expression(expression: VNNLibExpression) -> list[Any]:
    if isinstance(expression, VNNLibConstant):
        return ["constant", expression.lexeme]
    if isinstance(expression, VNNLibVariable):
        return ["variable", expression.tensor.name, list(expression.indices)]
    if isinstance(expression, VNNLibNegate):
        return ["negate", _encode_expression(expression.expression)]
    if isinstance(expression, VNNLibAdd):
        return ["add", [_encode_expression(item) for item in expression.expressions]]
    if isinstance(expression, VNNLibSubtract):
        return [
            "subtract",
            _encode_expression(expression.head),
            [_encode_expression(item) for item in expression.rest],
        ]
    if isinstance(expression, VNNLibMultiply):
        return ["multiply", [_encode_expression(item) for item in expression.expressions]]
    raise TypeError(f"Unsupported owned expression: {type(expression).__name__}")


def _decode_number(lexeme: str) -> int | float:
    return float(lexeme) if any(char in lexeme.lower() for char in ".e") else int(lexeme)


def _decode_expression(data: list[Any], tensors: dict[str, VNNLibTensor]) -> VNNLibExpression:
    tag = data[0]
    if tag == "constant":
        lexeme = str(data[1])
        return VNNLibConstant(value=_decode_number(lexeme), lexeme=lexeme)
    if tag == "variable":
        return VNNLibVariable(
            tensor=tensors[str(data[1])], indices=tuple(int(item) for item in data[2])
        )
    if tag == "negate":
        return VNNLibNegate(_decode_expression(data[1], tensors))
    if tag == "add":
        return VNNLibAdd(tuple(_decode_expression(item, tensors) for item in data[1]))
    if tag == "subtract":
        return VNNLibSubtract(
            head=_decode_expression(data[1], tensors),
            rest=tuple(_decode_expression(item, tensors) for item in data[2]),
        )
    if tag == "multiply":
        return VNNLibMultiply(tuple(_decode_expression(item, tensors) for item in data[1]))
    raise VNNLibQueryCodecError(f"Unknown expression tag: {tag!r}")


def _encode_query(query: VNNLibQuery) -> dict[str, Any]:
    return {
        "language_version": [query.version.major, query.version.minor],
        "parse_backend": query.parse_backend.value,
        "networks": [
            {
                "name": network.name,
                "inputs": [_encode_tensor(item) for item in network.inputs],
                "hidden": [_encode_tensor(item) for item in network.hidden],
                "outputs": [_encode_tensor(item) for item in network.outputs],
                "equal_to": network.equal_to,
                "isomorphic_to": network.isomorphic_to,
            }
            for network in query.networks
        ],
        "branches": [
            [
                [
                    constraint.comparator.value,
                    _encode_expression(constraint.lhs),
                    _encode_expression(constraint.rhs),
                ]
                for constraint in branch
            ]
            for branch in query.branches
        ],
        "capabilities": {
            name: getattr(query.capabilities, name)
            for name in VNNLibCapabilities.__dataclass_fields__
        },
        "input_domain": {
            "tensor_names": list(query.global_input_domain.tensor_names),
            "tensor_shapes": [list(shape) for shape in query.global_input_domain.tensor_shapes],
            "records": base64.b64encode(query.global_input_domain.records).decode("ascii"),
            "lexemes": base64.b64encode(query.global_input_domain.lexemes).decode("ascii"),
        },
        "branch_input_domains": [
            {
                "records": base64.b64encode(domain.records).decode("ascii"),
                "lexemes": base64.b64encode(domain.lexemes).decode("ascii"),
            }
            for domain in query.branch_input_domains
        ],
    }


def _decode_query(data: dict[str, Any]) -> VNNLibQuery:
    networks = tuple(
        VNNLibNetwork(
            name=str(item["name"]),
            inputs=tuple(_decode_tensor(tensor) for tensor in item["inputs"]),
            hidden=tuple(_decode_tensor(tensor) for tensor in item["hidden"]),
            outputs=tuple(_decode_tensor(tensor) for tensor in item["outputs"]),
            equal_to=item["equal_to"],
            isomorphic_to=item["isomorphic_to"],
        )
        for item in data["networks"]
    )
    tensors = {
        tensor.name: tensor
        for network in networks
        for tensor in (*network.inputs, *network.hidden, *network.outputs)
    }
    branches = tuple(
        tuple(
            VNNLibConstraint(
                comparator=VNNLibComparator(item[0]),
                lhs=_decode_expression(item[1], tensors),
                rhs=_decode_expression(item[2], tensors),
            )
            for item in branch
        )
        for branch in data["branches"]
    )
    domain = data["input_domain"]
    input_tensors = tuple(tensor for network in networks for tensor in network.inputs)
    branch_domains = tuple(
        VNNLibInputDomain(
            tensor_names=tuple(tensor.name for tensor in input_tensors),
            tensor_shapes=tuple(tensor.shape for tensor in input_tensors),
            records=base64.b64decode(item["records"], validate=True),
            lexemes=base64.b64decode(item["lexemes"], validate=True),
        )
        for item in data["branch_input_domains"]
    )
    language_version = data["language_version"]
    return VNNLibQuery(
        version=VNNLibVersion(int(language_version[0]), int(language_version[1])),
        networks=networks,
        branches=branches,
        capabilities=VNNLibCapabilities(**data["capabilities"]),
        global_input_domain=VNNLibInputDomain(
            tensor_names=tuple(str(item) for item in domain["tensor_names"]),
            tensor_shapes=tuple(
                tuple(int(dimension) for dimension in shape) for shape in domain["tensor_shapes"]
            ),
            records=base64.b64decode(domain["records"], validate=True),
            lexemes=base64.b64decode(domain["lexemes"], validate=True),
        ),
        branch_input_domains=branch_domains,
        parse_backend=VNNLibParseBackend(data["parse_backend"]),
    )


def dump_query(query: VNNLibQuery, path: str | Path, *, source_sha256: str | None = None) -> None:
    """Atomically serialize a query without executable object formats."""
    encoded = _encode_query(query)
    payload = json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode()
    document = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "vnnlib_parser_version": distribution_version("vnnlib"),
        "streaming_abi": STREAMING_ABI,
        "source_sha256": source_sha256,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload": encoded,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_query(path: str | Path, *, expected_source_sha256: str | None = None) -> VNNLibQuery:
    """Load and validate a safe query document."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document["schema"] != _SCHEMA or document["schema_version"] != _SCHEMA_VERSION:
            raise VNNLibQueryCodecError("Unsupported TorchVNNLIB query schema")
        if document["vnnlib_parser_version"] != distribution_version("vnnlib"):
            raise VNNLibQueryCodecError("VNNLIB parser version does not match the cache")
        if document["streaming_abi"] != STREAMING_ABI:
            raise VNNLibQueryCodecError("VNNLIB streaming parser ABI does not match the cache")
        if (
            expected_source_sha256 is not None
            and document["source_sha256"] != expected_source_sha256
        ):
            raise VNNLibQueryCodecError("VNNLIB source digest does not match the cache")
        payload_data = document.pop("payload")
        payload = json.dumps(payload_data, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(payload).hexdigest() != document["payload_sha256"]:
            raise VNNLibQueryCodecError("VNNLIB query cache checksum mismatch")
        return _decode_query(payload_data)
    except VNNLibQueryCodecError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VNNLibQueryCodecError("Invalid TorchVNNLIB query cache") from error
