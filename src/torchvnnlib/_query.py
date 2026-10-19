"""Owned, immutable representation of a VNN-LIB 2.0 query."""

from __future__ import annotations

__docformat__ = "restructuredtext"
__all__ = [
    "VNNLibAdd",
    "VNNLibBranch",
    "VNNLibCapabilities",
    "VNNLibComparator",
    "VNNLibConstant",
    "VNNLibConstraint",
    "VNNLibExpression",
    "VNNLibInputBound",
    "VNNLibInputDomain",
    "VNNLibMultiply",
    "VNNLibNegate",
    "VNNLibNetwork",
    "VNNLibParseBackend",
    "VNNLibQuery",
    "VNNLibSubtract",
    "VNNLibTensor",
    "VNNLibTensorKind",
    "VNNLibVariable",
    "VNNLibVersion",
]

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from math import prod
from struct import Struct
from typing import TypeAlias

_BOUND_RECORD = Struct("<IIIIB")


@dataclass(frozen=True, slots=True, order=True)
class VNNLibVersion:
    """A parsed VNN-LIB language version."""

    major: int
    minor: int

    def __str__(self) -> str:
        """Return the canonical dotted representation."""
        return f"{self.major}.{self.minor}"


class VNNLibTensorKind(StrEnum):
    """Role of a declared tensor in a network interface."""

    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class VNNLibParseBackend(StrEnum):
    """Parser implementation that produced an owned query."""

    OFFICIAL_AST = "official-ast"
    CANONICAL_STREAM = "canonical-stream"


@dataclass(frozen=True, slots=True)
class VNNLibTensor:
    """One tensor declared by a VNN-LIB network interface."""

    name: str
    network_name: str
    kind: VNNLibTensorKind
    dtype: str
    shape: tuple[int, ...]
    onnx_name: str | None = None


@dataclass(frozen=True, slots=True)
class VNNLibNetwork:
    """A network declaration and its ordered tensor interface."""

    name: str
    inputs: tuple[VNNLibTensor, ...]
    hidden: tuple[VNNLibTensor, ...]
    outputs: tuple[VNNLibTensor, ...]
    equal_to: str | None = None
    isomorphic_to: str | None = None


@dataclass(frozen=True, slots=True)
class VNNLibConstant:
    """A numeric literal, retaining both its value and source spelling."""

    value: int | float
    lexeme: str


@dataclass(frozen=True, slots=True)
class VNNLibVariable:
    """A scalar tensor element reference."""

    tensor: VNNLibTensor
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VNNLibNegate:
    """Arithmetic negation."""

    expression: VNNLibExpression


@dataclass(frozen=True, slots=True)
class VNNLibAdd:
    """N-ary addition."""

    expressions: tuple[VNNLibExpression, ...]


@dataclass(frozen=True, slots=True)
class VNNLibSubtract:
    """N-ary subtraction with one head and zero or more trailing operands."""

    head: VNNLibExpression
    rest: tuple[VNNLibExpression, ...]


@dataclass(frozen=True, slots=True)
class VNNLibMultiply:
    """N-ary multiplication."""

    expressions: tuple[VNNLibExpression, ...]


VNNLibExpression: TypeAlias = (
    VNNLibConstant | VNNLibVariable | VNNLibNegate | VNNLibAdd | VNNLibSubtract | VNNLibMultiply
)


class VNNLibComparator(StrEnum):
    """Comparison operators supported by VNN-LIB 2.0."""

    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="
    GE = ">="
    GT = ">"


@dataclass(frozen=True, slots=True)
class VNNLibConstraint:
    """One atomic comparison in a DNF branch."""

    comparator: VNNLibComparator
    lhs: VNNLibExpression
    rhs: VNNLibExpression


VNNLibBranch: TypeAlias = tuple[VNNLibConstraint, ...]


@dataclass(frozen=True, slots=True)
class VNNLibInputBound:
    """One decoded global input bound.

    :param tensor_name: Declared input tensor name.
    :param indices: Tensor element indices in declaration order.
    :param comparator: Comparison normalized with the variable on the left.
    :param lexeme: Exact source spelling of the numeric bound.
    """

    tensor_name: str
    indices: tuple[int, ...]
    comparator: VNNLibComparator
    lexeme: str


@dataclass(frozen=True, slots=True)
class VNNLibInputDomain:
    """Packed conjunction of top-level scalar input bounds.

    Fixed-size records and one UTF-8 string table avoid allocating an expression
    tree for every scalar bound in large image properties.

    :param tensor_names: Input tensor names referenced by packed records.
    :param tensor_shapes: Shapes parallel to ``tensor_names``.
    :param records: Packed bound records.
    :param lexemes: Concatenated UTF-8 source lexemes.
    """

    tensor_names: tuple[str, ...] = ()
    tensor_shapes: tuple[tuple[int, ...], ...] = ()
    records: bytes = b""
    lexemes: bytes = b""

    def __post_init__(self) -> None:
        """Validate the packed representation."""
        if len(self.tensor_names) != len(self.tensor_shapes):
            raise ValueError("Input-domain tensor names and shapes must have equal length")
        if len(self.records) % _BOUND_RECORD.size:
            raise ValueError("Input-domain records are truncated")

    def __len__(self) -> int:
        """Return the number of packed bounds."""
        return len(self.records) // _BOUND_RECORD.size

    def bound_at(self, index: int) -> VNNLibInputBound:
        """Decode one packed bound by index."""
        count = len(self)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError(index)
        tensor_id, flat_index, lexeme_start, lexeme_length, comparator_code = (
            _BOUND_RECORD.unpack_from(self.records, index * _BOUND_RECORD.size)
        )
        try:
            name = self.tensor_names[tensor_id]
            shape = self.tensor_shapes[tensor_id]
            comparator = tuple(VNNLibComparator)[comparator_code]
        except IndexError as error:
            raise ValueError("Input-domain record references invalid metadata") from error
        lexeme_end = lexeme_start + lexeme_length
        if lexeme_end > len(self.lexemes):
            raise ValueError("Input-domain record references an invalid lexeme")
        indices = _unravel_index(flat_index, shape)
        return VNNLibInputBound(
            tensor_name=name,
            indices=indices,
            comparator=comparator,
            lexeme=self.lexemes[lexeme_start:lexeme_end].decode("utf-8"),
        )

    def __iter__(self) -> Iterator[VNNLibInputBound]:
        """Yield decoded bounds without retaining per-bound objects."""
        for index in range(len(self)):
            yield self.bound_at(index)


def _unravel_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Convert a row-major flat index to tensor indices."""
    size = prod(shape)
    if flat_index >= size:
        raise ValueError("Input-domain record references an invalid tensor index")
    indices: list[int] = []
    remaining = flat_index
    for dimension in reversed(shape):
        indices.append(remaining % dimension)
        remaining //= dimension
    return tuple(reversed(indices))


def _pack_input_domain(
    inputs: tuple[VNNLibTensor, ...],
    bounds: list[tuple[str, tuple[int, ...], VNNLibComparator, str]],
) -> VNNLibInputDomain:
    """Pack normalized input bounds into the owned binary representation."""
    tensors = {tensor.name: (index, tensor) for index, tensor in enumerate(inputs)}
    records = bytearray()
    lexemes = bytearray()
    lexeme_locations: dict[bytes, tuple[int, int]] = {}
    comparator_codes = {comparator: code for code, comparator in enumerate(VNNLibComparator)}
    for name, indices, comparator, lexeme in bounds:
        try:
            tensor_id, tensor = tensors[name]
        except KeyError as error:
            raise ValueError(f"Unknown input tensor: {name}") from error
        if len(indices) != len(tensor.shape):
            raise ValueError("Input bound index rank does not match its tensor declaration")
        flat_index = 0
        for index, dimension in zip(indices, tensor.shape, strict=True):
            if not 0 <= index < dimension:
                raise ValueError("Input bound index is outside its tensor declaration")
            flat_index = flat_index * dimension + index
        encoded = lexeme.encode("utf-8")
        location = lexeme_locations.get(encoded)
        if location is None:
            location = (len(lexemes), len(encoded))
            lexemes.extend(encoded)
            lexeme_locations[encoded] = location
        start, length = location
        records.extend(
            _BOUND_RECORD.pack(
                tensor_id,
                flat_index,
                start,
                length,
                comparator_codes[comparator],
            )
        )
    return VNNLibInputDomain(
        tensor_names=tuple(tensor.name for tensor in inputs),
        tensor_shapes=tuple(tensor.shape for tensor in inputs),
        records=bytes(records),
        lexemes=bytes(lexemes),
    )


@dataclass(frozen=True, slots=True)
class VNNLibCapabilities:
    """Structural features present in a query."""

    has_hidden: bool
    has_multiple_networks: bool
    has_multiple_inputs: bool
    has_multiple_outputs: bool
    has_nonlinear: bool
    has_strict_comparisons: bool


@dataclass(frozen=True, slots=True)
class VNNLibQuery:
    """A VNN-LIB 2.0 query independent from the upstream parser AST."""

    version: VNNLibVersion
    networks: tuple[VNNLibNetwork, ...]
    branches: tuple[VNNLibBranch, ...]
    capabilities: VNNLibCapabilities
    global_input_domain: VNNLibInputDomain = VNNLibInputDomain()
    branch_input_domains: tuple[VNNLibInputDomain, ...] = ()
    parse_backend: VNNLibParseBackend = VNNLibParseBackend.OFFICIAL_AST

    def __post_init__(self) -> None:
        """Reject a partial branch-domain table."""
        if self.branch_input_domains and len(self.branch_input_domains) != len(self.branches):
            raise ValueError("Branch input domains must align one-to-one with DNF branches")
