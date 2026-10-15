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
    "VNNLibMultiply",
    "VNNLibNegate",
    "VNNLibNetwork",
    "VNNLibQuery",
    "VNNLibSubtract",
    "VNNLibTensor",
    "VNNLibTensorKind",
    "VNNLibVariable",
    "VNNLibVersion",
]

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


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
