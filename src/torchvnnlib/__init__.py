"""TorchVNNLIB: Convert VNN-LIB files to PyTorch tensors or NumPy arrays.

Provides conversion from VNN-LIB specification format to PyTorch tensor
or NumPy array representations for neural network verification.

.. seealso:: ``torchvnnlib/CONVENTIONS.md`` for code conventions.
"""

__docformat__ = "restructuredtext"
__version__ = "2026.10.2"
__all__ = [
    "TensorProperties",
    "TensorProperty",
    "TorchVNNLIB",
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
    "VNNLibQueryCodecError",
    "VNNLibSubtract",
    "VNNLibTensor",
    "VNNLibTensorKind",
    "VNNLibVariable",
    "VNNLibVersion",
    "__version__",
    "dump_query",
    "load_query",
]

import logging
from typing import TYPE_CHECKING, Any

from torchvnnlib._query import (
    VNNLibAdd,
    VNNLibBranch,
    VNNLibCapabilities,
    VNNLibComparator,
    VNNLibConstant,
    VNNLibConstraint,
    VNNLibExpression,
    VNNLibInputBound,
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
from torchvnnlib._query_codec import VNNLibQueryCodecError, dump_query, load_query

if TYPE_CHECKING:
    from torchvnnlib._torchvnnlib import TensorProperties, TensorProperty, TorchVNNLIB

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def __getattr__(name: str) -> Any:
    """Load the legacy tensor converter without burdening query-only workers."""
    if name not in {"TensorProperties", "TensorProperty", "TorchVNNLIB"}:
        raise AttributeError(name)
    from torchvnnlib import _torchvnnlib

    value = getattr(_torchvnnlib, name)
    globals()[name] = value
    return value
