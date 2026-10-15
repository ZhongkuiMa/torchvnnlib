"""TorchVNNLIB: Convert VNN-LIB files to PyTorch tensors or NumPy arrays.

Provides conversion from VNN-LIB specification format to PyTorch tensor
or NumPy array representations for neural network verification.

.. seealso:: ``torchvnnlib/CONVENTIONS.md`` for code conventions.
"""

__docformat__ = "restructuredtext"
__version__ = "2026.10.1"
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
    "VNNLibMultiply",
    "VNNLibNegate",
    "VNNLibNetwork",
    "VNNLibQuery",
    "VNNLibSubtract",
    "VNNLibTensor",
    "VNNLibTensorKind",
    "VNNLibVariable",
    "VNNLibVersion",
    "__version__",
]

import logging

from torchvnnlib._query import (
    VNNLibAdd,
    VNNLibBranch,
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
from torchvnnlib._torchvnnlib import TensorProperties, TensorProperty, TorchVNNLIB

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())
