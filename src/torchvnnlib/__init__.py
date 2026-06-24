"""TorchVNNLIB: Convert VNN-LIB files to PyTorch tensors or NumPy arrays.

Provides conversion from VNN-LIB specification format to PyTorch tensor
or NumPy array representations for neural network verification.

.. seealso:: ``torchvnnlib/CONVENTIONS.md`` for code conventions.
"""

__docformat__ = "restructuredtext"
__version__ = "2026.6.1"
__all__ = ["TorchVNNLIB", "__version__"]

import logging

from torchvnnlib._torchvnnlib import TorchVNNLIB

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())
