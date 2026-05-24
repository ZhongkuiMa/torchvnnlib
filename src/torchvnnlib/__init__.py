"""TorchVNNLIB: Convert VNN-LIB files to PyTorch tensors or NumPy arrays.

Provides conversion from VNN-LIB specification format to PyTorch tensor
or NumPy array representations for neural network verification.
"""

__docformat__ = "restructuredtext"
__version__ = "2026.5.2"
__all__ = ["TorchVNNLIB", "__version__"]

import logging

from torchvnnlib._logging import _enable_verbose
from torchvnnlib._torchvnnlib import TorchVNNLIB

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

# Backward-compatible alias; prefer _enable_verbose.
_ensure_verbose_handler = _enable_verbose
