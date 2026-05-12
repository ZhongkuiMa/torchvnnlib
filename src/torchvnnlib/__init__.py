"""TorchVNNLIB: Convert VNN-LIB files to PyTorch tensors or NumPy arrays.

Provides conversion from VNN-LIB specification format to PyTorch tensor
or NumPy array representations for neural network verification.
"""

__docformat__ = "restructuredtext"
__version__ = "2026.5.1"
__all__ = ["TorchVNNLIB", "__version__"]

import logging

from torchvnnlib._torchvnnlib import TorchVNNLIB

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def _ensure_verbose_handler() -> None:
    """Configure package-level logger to emit to stderr at DEBUG level."""
    pkg_logger = logging.getLogger("torchvnnlib")
    if not any(isinstance(h, logging.StreamHandler) for h in pkg_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.DEBUG)
