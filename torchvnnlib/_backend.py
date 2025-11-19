"""Backend abstraction for torch and numpy.

This module provides a unified interface for tensor operations
that works with both PyTorch tensors and NumPy arrays.
"""

__docformat__ = "restructuredtext"
__all__ = ["Backend", "TorchBackend", "NumpyBackend", "get_backend", "TensorLike"]

from abc import ABC, abstractmethod
from typing import Any

try:
    import torch

    TORCH_AVAILABLE = True
    TensorLike = torch.Tensor | Any  # Any for numpy when torch unavailable
except ImportError:
    TORCH_AVAILABLE = False
    TensorLike = Any

import numpy as np

if TORCH_AVAILABLE:
    TensorLike = torch.Tensor | np.ndarray[Any, Any]
else:
    TensorLike = np.ndarray[Any, Any]


class Backend(ABC):
    """Abstract backend interface for tensor operations."""

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: str = "float64") -> TensorLike:
        """Create array of zeros.

        :param shape: Shape of the array
        :param dtype: Data type
        :return: Zero-filled array
        """
        pass

    @abstractmethod
    def full(
        self, shape: tuple[int, ...], fill_value: float, dtype: str = "float64"
    ) -> TensorLike:
        """Create array filled with value.

        :param shape: Shape of the array
        :param fill_value: Fill value
        :param dtype: Data type
        :return: Filled array
        """
        pass

    @abstractmethod
    def tensor(self, data: list | Any, dtype: str = "float64") -> TensorLike:
        """Create array from data.

        :param data: Input data
        :param dtype: Data type
        :return: Array
        """
        pass

    @abstractmethod
    def stack(self, arrays: list[TensorLike], axis: int = 0) -> TensorLike:
        """Stack arrays along axis.

        :param arrays: List of arrays
        :param axis: Axis to stack along
        :return: Stacked array
        """
        pass

    @abstractmethod
    def isnan(self, arr: TensorLike) -> TensorLike:
        """Check for NaN values.

        :param arr: Input array
        :return: Boolean array
        """
        pass

    @abstractmethod
    def where(self, condition: TensorLike) -> tuple[TensorLike, ...]:
        """Find where condition is True.

        :param condition: Boolean array
        :return: Tuple of indices
        """
        pass

    @abstractmethod
    def save(self, data: dict[str, Any], file_path: str) -> None:
        """Save data to file.

        :param data: Data dictionary
        :param file_path: Output file path
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name.

        :return: Name of the backend
        """
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for saved files.

        :return: File extension
        """
        pass


class TorchBackend(Backend):
    """PyTorch backend implementation."""

    def zeros(self, shape: tuple[int, ...], dtype: str = "float64") -> TensorLike:
        dtype_map = {
            "float64": torch.float64,
            "float32": torch.float32,
            "int64": torch.long,
        }
        return torch.zeros(shape, dtype=dtype_map.get(dtype, torch.float64))

    def full(
        self, shape: tuple[int, ...], fill_value: float, dtype: str = "float64"
    ) -> TensorLike:
        dtype_map = {
            "float64": torch.float64,
            "float32": torch.float32,
            "int64": torch.long,
        }
        return torch.full(shape, fill_value, dtype=dtype_map.get(dtype, torch.float64))

    def tensor(self, data: list | Any, dtype: str = "float64") -> TensorLike:
        dtype_map = {
            "float64": torch.float64,
            "float32": torch.float32,
            "int64": torch.long,
        }
        return torch.tensor(data, dtype=dtype_map.get(dtype, torch.float64))

    def stack(self, arrays: list[TensorLike], axis: int = 0) -> TensorLike:
        return torch.stack(arrays, dim=axis)

    def isnan(self, arr: TensorLike) -> TensorLike:
        return torch.isnan(arr)

    def where(self, condition: TensorLike) -> tuple[TensorLike, ...]:
        return torch.where(condition)

    def save(self, data: dict[str, Any], file_path: str) -> None:
        torch.save(data, file_path)

    @property
    def name(self) -> str:
        return "torch"

    @property
    def file_extension(self) -> str:
        return ".pth"


class NumpyBackend(Backend):
    """NumPy backend implementation."""

    def zeros(self, shape: tuple[int, ...], dtype: str = "float64") -> TensorLike:
        return np.zeros(shape, dtype=dtype)

    def full(
        self, shape: tuple[int, ...], fill_value: float, dtype: str = "float64"
    ) -> TensorLike:
        return np.full(shape, fill_value, dtype=dtype)

    def tensor(self, data: list | Any, dtype: str = "float64") -> TensorLike:
        return np.array(data, dtype=dtype)

    def stack(self, arrays: list[TensorLike], axis: int = 0) -> TensorLike:
        return np.stack(arrays, axis=axis)

    def isnan(self, arr: TensorLike) -> TensorLike:
        return np.isnan(arr)

    def where(self, condition: TensorLike) -> tuple[TensorLike, ...]:
        return np.where(condition)

    def save(self, data: dict[str, Any], file_path: str) -> None:
        np.savez_compressed(file_path, **data)

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def file_extension(self) -> str:
        return ".npz"


def get_backend(backend_name: str = "torch") -> Backend:
    """Get backend instance.

    :param backend_name: Backend name ('torch' or 'numpy')
    :return: Backend instance
    """
    if backend_name == "torch":
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install it with 'pip install torch' or use backend='numpy'"
            )
        return TorchBackend()
    elif backend_name == "numpy":
        return NumpyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Use 'torch' or 'numpy'")
