"""Backend abstraction for torch and numpy.

This module provides a unified interface for tensor operations
that works with both PyTorch tensors and NumPy arrays.
"""

__docformat__ = "restructuredtext"
__all__ = ["Backend", "NumpyBackend", "TensorLike", "TorchBackend", "get_backend"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# TensorLike union type - supports both torch.Tensor and np.ndarray
if TYPE_CHECKING:
    TensorLike: TypeAlias = torch.Tensor | np.ndarray[Any, Any]
else:
    TensorLike: TypeAlias = Any  # Runtime: use Any for compatibility


class Backend(ABC):
    """Abstract backend interface for tensor operations."""

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: str = "float64") -> TensorLike:
        """Create array of zeros.

        :param shape: Shape of the array
        :param dtype: Data type
        :return: Zero-filled array
        """

    @abstractmethod
    def full(self, shape: tuple[int, ...], fill_value: float, dtype: str = "float64") -> TensorLike:
        """Create array filled with value.

        :param shape: Shape of the array
        :param fill_value: Fill value
        :param dtype: Data type
        :return: Filled array
        """

    @abstractmethod
    def tensor(self, data: list | Any, dtype: str = "float64") -> TensorLike:
        """Create array from data.

        :param data: Input data
        :param dtype: Data type
        :return: Array
        """

    @abstractmethod
    def stack(self, arrays: list[TensorLike], axis: int = 0) -> TensorLike:
        """Stack arrays along axis.

        :param arrays: List of arrays
        :param axis: Axis to stack along
        :return: Stacked array
        """

    @abstractmethod
    def isnan(self, arr: TensorLike) -> TensorLike:
        """Check for NaN values.

        :param arr: Input array
        :return: Boolean array
        """

    @abstractmethod
    def where(self, condition: TensorLike) -> tuple[TensorLike, ...]:
        """Find where condition is True.

        :param condition: Boolean array
        :return: Tuple of indices
        """

    @abstractmethod
    def save(self, data: dict[str, Any], file_path: str) -> None:
        """Save data to file.

        :param data: Data dictionary
        :param file_path: Output file path
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name.

        :return: Name of the backend
        """

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for saved files.

        :return: File extension
        """


class TorchBackend(Backend):
    """PyTorch backend implementation using ``torch`` tensors."""

    def zeros(self, shape: tuple[int, ...], dtype: str = "float64") -> TensorLike:
        """Create a zero-filled tensor via ``torch.zeros``.

        Maps string dtype names to ``torch`` dtype objects before allocation.

        :param shape: Shape of the tensor.
        :param dtype: Data type string (``"float64"``, ``"float32"``, ``"int64"``).
        :return: Zero-filled ``torch.Tensor``.
        """
        dtype_map = {
            "float64": torch.float64,
            "float32": torch.float32,
            "int64": torch.long,
        }
        return torch.zeros(shape, dtype=dtype_map.get(dtype, torch.float64))

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: str = "float64") -> TensorLike:
        """Create a constant-filled tensor via ``torch.full``.

        :param shape: Shape of the tensor.
        :param fill_value: Scalar value to fill.
        :param dtype: Data type string (``"float64"``, ``"float32"``, ``"int64"``).
        :return: Constant-filled ``torch.Tensor``.
        """
        dtype_map = {
            "float64": torch.float64,
            "float32": torch.float32,
            "int64": torch.long,
        }
        return torch.full(shape, fill_value, dtype=dtype_map.get(dtype, torch.float64))

    def tensor(self, data: list | Any, dtype: str = "float64") -> TensorLike:
        """Create a ``torch.Tensor`` from Python data.

        :param data: Input data (list, scalar, or nested sequence).
        :param dtype: Data type string (``"float64"``, ``"float32"``, ``"int64"``).
        :return: ``torch.Tensor`` constructed from *data*.
        """
        dtype_map = {
            "float64": torch.float64,
            "float32": torch.float32,
            "int64": torch.long,
        }
        return torch.tensor(data, dtype=dtype_map.get(dtype, torch.float64))

    def stack(self, arrays: list[TensorLike], axis: int = 0) -> TensorLike:
        """Stack tensors along *axis* via ``torch.stack``.

        Non-tensor inputs are converted to ``torch.Tensor`` before stacking.

        :param arrays: List of tensors or array-like objects.
        :param axis: Dimension to stack along.
        :return: Stacked ``torch.Tensor``.
        """
        return torch.stack(
            [a if isinstance(a, torch.Tensor) else torch.tensor(a) for a in arrays], dim=axis
        )

    def isnan(self, arr: TensorLike) -> TensorLike:
        """Element-wise NaN check via ``torch.isnan``.

        :param arr: Input tensor or array-like.
        :return: Boolean ``torch.Tensor`` with ``True`` at NaN positions.
        """
        tensor = arr if isinstance(arr, torch.Tensor) else torch.tensor(arr)
        return torch.isnan(tensor)

    def where(self, condition: TensorLike) -> tuple[TensorLike, ...]:
        """Return indices where *condition* is ``True`` via ``torch.where``.

        :param condition: Boolean tensor or array-like.
        :return: Tuple of index tensors, one per dimension.
        """
        tensor = condition if isinstance(condition, torch.Tensor) else torch.tensor(condition)
        return cast(tuple[TensorLike, ...], torch.where(tensor))

    def save(self, data: dict[str, Any], file_path: str) -> None:
        """Serialize *data* to disk via ``torch.save``.

        :param data: Dictionary mapping names to tensors/values.
        :param file_path: Destination path (typically ``*.pth``).
        """
        torch.save(data, file_path)

    @property
    def name(self) -> str:
        """Return ``"torch"`` as the backend identifier."""
        return "torch"

    @property
    def file_extension(self) -> str:
        """Return ``".pth"`` for PyTorch serialization files."""
        return ".pth"


class NumpyBackend(Backend):
    """NumPy backend implementation using ``np.ndarray``."""

    def zeros(self, shape: tuple[int, ...], dtype: str = "float64") -> TensorLike:
        """Create a zero-filled array via ``np.zeros``.

        :param shape: Shape of the array.
        :param dtype: NumPy dtype string (passed directly to ``np.zeros``).
        :return: Zero-filled ``np.ndarray``.
        """
        return np.zeros(shape, dtype=dtype)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: str = "float64") -> TensorLike:
        """Create a constant-filled array via ``np.full``.

        :param shape: Shape of the array.
        :param fill_value: Scalar value to fill.
        :param dtype: NumPy dtype string.
        :return: Constant-filled ``np.ndarray``.
        """
        return np.full(shape, fill_value, dtype=dtype)

    def tensor(self, data: list | Any, dtype: str = "float64") -> TensorLike:
        """Create an ``np.ndarray`` from Python data via ``np.array``.

        :param data: Input data (list, scalar, or nested sequence).
        :param dtype: NumPy dtype string.
        :return: ``np.ndarray`` constructed from *data*.
        """
        return np.array(data, dtype=dtype)

    def stack(self, arrays: list[TensorLike], axis: int = 0) -> TensorLike:
        """Stack arrays along *axis* via ``np.stack``.

        :param arrays: List of ``np.ndarray`` objects.
        :param axis: Axis to stack along.
        :return: Stacked ``np.ndarray``.
        """
        return np.stack(arrays, axis=axis)

    def isnan(self, arr: TensorLike) -> TensorLike:
        """Element-wise NaN check via ``np.isnan``.

        :param arr: Input array.
        :return: Boolean ``np.ndarray`` with ``True`` at NaN positions.
        """
        return np.isnan(arr)

    def where(self, condition: TensorLike) -> tuple[TensorLike, ...]:
        """Return indices where *condition* is ``True`` via ``np.where``.

        :param condition: Boolean array.
        :return: Tuple of index arrays, one per dimension.
        """
        return cast(tuple[TensorLike, ...], np.where(condition))

    def save(self, data: dict[str, Any], file_path: str) -> None:
        """Serialize *data* to compressed ``.npz`` via ``np.savez_compressed``.

        :param data: Dictionary mapping names to arrays/values.
        :param file_path: Destination path (typically ``*.npz``).
        """
        np.savez_compressed(file_path, **data)

    @property
    def name(self) -> str:
        """Return ``"numpy"`` as the backend identifier."""
        return "numpy"

    @property
    def file_extension(self) -> str:
        """Return ``".npz"`` for NumPy compressed archive files."""
        return ".npz"


def get_backend(backend_name: str = "torch") -> Backend:
    """Get backend instance.

    :param backend_name: Backend name ('torch' or 'numpy')
    :return: Backend instance
    """
    if backend_name == "torch":
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install it with "
                "'pip install torch' or use backend='numpy'"
            )
        return TorchBackend()
    if backend_name == "numpy":
        return NumpyBackend()
    raise ValueError(f"Unknown backend: {backend_name}. Use 'torch' or 'numpy'")
