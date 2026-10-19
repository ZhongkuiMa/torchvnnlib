"""Backend abstraction for torch and numpy.

This module provides a unified interface for tensor operations
that works with both PyTorch tensors and NumPy arrays.
"""

__docformat__ = "restructuredtext"
__all__ = ["Backend", "NumpyBackend", "TensorLike", "TorchBackend", "get_backend"]

from abc import ABC, abstractmethod
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np

# TensorLike union type - supports both torch.Tensor and np.ndarray
if TYPE_CHECKING:
    import torch

    TensorLike: TypeAlias = torch.Tensor | np.ndarray[Any, Any]
else:
    TensorLike: TypeAlias = Any  # Runtime: use Any for compatibility


@cache
def _load_torch() -> Any:
    """Import PyTorch only when the torch backend is explicitly requested."""
    try:
        return import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install torch' or use backend='numpy'"
        ) from exc


class Backend(ABC):
    """Abstract backend interface for tensor operations."""

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: str = "float64") -> TensorLike:
        """Create array of zeros.

        :param shape: Shape of the array.

        :param dtype: Data type.

        :return: Zero-filled array
        """

    @abstractmethod
    def full(self, shape: tuple[int, ...], fill_value: float, dtype: str = "float64") -> TensorLike:
        """Create array filled with value.

        :param shape: Shape of the array.

        :param fill_value: Fill value.

        :param dtype: Data type.

        :return: Filled array
        """

    @abstractmethod
    def tensor(self, data: list | Any, dtype: str = "float64") -> TensorLike:
        """Create array from data.

        :param data: Input data.

        :param dtype: Data type.

        :return: Array
        """

    @abstractmethod
    def stack(self, arrays: list[TensorLike], axis: int = 0) -> TensorLike:
        """Stack arrays along axis.

        :param arrays: List of arrays.

        :param axis: Axis to stack along.

        :return: Stacked array
        """

    @abstractmethod
    def isnan(self, arr: TensorLike) -> TensorLike:
        """Check for NaN values.

        :param arr: Input array.

        :return: Boolean array
        """

    @abstractmethod
    def where(self, condition: TensorLike) -> tuple[TensorLike, ...]:
        """Find where condition is True.

        :param condition: Boolean array.

        :return: Tuple of indices
        """

    @abstractmethod
    def save(self, data: dict[str, Any], file_path: str) -> None:
        """Save data to file.

        :param data: Data dictionary.

        :param file_path: Output file path.

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

    def __init__(self) -> None:
        """Load PyTorch after the caller explicitly selects this backend."""
        self._torch = _load_torch()
        self._dtype_map = {
            "float64": self._torch.float64,
            "float32": self._torch.float32,
            "int64": self._torch.long,
        }

    def zeros(self, shape: tuple[int, ...], dtype: str = "float64") -> TensorLike:
        """Create a zero-filled tensor via ``torch.zeros``.

        Maps string dtype names to ``torch`` dtype objects before allocation.

        :param shape: Shape of the tensor.
        :param dtype: Data type string (``"float64"``, ``"float32"``, ``"int64"``).
        :return: Zero-filled ``torch.Tensor``.
        """
        return cast(
            TensorLike,
            self._torch.zeros(shape, dtype=self._dtype_map.get(dtype, self._torch.float64)),
        )

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: str = "float64") -> TensorLike:
        """Create a constant-filled tensor via ``torch.full``.

        :param shape: Shape of the tensor.
        :param fill_value: Scalar value to fill.
        :param dtype: Data type string (``"float64"``, ``"float32"``, ``"int64"``).
        :return: Constant-filled ``torch.Tensor``.
        """
        return cast(
            TensorLike,
            self._torch.full(
                shape, fill_value, dtype=self._dtype_map.get(dtype, self._torch.float64)
            ),
        )

    def tensor(self, data: list | Any, dtype: str = "float64") -> TensorLike:
        """Create a ``torch.Tensor`` from Python data.

        :param data: Input data (list, scalar, or nested sequence).
        :param dtype: Data type string (``"float64"``, ``"float32"``, ``"int64"``).
        :return: ``torch.Tensor`` constructed from *data*.
        """
        return cast(
            TensorLike,
            self._torch.tensor(data, dtype=self._dtype_map.get(dtype, self._torch.float64)),
        )

    def stack(self, arrays: list[TensorLike], axis: int = 0) -> TensorLike:
        """Stack tensors along *axis* via ``torch.stack``.

        Non-tensor inputs are converted to ``torch.Tensor`` before stacking.

        :param arrays: List of tensors or array-like objects.
        :param axis: Dimension to stack along.
        :return: Stacked ``torch.Tensor``.
        """
        return cast(
            TensorLike,
            self._torch.stack(
                [a if isinstance(a, self._torch.Tensor) else self._torch.tensor(a) for a in arrays],
                dim=axis,
            ),
        )

    def isnan(self, arr: TensorLike) -> TensorLike:
        """Element-wise NaN check via ``torch.isnan``.

        :param arr: Input tensor or array-like.
        :return: Boolean ``torch.Tensor`` with ``True`` at NaN positions.
        """
        tensor = arr if isinstance(arr, self._torch.Tensor) else self._torch.tensor(arr)
        return cast(TensorLike, self._torch.isnan(tensor))

    def where(self, condition: TensorLike) -> tuple[TensorLike, ...]:
        """Return indices where *condition* is ``True`` via ``torch.where``.

        :param condition: Boolean tensor or array-like.
        :return: Tuple of index tensors, one per dimension.
        """
        tensor = (
            condition
            if isinstance(condition, self._torch.Tensor)
            else self._torch.tensor(condition)
        )
        return cast(tuple[TensorLike, ...], self._torch.where(tensor))

    def save(self, data: dict[str, Any], file_path: str) -> None:
        """Serialize *data* to disk via ``torch.save``.

        :param data: Dictionary mapping names to tensors/values.
        :param file_path: Destination path (typically ``*.pth``).
        """
        self._torch.save(data, file_path)

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


def get_backend(backend_name: Literal["torch", "numpy"] = "torch") -> Backend:
    """Get backend instance.

    :param backend_name: ``"torch"`` or ``"numpy"``.
    :return: Backend instance.
    :raises ImportError: If PyTorch is not installed and ``"torch"`` is requested.
    :raises ValueError: If ``backend_name`` is not one of the accepted values.
    """
    if backend_name == "torch":
        return TorchBackend()
    if backend_name == "numpy":
        return NumpyBackend()
    raise ValueError(f"Unknown backend: {backend_name}. Use 'torch' or 'numpy'")
