"""Unit tests for NumpyBackend implementation.

Tests the numpy-specific backend for array operations, file I/O, and dtype handling.
NumPy is always available, unlike PyTorch.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from torchvnnlib._backend import NumpyBackend, get_backend


class TestNumpyBackendBasics:
    """Test basic NumpyBackend properties and instantiation."""

    def test_numpy_backend_name(self):
        """Test that backend name is 'numpy'."""
        backend = NumpyBackend()
        assert backend.name == "numpy"

    def test_numpy_backend_file_extension(self):
        """Test that numpy backend uses .npz extension."""
        backend = NumpyBackend()
        assert backend.file_extension == ".npz"

    def test_get_backend_numpy(self):
        """Test get_backend() returns NumpyBackend."""
        backend = get_backend("numpy")
        assert isinstance(backend, NumpyBackend)
        assert backend.name == "numpy"

    def test_backend_instance_check(self):
        """Test isinstance check on NumpyBackend."""
        backend = NumpyBackend()
        assert isinstance(backend, NumpyBackend)


class TestNumpyArrayCreation:
    """Test array creation operations."""

    def test_zeros_basic(self):
        """Test creating zero array."""
        backend = NumpyBackend()
        arr = backend.zeros((3, 2))
        assert arr.shape == (3, 2)
        assert np.all(arr == 0.0)

    def test_zeros_1d(self):
        """Test creating 1D zero array."""
        backend = NumpyBackend()
        arr = backend.zeros((5,))
        assert arr.shape == (5,)

    def test_zeros_3d(self):
        """Test creating 3D zero array."""
        backend = NumpyBackend()
        arr = backend.zeros((2, 3, 4))
        assert arr.shape == (2, 3, 4)

    def test_zeros_dtype_float64(self):
        """Test zeros with float64 dtype."""
        backend = NumpyBackend()
        arr = backend.zeros((2, 2), dtype="float64")
        assert arr.dtype == np.float64

    def test_zeros_dtype_float32(self):
        """Test zeros with float32 dtype."""
        backend = NumpyBackend()
        arr = backend.zeros((2, 2), dtype="float32")
        assert arr.dtype == np.float32

    def test_full_basic(self):
        """Test creating array filled with value."""
        backend = NumpyBackend()
        arr = backend.full((2, 3), 5.0)
        assert arr.shape == (2, 3)
        assert np.all(arr == 5.0)

    def test_full_nan(self):
        """Test creating array filled with NaN."""
        backend = NumpyBackend()
        arr = backend.full((2, 2), float("nan"))
        assert arr.shape == (2, 2)
        assert np.all(np.isnan(arr))

    def test_full_negative(self):
        """Test creating array filled with negative value."""
        backend = NumpyBackend()
        arr = backend.full((3, 3), -1.5)
        assert np.all(arr == -1.5)

    def test_tensor_from_list(self):
        """Test creating array from list."""
        backend = NumpyBackend()
        arr = backend.tensor([1.0, 2.0, 3.0])
        assert np.allclose(arr, np.array([1.0, 2.0, 3.0]))

    def test_tensor_from_nested_list(self):
        """Test creating array from nested list."""
        backend = NumpyBackend()
        arr = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert arr.shape == (2, 2)


class TestNumpyArrayOperations:
    """Test array operations."""

    def test_stack_2d_arrays(self):
        """Test stacking 2D arrays."""
        backend = NumpyBackend()
        a1 = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        a2 = backend.tensor([[5.0, 6.0], [7.0, 8.0]])
        stacked = backend.stack([a1, a2], axis=0)
        assert stacked.shape == (2, 2, 2)

    def test_stack_axis_0(self):
        """Test stacking along axis 0."""
        backend = NumpyBackend()
        a1 = backend.tensor([1.0, 2.0])
        a2 = backend.tensor([3.0, 4.0])
        stacked = backend.stack([a1, a2], axis=0)
        assert stacked.shape == (2, 2)
        assert stacked[0, 0] == 1.0
        assert stacked[1, 0] == 3.0

    def test_stack_axis_1(self):
        """Test stacking along axis 1."""
        backend = NumpyBackend()
        a1 = backend.tensor([[1.0, 2.0]])
        a2 = backend.tensor([[3.0, 4.0]])
        stacked = backend.stack([a1, a2], axis=1)
        assert stacked.shape == (1, 2, 2)

    def test_isnan_detection(self):
        """Test NaN detection in array."""
        backend = NumpyBackend()
        arr = backend.full((2, 2), float("nan"))
        nan_mask = backend.isnan(arr)
        assert np.all(nan_mask)

    def test_isnan_no_nans(self):
        """Test NaN detection when no NaNs present."""
        backend = NumpyBackend()
        arr = backend.tensor([1.0, 2.0, 3.0])
        nan_mask = backend.isnan(arr)
        assert not np.any(nan_mask)

    def test_isnan_mixed(self):
        """Test NaN detection with mixed values."""
        backend = NumpyBackend()
        arr = backend.tensor([1.0, float("nan"), 3.0])
        nan_mask = backend.isnan(arr)
        assert nan_mask[1]
        assert not nan_mask[0]
        assert not nan_mask[2]

    def test_where_condition(self):
        """Test where operation for conditional indexing."""
        backend = NumpyBackend()
        arr = backend.tensor([1.0, 2.0, 3.0, 4.0])
        condition = arr > 2.0
        indices = backend.where(condition)
        assert len(indices) > 0


class TestNumpyFileIO:
    """Test file I/O operations with .npz format."""

    def test_save_basic(self):
        """Test saving arrays to file."""
        backend = NumpyBackend()
        data = {"array": backend.tensor([1.0, 2.0, 3.0])}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.npz"
            backend.save(data, str(filepath))
            assert filepath.exists()

    def test_save_multiple_arrays(self):
        """Test saving multiple arrays in dict."""
        backend = NumpyBackend()
        data = {
            "input": backend.tensor([[1.0, 2.0]]),
            "output": [backend.tensor([[3.0, 4.0]])],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.npz"
            backend.save(data, str(filepath))
            assert filepath.exists()

    def test_save_load_roundtrip(self):
        """Test saving and loading arrays."""
        backend = NumpyBackend()
        original_data = {"value": backend.tensor([1.5, 2.5, 3.5])}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.npz"
            backend.save(original_data, str(filepath))

            loaded_data = np.load(filepath, allow_pickle=True)
            assert np.allclose(loaded_data["value"], original_data["value"])

    def test_save_empty_dict(self):
        """Test saving empty dictionary."""
        backend = NumpyBackend()
        data = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.npz"
            backend.save(data, str(filepath))
            assert filepath.exists()

    def test_save_file_extension_npz(self):
        """Test that saved files use .npz extension."""
        backend = NumpyBackend()
        data = {"test": backend.tensor([1.0])}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "data.npz"
            backend.save(data, str(filepath))
            assert filepath.suffix == ".npz"

    def test_save_overwrites_existing(self):
        """Test that save overwrites existing file."""
        backend = NumpyBackend()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.npz"

            # Save first file
            data1 = {"value": backend.tensor([1.0])}
            backend.save(data1, str(filepath))
            mtime1 = filepath.stat().st_mtime

            # Wait a tiny bit to ensure time difference
            import time

            time.sleep(0.01)

            # Save second file (overwrite)
            data2 = {"value": backend.tensor([1.0, 2.0, 3.0])}
            backend.save(data2, str(filepath))
            mtime2 = filepath.stat().st_mtime

            # File should have been modified
            assert mtime2 >= mtime1
            # Verify file still exists and has correct data
            loaded = np.load(str(filepath), allow_pickle=True)
            assert len(loaded["value"]) == 3


class TestNumpyDtypeHandling:
    """Test dtype handling and conversions."""

    def test_dtype_float64_explicit(self):
        """Test explicit float64 dtype."""
        backend = NumpyBackend()
        arr = backend.zeros((2, 2), dtype="float64")
        assert arr.dtype == np.float64

    def test_dtype_float32_explicit(self):
        """Test explicit float32 dtype."""
        backend = NumpyBackend()
        arr = backend.zeros((2, 2), dtype="float32")
        assert arr.dtype == np.float32

    def test_tensor_dtype_inference(self):
        """Test dtype inference from values."""
        backend = NumpyBackend()
        arr = backend.tensor([1.0, 2.0, 3.0])
        assert arr.dtype in [np.float64, np.float32]


class TestNumpyEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_array(self):
        """Test creating empty array."""
        backend = NumpyBackend()
        arr = backend.zeros((0,))
        assert arr.shape == (0,)

    def test_large_array(self):
        """Test creating large array."""
        backend = NumpyBackend()
        arr = backend.zeros((100, 100))
        assert arr.shape == (100, 100)

    def test_negative_values(self):
        """Test array with negative values."""
        backend = NumpyBackend()
        arr = backend.tensor([-1.0, -2.0, -3.0])
        assert np.all(arr < 0)

    def test_scientific_notation(self):
        """Test array with scientific notation values."""
        backend = NumpyBackend()
        arr = backend.tensor([1.5e-10, -2.3e05])
        assert arr[0] == pytest.approx(1.5e-10)
        assert arr[1] == pytest.approx(-2.3e05)

    def test_very_small_values(self):
        """Test array with very small values."""
        backend = NumpyBackend()
        arr = backend.tensor([1e-308, 1e-300])
        assert np.isfinite(arr).all()

    def test_stack_single_array(self):
        """Test stacking single array."""
        backend = NumpyBackend()
        arr = backend.tensor([1.0, 2.0])
        stacked = backend.stack([arr], axis=0)
        assert stacked.shape == (1, 2)

    def test_stack_many_arrays(self):
        """Test stacking many arrays."""
        backend = NumpyBackend()
        arrays = [backend.tensor([float(i)]) for i in range(10)]
        stacked = backend.stack(arrays, axis=0)
        assert stacked.shape == (10, 1)

    def test_full_zero_value(self):
        """Test full with zero value."""
        backend = NumpyBackend()
        arr = backend.full((3, 3), 0.0)
        assert np.all(arr == 0.0)

    def test_array_is_ndarray(self):
        """Test that created arrays are numpy arrays."""
        backend = NumpyBackend()
        arr = backend.tensor([1.0, 2.0])
        assert isinstance(arr, np.ndarray)
