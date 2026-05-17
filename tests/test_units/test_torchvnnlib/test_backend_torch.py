"""Unit tests for TorchBackend implementation.

Tests the torch-specific backend for tensor operations, file I/O, and dtype handling.
Requires PyTorch to be installed.
"""

__docformat__ = "restructuredtext"

import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from torchvnnlib._backend import TorchBackend, get_backend

torch = pytest.importorskip("torch")  # Skip tests if torch not available


class TestTorchBackendBasics:
    """Test basic TorchBackend properties and instantiation."""

    def test_torch_backend_name(self):
        """Test that backend name is 'torch'."""
        backend = TorchBackend()
        assert backend.name == "torch"

    def test_torch_backend_file_extension(self):
        """Test that torch backend uses .pth extension."""
        backend = TorchBackend()
        assert backend.file_extension == ".pth"

    def test_get_backend_torch(self):
        """Test get_backend() returns TorchBackend."""
        backend = get_backend("torch")
        assert isinstance(backend, TorchBackend)
        assert backend.name == "torch"

    def test_backend_instance_check(self):
        """Test isinstance check on TorchBackend."""
        backend = TorchBackend()
        assert isinstance(backend, TorchBackend)


class TestTorchTensorCreation:
    """Test tensor creation operations."""

    def test_zeros_basic(self):
        """Test creating zero tensor."""
        backend = TorchBackend()
        t = backend.zeros((3, 2))
        assert t.shape == (3, 2)
        assert torch.all(t == 0.0)

    @pytest.mark.parametrize(
        "shape",
        [
            (5,),
            (2, 3, 4),
            (0,),
            (100, 100),
        ],
    )
    def test_zeros_with_shape(self, shape):
        """Test creating zero tensors with various shapes (STR11: merged with test_tensor_shapes)."""
        backend = TorchBackend()
        t = backend.zeros(shape)
        assert t.shape == shape

    # [REVIEW] Deleted: test_zeros_1d, test_zeros_3d. STR2: merged 2 MED_DUP.
    # [REVIEW] Deleted: test_tensor_shapes (TestTorchEdgeCases). STR11: merged into test_zeros_with_shape.

    @pytest.mark.parametrize(
        ("dtype", "expected_dtype"),
        [
            ("float64", torch.float64),
            ("float32", torch.float32),
        ],
    )
    def test_zeros_with_dtype(self, dtype, expected_dtype):
        """Test zeros with dtype specification."""
        backend = TorchBackend()
        t = backend.zeros((2, 2), dtype=dtype)
        assert t.dtype == expected_dtype

    # [REVIEW] Deleted: test_zeros_dtype_float64, test_zeros_dtype_float32.
    # STR2: merged 2 MED_DUP.

    def test_full_basic(self):
        """Test creating tensor filled with value."""
        backend = TorchBackend()
        t = backend.full((2, 3), 5.0)
        assert t.shape == (2, 3)
        assert torch.all(t == 5.0)

    def test_full_nan(self):
        """Test creating tensor filled with NaN."""
        backend = TorchBackend()
        t = backend.full((2, 2), float("nan"))
        assert t.shape == (2, 2)
        assert torch.all(torch.isnan(t))

    def test_full_negative(self):
        """Test creating tensor filled with negative value."""
        backend = TorchBackend()
        t = backend.full((3, 3), -1.5)
        assert torch.all(t == -1.5)

    def test_tensor_from_list(self):
        """Test creating tensor from list."""
        backend = TorchBackend()
        t = backend.tensor([1.0, 2.0, 3.0], dtype="float64")
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        assert torch.allclose(t, expected)

    def test_tensor_from_nested_list(self):
        """Test creating tensor from nested list."""
        backend = TorchBackend()
        t = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.shape == (2, 2)


class TestTorchArrayOperations:
    """Test array/tensor operations."""

    @pytest.mark.parametrize(
        ("t1_data", "t2_data", "axis", "expected_shape"),
        [
            ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], 0, (2, 2, 2)),
            ([1.0, 2.0], [3.0, 4.0], 0, (2, 2)),
            ([[1.0, 2.0]], [[3.0, 4.0]], 1, (1, 2, 2)),
        ],
    )
    def test_stack_operations(self, t1_data, t2_data, axis, expected_shape):
        """Test stacking tensors along various axes (STR11: merged 3 tests)."""
        backend = TorchBackend()
        t1 = backend.tensor(t1_data)
        t2 = backend.tensor(t2_data)
        stacked = backend.stack([t1, t2], axis=axis)
        assert stacked.shape == expected_shape

    def test_isnan_detection(self):
        """Test NaN detection in tensor."""
        backend = TorchBackend()
        t = backend.full((2, 2), float("nan"))
        nan_mask = backend.isnan(t)
        assert torch.all(nan_mask)

    def test_isnan_no_nans(self):
        """Test NaN detection when no NaNs present."""
        backend = TorchBackend()
        t = backend.tensor([1.0, 2.0, 3.0])
        nan_mask = backend.isnan(t)
        assert not torch.any(nan_mask)

    def test_isnan_mixed(self):
        """Test NaN detection with mixed values."""
        backend = TorchBackend()
        t = backend.tensor([1.0, float("nan"), 3.0])
        nan_mask = backend.isnan(t)
        assert nan_mask[1]
        assert not nan_mask[0]
        assert not nan_mask[2]

    def test_where_condition(self):
        """Test where operation for conditional indexing."""
        backend = TorchBackend()
        t = backend.tensor([1.0, 2.0, 3.0, 4.0])
        condition = t > 2.0
        indices = backend.where(condition)
        assert len(indices) > 0


class TestTorchFileIO:
    """Test file I/O operations."""

    def test_save_basic(self):
        """Test saving tensor to file."""
        backend = TorchBackend()
        data = {"tensor": backend.tensor([1.0, 2.0, 3.0])}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.pth"
            backend.save(data, str(filepath))
            assert filepath.exists()

    def test_save_multiple_tensors(self):
        """Test saving multiple tensors in dict."""
        backend = TorchBackend()
        data = {
            "input": backend.tensor([[1.0, 2.0]]),
            "output": [backend.tensor([[3.0, 4.0]])],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.pth"
            backend.save(data, str(filepath))
            assert filepath.exists()

    def test_save_load_roundtrip(self):
        """Test saving and loading tensor."""
        backend = TorchBackend()
        original_data = {"value": backend.tensor([1.5, 2.5, 3.5])}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.pth"
            backend.save(original_data, str(filepath))

            loaded_data = torch.load(filepath, weights_only=False)
            assert torch.allclose(loaded_data["value"], original_data["value"])

    def test_save_empty_dict(self):
        """Test saving empty dictionary."""
        backend = TorchBackend()
        data: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.pth"
            backend.save(data, str(filepath))
            assert filepath.exists()

    def test_save_file_extension_pth(self):
        """Test that saved files use .pth extension."""
        backend = TorchBackend()
        data = {"test": backend.tensor([1.0])}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "data.pth"
            backend.save(data, str(filepath))
            assert filepath.suffix == ".pth"

    def test_save_overwrites_existing(self):
        """Test that save overwrites existing file."""
        backend = TorchBackend()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.pth"

            # Save first file
            data1 = {"value": backend.tensor([1.0])}
            backend.save(data1, str(filepath))
            mtime1 = filepath.stat().st_mtime

            # Wait a tiny bit to ensure time difference
            time.sleep(0.01)

            # Save second file (overwrite)
            data2 = {"value": backend.tensor([1.0, 2.0, 3.0])}
            backend.save(data2, str(filepath))
            mtime2 = filepath.stat().st_mtime

            # File should have been modified
            assert mtime2 >= mtime1
            # Verify file still exists and has correct data
            loaded = torch.load(str(filepath), weights_only=False)
            assert len(loaded["value"]) == 3


class TestTorchDtypeHandling:
    """Test dtype handling and conversions."""

    # [REVIEW] [DELETE_CANDIDATE] test_dtype_float64_explicit and
    # test_dtype_float32_explicit: truly redundant with test_zeros_with_dtype.
    # STR2: group=4, but these are exact duplicates.

    def test_tensor_dtype_inference(self):
        """Test dtype inference from values."""
        backend = TorchBackend()
        t = backend.tensor([1.0, 2.0, 3.0])
        assert t.dtype == torch.float64 or t.dtype == torch.float32


class TestTorchEdgeCases:
    """Test edge cases and error handling."""

    # [REVIEW] Deleted: test_empty_tensor, test_large_tensor.
    # STR2: merged 2 MED_DUP.
    # [REVIEW] Deleted: test_tensor_shapes. STR11: merged into test_zeros_with_shape (TestTorchTensorCreation).

    def test_negative_values(self):
        """Test tensor with negative values."""
        backend = TorchBackend()
        t = backend.tensor([-1.0, -2.0, -3.0])
        assert torch.all(t < 0)

    def test_scientific_notation(self):
        """Test tensor with scientific notation values."""
        backend = TorchBackend()
        t = backend.tensor([1.5e-10, -2.3e05])
        assert t[0] == pytest.approx(1.5e-10)
        assert t[1] == pytest.approx(-2.3e05)

    def test_very_small_values(self):
        """Test tensor with very small values."""
        backend = TorchBackend()
        t = backend.tensor([1e-308, 1e-300])
        assert torch.isfinite(t).all()

    def test_stack_single_tensor(self):
        """Test stacking single tensor."""
        backend = TorchBackend()
        t = backend.tensor([1.0, 2.0])
        stacked = backend.stack([t], axis=0)
        assert stacked.shape == (1, 2)

    def test_stack_many_tensors(self):
        """Test stacking many tensors."""
        backend = TorchBackend()
        tensors = [backend.tensor([float(i)]) for i in range(10)]
        stacked = backend.stack(tensors, axis=0)
        assert stacked.shape == (10, 1)

    def test_full_zero_value(self):
        """Test full with zero value."""
        backend = TorchBackend()
        t = backend.full((3, 3), 0.0)
        assert torch.all(t == 0.0)
