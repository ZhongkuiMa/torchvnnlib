"""Unit tests for TorchBackend implementation.

Tests the torch-specific backend for tensor operations, file I/O, and dtype handling.
Requires PyTorch to be installed.
"""

import tempfile
from pathlib import Path

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

    def test_zeros_1d(self):
        """Test creating 1D zero tensor."""
        backend = TorchBackend()
        t = backend.zeros((5,))
        assert t.shape == (5,)

    def test_zeros_3d(self):
        """Test creating 3D zero tensor."""
        backend = TorchBackend()
        t = backend.zeros((2, 3, 4))
        assert t.shape == (2, 3, 4)

    def test_zeros_dtype_float64(self):
        """Test zeros with float64 dtype."""
        backend = TorchBackend()
        t = backend.zeros((2, 2), dtype="float64")
        assert t.dtype == torch.float64

    def test_zeros_dtype_float32(self):
        """Test zeros with float32 dtype."""
        backend = TorchBackend()
        t = backend.zeros((2, 2), dtype="float32")
        assert t.dtype == torch.float32

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

    def test_stack_2d_tensors(self):
        """Test stacking 2D tensors."""
        backend = TorchBackend()
        t1 = backend.tensor([[1.0, 2.0], [3.0, 4.0]])
        t2 = backend.tensor([[5.0, 6.0], [7.0, 8.0]])
        stacked = backend.stack([t1, t2], axis=0)
        assert stacked.shape == (2, 2, 2)

    def test_stack_axis_0(self):
        """Test stacking along axis 0."""
        backend = TorchBackend()
        t1 = backend.tensor([1.0, 2.0])
        t2 = backend.tensor([3.0, 4.0])
        stacked = backend.stack([t1, t2], axis=0)
        assert stacked.shape == (2, 2)
        assert stacked[0, 0] == 1.0
        assert stacked[1, 0] == 3.0

    def test_stack_axis_1(self):
        """Test stacking along axis 1."""
        backend = TorchBackend()
        t1 = backend.tensor([[1.0, 2.0]])
        t2 = backend.tensor([[3.0, 4.0]])
        stacked = backend.stack([t1, t2], axis=1)
        assert stacked.shape == (1, 2, 2)

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
        data = {}

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
            import time

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

    def test_dtype_float64_explicit(self):
        """Test explicit float64 dtype."""
        backend = TorchBackend()
        t = backend.zeros((2, 2), dtype="float64")
        assert t.dtype == torch.float64

    def test_dtype_float32_explicit(self):
        """Test explicit float32 dtype."""
        backend = TorchBackend()
        t = backend.zeros((2, 2), dtype="float32")
        assert t.dtype == torch.float32

    def test_tensor_dtype_inference(self):
        """Test dtype inference from values."""
        backend = TorchBackend()
        t = backend.tensor([1.0, 2.0, 3.0])
        assert t.dtype == torch.float64 or t.dtype == torch.float32


class TestTorchEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tensor(self):
        """Test creating empty tensor."""
        backend = TorchBackend()
        t = backend.zeros((0,))
        assert t.shape == (0,)

    def test_large_tensor(self):
        """Test creating large tensor."""
        backend = TorchBackend()
        t = backend.zeros((100, 100))
        assert t.shape == (100, 100)

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
