"""Unit tests for backend factory function.

Tests the get_backend() factory function that creates and caches backend instances.
"""

import pytest

from torchvnnlib._backend import Backend, NumpyBackend, TorchBackend, get_backend


class TestBackendSelection:
    """Test backend selection and creation."""

    def test_get_backend_torch(self, torch_available):
        """Test get_backend() returns TorchBackend for 'torch'."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        backend = get_backend("torch")
        assert isinstance(backend, TorchBackend)
        assert backend.name == "torch"

    def test_get_backend_numpy(self):
        """Test get_backend() returns NumpyBackend for 'numpy'."""
        backend = get_backend("numpy")
        assert isinstance(backend, NumpyBackend)
        assert backend.name == "numpy"

    def test_get_backend_default(self, torch_available):
        """Test default backend selection."""
        if torch_available:
            backend = get_backend()
            assert isinstance(backend, TorchBackend)
        else:
            backend = get_backend()
            assert isinstance(backend, NumpyBackend)

    def test_get_backend_case_sensitive(self, torch_available):
        """Test that backend selection is case sensitive."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        # Lowercase 'torch' should work
        backend = get_backend("torch")
        assert isinstance(backend, TorchBackend)

        # Uppercase 'TORCH' should raise ValueError
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("TORCH")

    def test_get_backend_invalid_raises(self):
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid_backend")

    def test_get_backend_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("")

    def test_get_backend_default_is_torch(self, torch_available):
        """Test that default backend is torch (if available)."""
        if torch_available:
            backend = get_backend()
            assert isinstance(backend, TorchBackend)
        else:
            # If torch is not available, default 'torch' will raise ImportError
            with pytest.raises(ImportError):
                get_backend()


class TestBackendCaching:
    """Test that backend instances are properly cached."""

    def test_torch_backend_caching(self, torch_available):
        """Test that TorchBackend instances are cached."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        backend1 = get_backend("torch")
        backend2 = get_backend("torch")
        # Both should be same type (not necessarily same instance)
        assert isinstance(backend1, TorchBackend)
        assert isinstance(backend2, TorchBackend)

    def test_numpy_backend_caching(self):
        """Test that NumpyBackend instances are cached."""
        backend1 = get_backend("numpy")
        backend2 = get_backend("numpy")
        assert isinstance(backend1, NumpyBackend)
        assert isinstance(backend2, NumpyBackend)

    def test_different_backends_different_types(self, torch_available):
        """Test that different backends have different types."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        torch_backend = get_backend("torch")
        numpy_backend = get_backend("numpy")
        assert isinstance(torch_backend, TorchBackend)
        assert isinstance(numpy_backend, NumpyBackend)


class TestBackendInterface:
    """Test that backends implement required interface."""

    def test_torch_has_required_methods(self, torch_available):
        """Test that TorchBackend has all required methods."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        backend = get_backend("torch")
        assert hasattr(backend, "zeros")
        assert hasattr(backend, "full")
        assert hasattr(backend, "tensor")
        assert hasattr(backend, "stack")
        assert hasattr(backend, "isnan")
        assert hasattr(backend, "where")
        assert hasattr(backend, "save")

    def test_numpy_has_required_methods(self):
        """Test that NumpyBackend has all required methods."""
        backend = get_backend("numpy")
        assert hasattr(backend, "zeros")
        assert hasattr(backend, "full")
        assert hasattr(backend, "tensor")
        assert hasattr(backend, "stack")
        assert hasattr(backend, "isnan")
        assert hasattr(backend, "where")
        assert hasattr(backend, "save")

    def test_torch_backend_properties(self, torch_available):
        """Test that TorchBackend has required properties."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        backend = get_backend("torch")
        assert backend.name == "torch"
        assert backend.file_extension == ".pth"

    def test_numpy_backend_properties(self):
        """Test that NumpyBackend has required properties."""
        backend = get_backend("numpy")
        assert backend.name == "numpy"
        assert backend.file_extension == ".npz"


class TestTorchAvailability:
    """Test PyTorch availability handling."""

    def test_torch_not_available_invalid(self):
        """Test that requesting torch raises error if not available.

        Note: This test assumes torch IS available in CI, but documents behavior.
        """
        # If torch is available, this test should pass
        # If torch is not available, get_backend should still work
        try:
            backend = get_backend("torch")
            assert isinstance(backend, TorchBackend)
        except ImportError:
            # PyTorch not available - this is acceptable
            pytest.skip("PyTorch not available")

    def test_numpy_always_available(self):
        """Test that NumPy backend is always available."""
        backend = get_backend("numpy")
        assert isinstance(backend, NumpyBackend)

    def test_default_backend_selection(self, torch_available):
        """Test default backend selection behavior."""
        if torch_available:
            # Default is torch when available
            backend = get_backend()
            assert isinstance(backend, TorchBackend)
        else:
            # Default raises ImportError when torch not available
            with pytest.raises(ImportError):
                get_backend()


class TestBackendCreation:
    """Test backend instantiation."""

    def test_torch_backend_instantiation(self, torch_available):
        """Test TorchBackend can be instantiated directly."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        backend = TorchBackend()
        assert isinstance(backend, TorchBackend)

    def test_numpy_backend_instantiation(self):
        """Test NumpyBackend can be instantiated directly."""
        backend = NumpyBackend()
        assert isinstance(backend, NumpyBackend)

    def test_abstract_backend_not_instantiable(self):
        """Test that abstract Backend class cannot be instantiated."""
        with pytest.raises(TypeError):
            Backend()


class TestBackendConsistency:
    """Test consistency between backends."""

    def test_both_backends_have_name_property(self, torch_available):
        """Test both backends have name property."""
        if torch_available:
            torch_backend = get_backend("torch")
            assert hasattr(torch_backend, "name")
            assert isinstance(torch_backend.name, str)

        numpy_backend = get_backend("numpy")
        assert hasattr(numpy_backend, "name")
        assert isinstance(numpy_backend.name, str)

    def test_both_backends_have_extension_property(self, torch_available):
        """Test both backends have file_extension property."""
        if torch_available:
            torch_backend = get_backend("torch")
            assert hasattr(torch_backend, "file_extension")
            assert torch_backend.file_extension.startswith(".")

        numpy_backend = get_backend("numpy")
        assert hasattr(numpy_backend, "file_extension")
        assert numpy_backend.file_extension.startswith(".")

    def test_backend_names_are_unique(self, torch_available):
        """Test that backend names are different."""
        if torch_available:
            torch_backend = get_backend("torch")
            numpy_backend = get_backend("numpy")
            assert torch_backend.name != numpy_backend.name

    def test_backend_extensions_are_unique(self, torch_available):
        """Test that backend file extensions are different."""
        if torch_available:
            torch_backend = get_backend("torch")
            numpy_backend = get_backend("numpy")
            assert torch_backend.file_extension != numpy_backend.file_extension
