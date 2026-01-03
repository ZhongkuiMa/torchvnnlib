"""Shared pytest configuration and fixtures for all torchvnnlib tests."""

import importlib.util
import sys
from pathlib import Path

import pytest

# Add test directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def test_dir():
    """Test directory path."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def torch_available():
    """Check if torch is available."""
    return importlib.util.find_spec("torch") is not None


@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Temporary directory for tests."""
    return tmp_path_factory.mktemp("test_data")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "benchmark: mark test as slow benchmark test (deselect with '-m \"not benchmark\"')",
    )
    config.addinivalue_line(
        "markers", "update: mark test as baseline update test (requires explicit -m update)"
    )
    config.addinivalue_line("markers", "torch: mark test as requiring PyTorch")
    config.addinivalue_line("markers", "numpy: mark test as requiring NumPy")


def pytest_collection_modifyitems(config, items):
    """Skip update tests by default unless -m update is passed."""
    # Check if user explicitly requested update tests with -m update
    update_requested = config.option.markexpr and "update" in config.option.markexpr

    # If not explicitly requested, skip tests marked with "update"
    if not update_requested:
        skip_update = pytest.mark.skip(reason="Update tests require explicit -m update flag")
        for item in items:
            if "update" in item.keywords:
                item.add_marker(skip_update)
