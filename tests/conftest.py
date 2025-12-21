"""Pytest configuration and fixtures for torchvnnlib tests."""

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
def benchmarks_dir(test_dir):
    """Benchmarks directory path."""
    return test_dir / "benchmarks"


@pytest.fixture(scope="session")
def results_dir(test_dir):
    """Results directory path."""
    results = test_dir / "results"
    results.mkdir(exist_ok=True)
    return results


@pytest.fixture(scope="session")
def baselines_dir(test_dir):
    """Baselines directory path."""
    baselines = test_dir / "baselines"
    baselines.mkdir(exist_ok=True)
    return baselines


@pytest.fixture(scope="session")
def torch_available():
    """Check if torch is available."""
    return importlib.util.find_spec("torch") is not None


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="Backend to use: torch, numpy, or both (default: both)",
    )
    parser.addoption(
        "--sample-size",
        action="store",
        type=int,
        default=None,
        help="Number of files to sample for testing",
    )


def pytest_configure(config):
    """Register custom marker."""
    config.addinivalue_line(
        "markers", "update: mark test as baseline update test (requires explicit -m update)"
    )


def pytest_generate_tests(metafunc):
    """Parametrize backend based on command line."""
    if "backend" in metafunc.fixturenames:
        backend_option = metafunc.config.getoption("backend")
        if backend_option == "torch":
            metafunc.parametrize("backend", ["torch"])
        elif backend_option == "numpy":
            metafunc.parametrize("backend", ["numpy"])
        elif backend_option == "both" or backend_option is None:
            metafunc.parametrize("backend", ["torch", "numpy"])


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
