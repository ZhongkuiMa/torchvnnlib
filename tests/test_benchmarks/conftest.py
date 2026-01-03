"""Fixtures for benchmark tests in test_benchmarks directory."""

import pytest


@pytest.fixture(scope="session")
def benchmarks_dir(test_dir):
    """Benchmarks directory path (tests/test_benchmarks/benchmarks)."""
    benchmarks = test_dir / "benchmarks"
    benchmarks.mkdir(parents=True, exist_ok=True)
    return benchmarks


@pytest.fixture(scope="session")
def results_dir(test_dir):
    """Results directory path (tests/test_benchmarks/results)."""
    results = test_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    return results


@pytest.fixture(scope="session")
def baselines_dir(test_dir):
    """Baselines directory path (tests/test_benchmarks/baselines)."""
    baselines = test_dir / "baselines"
    baselines.mkdir(parents=True, exist_ok=True)
    return baselines


def pytest_addoption(parser):
    """Add custom command line options for benchmark tests."""
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


def pytest_generate_tests(metafunc):
    """Parametrize backend based on command line option."""
    if "backend" in metafunc.fixturenames:
        backend_option = metafunc.config.getoption("backend")
        if backend_option == "torch":
            metafunc.parametrize("backend", ["torch"])
        elif backend_option == "numpy":
            metafunc.parametrize("backend", ["numpy"])
        elif backend_option == "both" or backend_option is None:
            metafunc.parametrize("backend", ["torch", "numpy"])
