"""TorchVNNLib Regression Testing Suite.

Compares current outputs (results/) against golden references (baselines/) to ensure
code changes don't break existing functionality.

Workflow:
    1. Run test_torchvnnlib.py to generate results/
    2. Run this test to compare results/ vs baselines/
    3. If needed, run update_baselines.py to update baselines/

Usage::

    # Verify current results match baselines
    pytest test_torchvnnlib_regression.py -v
    python test_torchvnnlib_regression.py

    # Test specific benchmark
    pytest test_torchvnnlib_regression.py -v -k acasxu
"""

import sys
from pathlib import Path

import pytest

from torchvnnlib.tests.utils import compare_output_folders, find_benchmarks_folders


def get_all_benchmarks():
    """Collect all benchmark names for parametrization.

    :return: List of benchmark names
    """
    test_dir = Path(__file__).parent
    # benchmarks dir is at tests/test_benchmarks/benchmarks
    benchmarks_dir = test_dir / "benchmarks"
    if not benchmarks_dir.exists():
        return []

    benchmarks = find_benchmarks_folders(str(benchmarks_dir))
    return [Path(b).name for b in benchmarks]


@pytest.mark.benchmark
@pytest.mark.parametrize("benchmark_name", get_all_benchmarks())
def test_verify_baseline(benchmark_name, test_dir, baselines_dir, results_dir):
    """Verify results against baseline for one benchmark.

    Compares outputs in results/{benchmark_name}/ vs baselines/{benchmark_name}/

    :param benchmark_name: Name of benchmark (from parametrize)
    :param test_dir: Test directory fixture
    :param baselines_dir: Baselines directory fixture
    :param results_dir: Results directory fixture
    """
    baseline_bench_dir = baselines_dir / benchmark_name
    results_bench_dir = results_dir / benchmark_name

    # Check baseline exists
    if not baseline_bench_dir.exists():
        pytest.skip(f"No baselines found for {benchmark_name}")

    # Check results exist
    if not results_bench_dir.exists():
        pytest.fail(f"No results found for {benchmark_name}. Run test_torchvnnlib.py first.")

    # Find all property subdirectories in results
    property_dirs = [d for d in results_bench_dir.iterdir() if d.is_dir()]

    if not property_dirs:
        pytest.skip(f"No properties found in results/{benchmark_name}/")

    # Compare each property
    passed = 0
    failed = []
    missing = []

    for property_dir in sorted(property_dirs):
        property_name = property_dir.name
        baseline_property_path = baseline_bench_dir / property_name
        results_property_path = results_bench_dir / property_name

        # Check if baseline exists for this property
        if not baseline_property_path.exists():
            missing.append(property_name)
            continue

        # Compare folders
        success, mismatches = compare_output_folders(
            str(baseline_property_path), str(results_property_path)
        )

        if success:
            passed += 1
        else:
            failed.append((property_name, mismatches))

    # Report results
    if missing:
        print(f"\nWarning: {len(missing)} properties missing baselines in {benchmark_name}")

    # Assert that verification passed
    if failed:
        error_msg = f"Baseline mismatches for {benchmark_name}:\n"
        for prop_name, mismatches in failed[:3]:  # Show first 3 failures
            error_msg += f"  {prop_name}: {', '.join(mismatches[:2])}\n"
        raise AssertionError(error_msg)

    assert passed > 0 or len(missing) > 0, f"No tests run for {benchmark_name}"


def main():
    """Backward compatible entry point supporting pytest execution."""
    pytest_args = [__file__, "-v"]
    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
