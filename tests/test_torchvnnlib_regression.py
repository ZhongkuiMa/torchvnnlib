"""TorchVNNLib Regression Testing Suite.

Baseline management for regression testing - creates and verifies golden reference
outputs to ensure code changes don't break existing functionality.

Baselines are created using AST-only conversion (detect_fast_type=False) to ensure
correctness. Verification compares current outputs against these baselines.

Usage::

    # Verify baselines (default, safe, read-only)
    pytest test_torchvnnlib_regression.py -v
    python test_torchvnnlib_regression.py

    # Update baselines (requires explicit marker - be careful!)
    pytest test_torchvnnlib_regression.py -v -m update
    python test_torchvnnlib_regression.py update
"""

import shutil
import sys
import time
from pathlib import Path

import pytest

from torchvnnlib import TorchVNNLIB
from torchvnnlib.tests.utils import (
    compare_output_folders,
    find_all_vnnlib_files,
    find_benchmarks_folders,
    get_benchmark_name,
)


def get_baseline_path(vnnlib_path: str, baselines_dir: str = "baselines") -> str:
    """Get baseline folder path for a vnnlib file with benchmark subdirectory.

    The baseline folder is stored in a subdirectory matching the benchmark name.
    For example: benchmarks/acasxu/prop_6.vnnlib -> baselines/acasxu/prop_6/

    :param vnnlib_path: Path to vnnlib file
    :param baselines_dir: Root directory to store baseline folders
    :return: Path to baseline folder
    """
    # Extract benchmark name from path
    benchmark_name = get_benchmark_name(vnnlib_path)

    # Get vnnlib basename without .vnnlib extension
    basename = Path(vnnlib_path).name
    basename = basename.removesuffix(".vnnlib")

    # Create path with benchmark subdirectory
    return str(Path(baselines_dir) / benchmark_name / basename)


def get_results_path(vnnlib_path: str, results_dir: str = "results") -> str:
    """Get results folder path for a vnnlib file with benchmark subdirectory.

    :param vnnlib_path: Path to vnnlib file
    :param results_dir: Root directory to store result folders
    :return: Path to results folder
    """
    # Extract benchmark name from path
    benchmark_name = get_benchmark_name(vnnlib_path)

    # Get vnnlib basename without .vnnlib extension
    basename = Path(vnnlib_path).name
    basename = basename.removesuffix(".vnnlib")

    # Create path with benchmark subdirectory
    return str(Path(results_dir) / benchmark_name / basename)


def update_baseline(vnnlib_path: str, baselines_dir: str = "baselines"):
    """Create or update baseline for ONE vnnlib file.

    Runs TorchVNNLib.convert() with AST-only (detect_fast_type=False) and saves
    output to baselines directory. This ensures baselines are created using the
    known-correct AST method.

    :param vnnlib_path: Path to vnnlib file
    :param baselines_dir: Root directory to store baseline folders
    :return: Path to baseline folder
    """
    # IMPORTANT: Use AST-only (no fast_type) for baseline generation to ensure correctness
    converter = TorchVNNLIB(detect_fast_type=False)
    baseline_path = get_baseline_path(vnnlib_path, baselines_dir)

    # Remove existing baseline if it exists
    baseline_path_obj = Path(baseline_path)
    if baseline_path_obj.exists():
        shutil.rmtree(baseline_path_obj)

    # Run conversion
    converter.convert(vnnlib_path, target_folder_path=baseline_path)

    # Count .pth files
    pth_count = sum(1 for f in baseline_path_obj.rglob("*.pth"))

    benchmark_name = get_benchmark_name(vnnlib_path)
    print(f"OK - [{benchmark_name}] {Path(vnnlib_path).name} ({pth_count} files)")

    return baseline_path


def compare_baseline(
    vnnlib_path: str, baselines_dir: str = "baselines", results_dir: str = "results"
) -> bool:
    """Compare ONE vnnlib file's output against its baseline.

    Generates output to results/ folder and compares against baselines/ folder.

    :param vnnlib_path: Path to vnnlib file
    :param baselines_dir: Root directory containing baseline folders
    :param results_dir: Root directory to store result folders
    :return: True if outputs match baseline, False otherwise
    """
    baseline_path = get_baseline_path(vnnlib_path, baselines_dir)
    results_path = get_results_path(vnnlib_path, results_dir)

    baseline_path_obj = Path(baseline_path)
    if not baseline_path_obj.exists():
        return False

    # Remove existing results if they exist
    results_path_obj = Path(results_path)
    if results_path_obj.exists():
        shutil.rmtree(results_path_obj)

    # Run conversion
    converter = TorchVNNLIB()
    converter.convert(vnnlib_path, target_folder_path=results_path)

    # Compare folders
    success, mismatches = compare_output_folders(baseline_path, results_path)

    return success


def update_all_benchmarks(
    benchmarks_dir: str = "benchmarks",
    baselines_dir: str = "baselines",
    max_per_benchmark: int = 20,
):
    """Create/update baselines for all vnnlib files.

    :param benchmarks_dir: Root directory of benchmarks
    :param baselines_dir: Root directory to store baseline folders
    :param max_per_benchmark: Maximum files per benchmark to process
    """
    benchmark_dirs = find_benchmarks_folders(benchmarks_dir)
    vnnlib_files = find_all_vnnlib_files(benchmark_dirs, num_limit=max_per_benchmark)
    print(f"Creating baselines for {len(vnnlib_files)} files")
    print("=" * 70)

    success = 0
    failed = []
    start_time = time.perf_counter()

    for i, vnnlib_path in enumerate(vnnlib_files, 1):
        file_start = time.perf_counter()
        print(f"[{i}/{len(vnnlib_files)}] ", end="", flush=True)
        try:
            update_baseline(vnnlib_path, baselines_dir)
            success += 1
            elapsed = time.perf_counter() - file_start
            print(f"({elapsed:.2f}s)")
        except (OSError, RuntimeError, ValueError) as e:
            elapsed = time.perf_counter() - file_start
            print(f"ERROR ({elapsed:.2f}s): {e}")
            failed.append(vnnlib_path)

    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print("BASELINE UPDATE SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(vnnlib_files)}")
    print(f"Success: {success}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {total_time / len(vnnlib_files):.4f}s per file")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {Path(f).name}")


def verify_all_benchmarks(
    benchmarks_dir: str = "benchmarks",
    baselines_dir: str = "baselines",
    results_dir: str = "results",
    max_per_benchmark: int = 20,
):
    """Verify all vnnlib files against baselines.

    :param benchmarks_dir: Root directory of benchmarks
    :param baselines_dir: Root directory containing baseline folders
    :param results_dir: Root directory to store result folders
    :param max_per_benchmark: Maximum files per benchmark to verify
    """
    benchmark_dirs = find_benchmarks_folders(benchmarks_dir)
    vnnlib_files = find_all_vnnlib_files(benchmark_dirs, num_limit=max_per_benchmark)
    print(f"Verifying {len(vnnlib_files)} files against baselines")
    print("=" * 70)

    passed = 0
    failed = []
    missing = []
    start_time = time.perf_counter()

    for i, vnnlib_path in enumerate(vnnlib_files, 1):
        file_start = time.perf_counter()
        print(f"[{i}/{len(vnnlib_files)}] ", end="", flush=True)
        try:
            baseline_path = get_baseline_path(vnnlib_path, baselines_dir)
            if not Path(baseline_path).exists():
                print(f"SKIP (no baseline) - {Path(vnnlib_path).name}")
                missing.append(vnnlib_path)
                continue

            basename = Path(vnnlib_path).name
            if compare_baseline(vnnlib_path, baselines_dir, results_dir):
                passed += 1
                elapsed = time.perf_counter() - file_start
                print(f"OK ({elapsed:.2f}s) - {basename}")
            else:
                failed.append(vnnlib_path)
                elapsed = time.perf_counter() - file_start
                print(f"MISMATCH ({elapsed:.2f}s) - {basename}")
        except (OSError, RuntimeError, ValueError) as e:
            elapsed = time.perf_counter() - file_start
            print(f"ERROR ({elapsed:.2f}s): {e}")
            failed.append(vnnlib_path)

    total_time = time.perf_counter() - start_time
    tested = len(vnnlib_files) - len(missing)

    print("\n" + "=" * 70)
    print("BASELINE VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(vnnlib_files)}")
    print(f"Tested: {tested}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failed)}")
    if missing:
        print(f"Skipped (no baseline): {len(missing)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {total_time / tested:.4f}s per file" if tested > 0 else "")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {Path(f).name}")


def get_all_benchmarks():
    """Collect all benchmark names for parametrization.

    :return: List of benchmark names
    """
    test_dir = Path(__file__).parent
    benchmarks_dir = test_dir / "benchmarks"
    if not benchmarks_dir.exists():
        return []

    benchmarks = find_benchmarks_folders(str(benchmarks_dir))
    return [Path(b).name for b in benchmarks]


@pytest.mark.parametrize("benchmark_name", get_all_benchmarks())
@pytest.mark.update
def test_update_baseline(benchmark_name, test_dir, baselines_dir):
    """Update baseline for one benchmark.

    NOTE: This test is NOT run by default. It must be explicitly invoked:
      pytest test_baselines.py -v -m update
      OR: python test_baselines.py update

    :param benchmark_name: Name of benchmark (from parametrize)
    :param test_dir: Test directory fixture
    :param baselines_dir: Baselines directory fixture
    """
    vnnlib_files = find_all_vnnlib_files(
        [str(test_dir / "benchmarks" / benchmark_name)], num_limit=20
    )

    if not vnnlib_files:
        pytest.skip(f"No vnnlib files found in {benchmark_name}")

    success = 0
    failed = []

    for vnnlib_path in vnnlib_files:
        try:
            update_baseline(vnnlib_path, str(baselines_dir))
            success += 1
        except (OSError, RuntimeError, ValueError) as e:
            failed.append((vnnlib_path, str(e)))

    assert success > 0, f"No baselines updated for {benchmark_name}"
    if failed:
        # Log failures but don't fail the test if at least some succeeded
        print(f"\nWarning: {len(failed)} files failed to update")


@pytest.mark.parametrize("benchmark_name", get_all_benchmarks())
def test_verify_baseline(benchmark_name, test_dir, baselines_dir, results_dir):
    """Verify results against baseline for one benchmark.

    This is the DEFAULT test - runs with normal `pytest` command.

    :param benchmark_name: Name of benchmark (from parametrize)
    :param test_dir: Test directory fixture
    :param baselines_dir: Baselines directory fixture
    :param results_dir: Results directory fixture
    """
    vnnlib_files = find_all_vnnlib_files(
        [str(test_dir / "benchmarks" / benchmark_name)], num_limit=20
    )

    if not vnnlib_files:
        pytest.skip(f"No vnnlib files found in {benchmark_name}")

    passed = 0
    failed = []
    missing = []

    for vnnlib_path in vnnlib_files:
        baseline_path = get_baseline_path(vnnlib_path, str(baselines_dir))
        if not Path(baseline_path).exists():
            missing.append(Path(vnnlib_path).name)
            continue

        try:
            if compare_baseline(vnnlib_path, str(baselines_dir), str(results_dir)):
                passed += 1
            else:
                failed.append(Path(vnnlib_path).name)
        except (OSError, RuntimeError, ValueError) as e:
            failed.append(f"{Path(vnnlib_path).name} (ERROR: {e})")

    # Skip if no baselines exist for this benchmark
    if not passed and not failed and missing:
        pytest.skip(f"No baselines found for {benchmark_name}")

    # Assert that verification passed
    if failed:
        raise AssertionError(f"Baseline mismatches for {benchmark_name}: {', '.join(failed[:3])}")

    assert passed > 0 or len(missing) > 0, f"No tests run for {benchmark_name}"


def main():
    """Backward compatible entry point supporting both pytest and script execution."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"

    if mode == "update":
        # Run update tests (explicitly marked)
        pytest_args = [__file__, "-v", "-m", "update"]
    else:
        # Run verify tests (default - excludes update marker)
        pytest_args = [__file__, "-v"]

    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
