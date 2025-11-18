"""Baseline management for TorchVNNLib regression testing.

This module provides functions to create and compare conversion baselines.
Each vnnlib file is converted to a folder structure containing .pth files.

Usage::

    # Create/update baseline for one vnnlib file
    update_baseline("path/to/property.vnnlib")

    # Compare one vnnlib file against baseline
    compare_baseline("path/to/property.vnnlib")

    # Batch: Create baselines for all benchmarks
    update_all_benchmarks()
"""

import os
import shutil
import sys
import time

# Add parent directory to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from torchvnnlib import TorchVNNLIB
from utils import (
    find_benchmarks_folders,
    find_all_vnnlib_files,
    get_benchmark_name,
    compare_output_folders,
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
    basename = os.path.basename(vnnlib_path)
    if basename.endswith(".vnnlib"):
        basename = basename[:-7]

    # Create path with benchmark subdirectory
    return os.path.join(baselines_dir, benchmark_name, basename)


def get_results_path(vnnlib_path: str, results_dir: str = "results") -> str:
    """Get results folder path for a vnnlib file with benchmark subdirectory.

    :param vnnlib_path: Path to vnnlib file
    :param results_dir: Root directory to store result folders
    :return: Path to results folder
    """
    # Extract benchmark name from path
    benchmark_name = get_benchmark_name(vnnlib_path)

    # Get vnnlib basename without .vnnlib extension
    basename = os.path.basename(vnnlib_path)
    if basename.endswith(".vnnlib"):
        basename = basename[:-7]

    # Create path with benchmark subdirectory
    return os.path.join(results_dir, benchmark_name, basename)


def update_baseline(vnnlib_path: str, baselines_dir: str = "baselines"):
    """Create or update baseline for ONE vnnlib file.

    Runs TorchVNNLib.convert() with AST-only (detect_fast_type=False) and saves output to baselines directory.
    This ensures baselines are created using the known-correct AST method.

    :param vnnlib_path: Path to vnnlib file
    :param baselines_dir: Root directory to store baseline folders
    :return: Path to baseline folder
    """
    # IMPORTANT: Use AST-only (no fast_type) for baseline generation to ensure correctness
    converter = TorchVNNLIB(detect_fast_type=False)
    baseline_path = get_baseline_path(vnnlib_path, baselines_dir)

    # Remove existing baseline if it exists
    if os.path.exists(baseline_path):
        shutil.rmtree(baseline_path)

    # Run conversion
    converter.convert(vnnlib_path, target_folder_path=baseline_path)

    # Count .pth files
    pth_count = 0
    for root, dirs, files in os.walk(baseline_path):
        pth_count += sum(1 for f in files if f.endswith(".pth"))

    benchmark_name = get_benchmark_name(vnnlib_path)
    print(
        f"OK - [{benchmark_name}] {os.path.basename(vnnlib_path)} ({pth_count} files)"
    )

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

    if not os.path.exists(baseline_path):
        return False

    # Remove existing results if they exist
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

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
    """Helper to create/update baselines for all vnnlib files.

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
        except Exception as e:
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
    print(f"Average time: {total_time/len(vnnlib_files):.4f}s per file")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {os.path.basename(f)}")


def verify_all_benchmarks(
    benchmarks_dir: str = "benchmarks",
    baselines_dir: str = "baselines",
    results_dir: str = "results",
    max_per_benchmark: int = 20,
):
    """Helper to verify all vnnlib files against baselines.

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
            if not os.path.exists(baseline_path):
                print(f"SKIP (no baseline) - {os.path.basename(vnnlib_path)}")
                missing.append(vnnlib_path)
                continue

            basename = os.path.basename(vnnlib_path)
            if compare_baseline(vnnlib_path, baselines_dir, results_dir):
                passed += 1
                elapsed = time.perf_counter() - file_start
                print(f"OK ({elapsed:.2f}s) - {basename}")
            else:
                failed.append(vnnlib_path)
                elapsed = time.perf_counter() - file_start
                print(f"MISMATCH ({elapsed:.2f}s) - {basename}")
        except Exception as e:
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
    print(f"Average time: {total_time/tested:.4f}s per file" if tested > 0 else "")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {os.path.basename(f)}")


if __name__ == "__main__":
    import sys

    # Check command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        # Update baselines
        update_all_benchmarks(
            benchmarks_dir="benchmarks",
            baselines_dir="baselines",
            max_per_benchmark=20,
        )
    else:
        # Verify baselines (default)
        verify_all_benchmarks(
            benchmarks_dir="benchmarks",
            baselines_dir="baselines",
            results_dir="results",
            max_per_benchmark=20,
        )
