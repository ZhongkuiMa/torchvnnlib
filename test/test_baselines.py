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

    Runs TorchVNNLib.convert() and saves output to baselines directory.

    :param vnnlib_path: Path to vnnlib file
    :param baselines_dir: Root directory to store baseline folders
    :return: Path to baseline folder
    """
    converter = TorchVNNLIB()
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
        f"[{benchmark_name}] Saved {os.path.basename(vnnlib_path)} ({pth_count} .pth files)"
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
        print(f"No baseline: {os.path.basename(vnnlib_path)}")
        return False

    # Remove existing results if they exist
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    # Run conversion
    converter = TorchVNNLIB()
    converter.convert(vnnlib_path, target_folder_path=results_path)

    # Compare folders
    success, mismatches = compare_output_folders(baseline_path, results_path)

    if success:
        print(f"OK: {os.path.basename(vnnlib_path)}")
        return True

    print(f"MISMATCH: {os.path.basename(vnnlib_path)}")
    for mismatch in mismatches:
        print(f"  {mismatch}")

    return False


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
    print(f"Creating baselines for {len(vnnlib_files)} vnnlib files")

    success = 0
    failed = []
    start_time = time.perf_counter()

    for i, vnnlib_path in enumerate(vnnlib_files, 1):
        print(f"[{i}/{len(vnnlib_files)}] ", end="")
        try:
            update_baseline(vnnlib_path, baselines_dir)
            success += 1
        except Exception as e:
            print(f"Error: {e}")
            failed.append(vnnlib_path)

    total_time = time.perf_counter() - start_time

    print(f"\nCompleted: {success}/{len(vnnlib_files)} success, {len(failed)} failed")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {os.path.basename(f)}")
    print(
        f"Total time: {total_time:.2f}s (avg {total_time/len(vnnlib_files):.2f}s/file)"
    )


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
    print(f"Verifying {len(vnnlib_files)} vnnlib files")

    passed = 0
    failed = []
    missing = []
    start_time = time.perf_counter()

    for i, vnnlib_path in enumerate(vnnlib_files, 1):
        print(f"[{i}/{len(vnnlib_files)}] ", end="")
        try:
            baseline_path = get_baseline_path(vnnlib_path, baselines_dir)
            if not os.path.exists(baseline_path):
                print(f"Skip {os.path.basename(vnnlib_path)} - no baseline")
                missing.append(vnnlib_path)
                continue

            if compare_baseline(vnnlib_path, baselines_dir, results_dir):
                passed += 1
            else:
                failed.append(vnnlib_path)
        except Exception as e:
            print(f"Error: {e}")
            failed.append(vnnlib_path)

    total_time = time.perf_counter() - start_time
    tested = len(vnnlib_files) - len(missing)

    print(f"\nTested: {tested}/{len(vnnlib_files)}")
    print(f"Passed: {passed}/{tested}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  {os.path.basename(f)}")
    if missing:
        print(f"Missing baselines: {len(missing)}")
    print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    # Example 1: Update baseline for one vnnlib file
    # vnnlib_file = "benchmarks/vggnet16_2023/spec0_briard.vnnlib"
    # update_baseline(vnnlib_file, baselines_dir="baselines")

    # Example 2: Compare one vnnlib file against baseline
    # success = compare_baseline(
    #     vnnlib_file, baselines_dir="baselines", results_dir="results"
    # )

    # Example 3: Batch update baselines for all benchmarks
    # update_all_benchmarks(
    #     benchmarks_dir="benchmarks",
    #     baselines_dir="baselines",
    #     max_per_benchmark=20,
    # )

    # Example 4: Batch verify all benchmarks against their baselines
    verify_all_benchmarks(
        benchmarks_dir="benchmarks",
        baselines_dir="baselines",
        results_dir="results",
        max_per_benchmark=20,
    )
