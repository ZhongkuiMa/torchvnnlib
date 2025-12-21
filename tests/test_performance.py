"""Quick performance test for optimizations.

Usage::

    # Run performance test
    pytest test_performance.py -v -s

    # Backward compatible script execution
    python test_performance.py
"""

import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from torchvnnlib import TorchVNNLIB
from utils import find_benchmarks_folders, find_all_vnnlib_files


def test_performance(test_dir, benchmarks_dir):
    """Test performance on a sample of files.

    :param test_dir: Test directory fixture
    :param benchmarks_dir: Benchmarks directory fixture
    """
    benchmark_dirs = find_benchmarks_folders(str(benchmarks_dir))
    vnnlib_files = find_all_vnnlib_files(benchmark_dirs, num_limit=5)

    if not vnnlib_files:
        pytest.skip("No benchmark files found")

    print(f"\nTesting {len(vnnlib_files)} files...")
    print("=" * 70)

    converter = TorchVNNLIB()
    total_time = 0
    success = 0
    failed = 0

    for i, vnnlib_path in enumerate(vnnlib_files, 1):
        start = time.perf_counter()
        try:
            converter.convert(vnnlib_path, target_folder_path=f"_temp_perf_test_{i}")
            elapsed = time.perf_counter() - start
            total_time += elapsed
            success += 1
            print(
                f"[{i}/{len(vnnlib_files)}] OK ({elapsed:.4f}s) - {os.path.basename(vnnlib_path)}"
            )

            # Cleanup
            if os.path.exists(f"_temp_perf_test_{i}"):
                shutil.rmtree(f"_temp_perf_test_{i}")
        except Exception as e:
            failed += 1
            print(
                f"[{i}/{len(vnnlib_files)}] ERROR - {os.path.basename(vnnlib_path)}: {e}"
            )

    print("=" * 70)
    print(f"Total time: {total_time:.4f}s")
    if success > 0:
        print(f"Average time: {total_time/success:.4f}s per file")
    print(f"Success: {success}/{len(vnnlib_files)}")

    assert success > 0, f"Performance test failed: {failed}/{len(vnnlib_files)} files failed"


def main():
    """Backward compatible entry point supporting pytest execution."""
    pytest_args = [__file__, "-v", "-s"]
    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
