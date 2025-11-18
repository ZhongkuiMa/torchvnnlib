#!/usr/bin/env python3
"""Quick performance test for optimizations."""

import os
import sys
import time
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from torchvnnlib import TorchVNNLIB
from utils import find_benchmarks_folders, find_all_vnnlib_files

def test_performance():
    """Test performance on a sample of files."""
    # Test a few files from each type
    benchmark_dirs = find_benchmarks_folders("benchmarks")
    vnnlib_files = find_all_vnnlib_files(benchmark_dirs, num_limit=5)

    print(f"Testing {len(vnnlib_files)} files...")
    print("=" * 70)

    converter = TorchVNNLIB()
    total_time = 0

    for i, vnnlib_path in enumerate(vnnlib_files, 1):
        start = time.perf_counter()
        try:
            converter.convert(vnnlib_path, target_folder_path=f"_temp_perf_test_{i}")
            elapsed = time.perf_counter() - start
            total_time += elapsed
            print(
                f"[{i}/{len(vnnlib_files)}] OK ({elapsed:.4f}s) - {os.path.basename(vnnlib_path)}"
            )

            # Cleanup
            if os.path.exists(f"_temp_perf_test_{i}"):
                shutil.rmtree(f"_temp_perf_test_{i}")
        except Exception as e:
            print(f"[{i}/{len(vnnlib_files)}] ERROR - {os.path.basename(vnnlib_path)}: {e}")

    print("=" * 70)
    print(f"Total time: {total_time:.4f}s")
    print(f"Average time: {total_time/len(vnnlib_files):.4f}s per file")


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    test_performance()
