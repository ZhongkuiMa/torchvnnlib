"""VNNComp Benchmark Test Runner for TorchVNNLib.

Runs conversion on all VNNComp benchmark vnnlib files and validates results.
"""

import os
import sys
import time

# Add parent directory to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from torchvnnlib import TorchVNNLIB
from utils import (
    find_benchmarks_folders,
    find_all_vnnlib_files,
)

if __name__ == "__main__":
    dir_name = "benchmarks"

    benchmark_dirs = find_benchmarks_folders(dir_name)
    print(f"Collect {len(benchmark_dirs)} benchmark directories.")
    vnnlib_paths = find_all_vnnlib_files(benchmark_dirs)
    print(f"Collect {len(vnnlib_paths)} vnnlib files.")

    # Uncomment to test specific files
    # vnnlib_paths = [
    #     "benchmarks/acasxu/prop_6.vnnlib"
    # ]

    failed_vnnlib_paths = []
    verbose = False

    success_count = 0
    total_count = 0
    converter = TorchVNNLIB()

    for i, vnnlib_path in enumerate(vnnlib_paths):
        print(f"[{i}/{len(vnnlib_paths)}] ", end="")
        time_start = time.perf_counter()

        try:
            success = False
            # Convert to temporary output folder (will be cleaned up)
            output_folder = f"_temp_test_output_{i}"
            converter.convert(vnnlib_path, target_folder_path=output_folder)

            # Clean up temporary output
            import shutil
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)

            success = True
            success_count += 1
        except Exception as e:
            failed_vnnlib_paths.append(vnnlib_path)
            print(f"Failed to convert: {e}")
            # raise e  # Uncomment to stop on first error

        print(
            f"{'Success!' if success else 'Failure.'} "
            f"({time.perf_counter() - time_start:.2f}s) "
            f"for {os.path.basename(vnnlib_path)}"
        )
        total_count += 1

    if total_count > success_count:
        print(
            f"{len(failed_vnnlib_paths)} failed files:\n" + "\n".join(failed_vnnlib_paths)
        )

    print(f"\nSuccessfully converted {success_count}/{total_count} vnnlib files.")
