"""Script to collect vnnlib files from VNNComp 2024 benchmark directories.

This script copies vnnlib files from the VNNComp benchmarks directory into
a local benchmarks/ folder, organizing them into subdirectories by benchmark name.
"""

import os

from utils import (
    find_benchmarks_folders,
    find_vnnlib_folders,
    find_all_vnnlib_files,
)

if __name__ == "__main__":
    dir_name = "../../../vnncomp2024_benchmarks/benchmarks"

    benchmark_dirs = find_benchmarks_folders(dir_name)
    print(f"Collect {len(benchmark_dirs)} benchmark directories.")
    vnnlib_dirs = find_vnnlib_folders(benchmark_dirs)
    print(f"Collect {len(vnnlib_dirs)} vnnlib directories.")

    # Create a folder in the current directory to store the collected vnnlib files
    dir_name = "benchmarks/"
    os.makedirs(dir_name, exist_ok=True)
    # Create subfolders for each benchmark
    for bdir in benchmark_dirs:
        benchmark_name = os.path.basename(bdir)
        os.makedirs(os.path.join(dir_name, benchmark_name), exist_ok=True)

    max_vnnlib_per_benchmark = 20
    i = 0
    # Copy the collected vnnlib files to the corresponding benchmark subfolders
    for vnnlib_dir in vnnlib_dirs:
        benchmark_name = os.path.basename(os.path.dirname(vnnlib_dir))
        target_dir = os.path.join(dir_name, benchmark_name)

        vnnlib_files = find_all_vnnlib_files([vnnlib_dir], num_limit=max_vnnlib_per_benchmark)
        for vnnlib_path in vnnlib_files:
            target_path = os.path.join(target_dir, os.path.basename(vnnlib_path))
            with open(vnnlib_path, "rb") as src_file:
                with open(target_path, "wb") as dst_file:
                    dst_file.write(src_file.read())
            i += 1

    print(f"Copied {i} vnnlib files to {dir_name}")
