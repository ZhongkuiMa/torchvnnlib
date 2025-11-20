"""Script to collect VNN-LIB files from VNNComp 2024 benchmark directories.

This script copies VNN-LIB files from the VNNComp benchmarks directory into
a local benchmarks/ folder, organizing them into subdirectories by benchmark name.
"""

import os

from utils import find_benchmarks_folders, find_all_vnnlib_files


def copy_benchmarks(
    source_base_dir: str = "../../../vnncomp2024_benchmarks/benchmarks",
    target_dir: str = "benchmarks",
    max_per_benchmark: int = 20,
):
    """Copy VNN-LIB files from VNNComp benchmarks to local folder.

    Args:
        source_base_dir: Path to VNNComp benchmarks directory
        target_dir: Local directory to copy files to
        max_per_benchmark: Maximum VNN-LIB files per benchmark
    """
    if not os.path.exists(source_base_dir):
        print(f"Error: Source directory not found: {source_base_dir}")
        print("Please ensure vnncomp2024_benchmarks is available")
        return

    # Find all benchmark directories in source
    source_benchmark_dirs = find_benchmarks_folders(source_base_dir)
    print(f"Found {len(source_benchmark_dirs)} benchmark directories")

    # Create target benchmarks folder
    os.makedirs(target_dir, exist_ok=True)

    # Create subdirectories for each benchmark
    for bdir in source_benchmark_dirs:
        benchmark_name = os.path.basename(bdir)
        os.makedirs(os.path.join(target_dir, benchmark_name), exist_ok=True)

    # Copy VNN-LIB files
    total_copied = 0
    for source_bdir in source_benchmark_dirs:
        benchmark_name = os.path.basename(source_bdir)
        target_bdir = os.path.join(target_dir, benchmark_name)

        # Find VNN-LIB files in this benchmark
        vnnlib_files = find_all_vnnlib_files([source_bdir], num_limit=max_per_benchmark)

        if not vnnlib_files:
            print(f"[{benchmark_name}] No VNN-LIB files found")
            continue

        # Copy each file
        copied = 0
        for vnnlib_path in vnnlib_files:
            target_path = os.path.join(target_bdir, os.path.basename(vnnlib_path))
            with open(vnnlib_path, "rb") as src_file:
                with open(target_path, "wb") as dst_file:
                    dst_file.write(src_file.read())
            copied += 1
            total_copied += 1

        print(f"[{benchmark_name}] Copied {copied} VNN-LIB files")

    print(f"\nTotal: Copied {total_copied} VNN-LIB files to {target_dir}/")


if __name__ == "__main__":
    # Change to test directory
    os.chdir(os.path.dirname(__file__))
    copy_benchmarks()
