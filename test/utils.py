"""Utility functions for TorchVNNLib testing.

Extracted helper functions used across multiple test files.
"""

import os
import shutil
import torch


def find_benchmarks_folders(base_dir):
    """Find all benchmark directories in base_dir.

    :param base_dir: Root directory containing benchmark subdirectories
    :return: List of benchmark directory paths
    """
    benchmark_dirs = []
    # Only consider first-level subdirectories
    for entry in os.listdir(base_dir):
        subdir = os.path.normpath(os.path.join(base_dir, entry))
        if os.path.isdir(subdir):
            benchmark_dirs.append(subdir)
    return benchmark_dirs


def find_vnnlib_folders(benchmark_dirs):
    """Find vnnlib subdirectories in benchmark directories.

    :param benchmark_dirs: List of benchmark directory paths
    :return: List of vnnlib subdirectory paths
    """
    vnnlib_dirs = []
    for bdir in benchmark_dirs:
        vnnlib_subdir = os.path.join(bdir, "vnnlib")
        if os.path.isdir(vnnlib_subdir):
            vnnlib_dirs.append(vnnlib_subdir)
    return vnnlib_dirs


def find_all_vnnlib_files(benchmark_dirs, num_limit: int = 20):
    """Find all vnnlib files in benchmark directories.

    Works with both structures:
    - benchmarks/benchmark_name/*.vnnlib (new structure)
    - benchmarks/benchmark_name/vnnlib/*.vnnlib (old structure)

    :param benchmark_dirs: List of benchmark directory paths
    :param num_limit: Maximum vnnlib files per benchmark directory
    :return: List of vnnlib file paths
    """
    vnnlib_files = []
    for bdir in benchmark_dirs:
        # Try new structure first (vnnlib files directly in benchmark dir)
        i = 0
        for entry in os.listdir(bdir):
            if entry.endswith(".vnnlib"):
                vnnlib_path = os.path.normpath(os.path.join(bdir, entry))
                vnnlib_files.append(vnnlib_path)
                i += 1
                if i >= num_limit:
                    break

        # If no vnnlib files found, try old structure (vnnlib subdirectory)
        if i == 0:
            vnnlib_subdir = os.path.join(bdir, "vnnlib")
            if os.path.isdir(vnnlib_subdir):
                for entry in os.listdir(vnnlib_subdir):
                    if entry.endswith(".vnnlib"):
                        vnnlib_path = os.path.normpath(os.path.join(vnnlib_subdir, entry))
                        vnnlib_files.append(vnnlib_path)
                        i += 1
                        if i >= num_limit:
                            break
    return vnnlib_files


def get_benchmark_name(vnnlib_path: str, benchmarks_dir: str = "benchmarks") -> str:
    """Extract benchmark name from vnnlib file path.

    :param vnnlib_path: Path to vnnlib file
    :param benchmarks_dir: Root benchmarks directory name
    :return: Benchmark name (subdirectory name)
    """
    # Normalize path
    vnnlib_path = os.path.normpath(vnnlib_path)

    # Find benchmarks_dir in the path
    path_parts = vnnlib_path.split(os.sep)

    # Find the index of benchmarks_dir
    try:
        bench_idx = path_parts.index(benchmarks_dir)
        # The next part is the benchmark name
        if bench_idx + 1 < len(path_parts):
            return path_parts[bench_idx + 1]
    except ValueError:
        pass

    # Fallback: use the parent directory name
    return os.path.basename(os.path.dirname(vnnlib_path))


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Compare two PyTorch tensors with tolerance.

    :param tensor1: First tensor
    :param tensor2: Second tensor
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :return: True if tensors are close, False otherwise
    """
    if tensor1.shape != tensor2.shape:
        return False
    return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)


def compare_pth_files(file1: str, file2: str, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Compare two .pth files containing tensor dictionaries.

    :param file1: Path to first .pth file
    :param file2: Path to second .pth file
    :param rtol: Relative tolerance for tensor comparison
    :param atol: Absolute tolerance for tensor comparison
    :return: True if files contain identical data, False otherwise
    """
    data1 = torch.load(file1, weights_only=True)
    data2 = torch.load(file2, weights_only=True)

    # Check keys match
    if set(data1.keys()) != set(data2.keys()):
        return False

    # Compare each value
    for key in data1.keys():
        val1 = data1[key]
        val2 = data2[key]

        # Handle tensor values
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if not compare_tensors(val1, val2, rtol, atol):
                return False
        # Handle list of tensors
        elif isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                return False
            for t1, t2 in zip(val1, val2):
                if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                    if not compare_tensors(t1, t2, rtol, atol):
                        return False
                else:
                    if t1 != t2:
                        return False
        # Direct comparison for other types
        else:
            if val1 != val2:
                return False

    return True


def compare_output_folders(folder1: str, folder2: str, rtol: float = 1e-5, atol: float = 1e-8) -> tuple[bool, list[str]]:
    """Compare two output folders recursively.

    :param folder1: Path to first folder
    :param folder2: Path to second folder
    :param rtol: Relative tolerance for tensor comparison
    :param atol: Absolute tolerance for tensor comparison
    :return: Tuple of (success, list of mismatches)
    """
    mismatches = []

    # Check both folders exist
    if not os.path.exists(folder1):
        mismatches.append(f"Missing folder: {folder1}")
        return False, mismatches
    if not os.path.exists(folder2):
        mismatches.append(f"Missing folder: {folder2}")
        return False, mismatches

    # Get all .pth files in both folders (recursively)
    def get_pth_files(folder):
        pth_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".pth"):
                    rel_path = os.path.relpath(os.path.join(root, file), folder)
                    pth_files.append(rel_path)
        return sorted(pth_files)

    files1 = get_pth_files(folder1)
    files2 = get_pth_files(folder2)

    # Check file lists match
    if set(files1) != set(files2):
        only_in_1 = set(files1) - set(files2)
        only_in_2 = set(files2) - set(files1)
        if only_in_1:
            mismatches.append(f"Only in {folder1}: {only_in_1}")
        if only_in_2:
            mismatches.append(f"Only in {folder2}: {only_in_2}")
        return False, mismatches

    # Compare each file
    for rel_path in files1:
        path1 = os.path.join(folder1, rel_path)
        path2 = os.path.join(folder2, rel_path)
        if not compare_pth_files(path1, path2, rtol, atol):
            mismatches.append(f"Mismatch: {rel_path}")

    return len(mismatches) == 0, mismatches
