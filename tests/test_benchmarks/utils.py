"""Utility functions for TorchVNNLib testing.

Extracted helper functions used across multiple test files.
"""

from pathlib import Path

import torch


def find_benchmarks_folders(base_dir):
    """Find all benchmark directories in base_dir.

    :param base_dir: Root directory containing benchmark subdirectories
    :return: List of benchmark directory paths
    """
    # Only consider first-level subdirectories
    base_path = Path(base_dir)
    return [str(subdir) for subdir in base_path.iterdir() if subdir.is_dir()]


def find_vnnlib_folders(benchmark_dirs):
    """Find vnnlib subdirectories in benchmark directories.

    :param benchmark_dirs: List of benchmark directory paths
    :return: List of vnnlib subdirectory paths
    """
    vnnlib_dirs = []
    for bdir in benchmark_dirs:
        vnnlib_subdir = Path(bdir) / "vnnlib"
        if vnnlib_subdir.is_dir():
            vnnlib_dirs.append(str(vnnlib_subdir))
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
        bdir_path = Path(bdir)
        for entry in bdir_path.glob("*.vnnlib"):
            vnnlib_files.append(str(entry))
            i += 1
            if i >= num_limit:
                break

        # If no vnnlib files found, try old structure (vnnlib subdirectory)
        if i == 0:
            vnnlib_subdir = bdir_path / "vnnlib"
            if vnnlib_subdir.is_dir():
                for entry in vnnlib_subdir.glob("*.vnnlib"):
                    vnnlib_files.append(str(entry))
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
    vnnlib_path_obj = Path(vnnlib_path)

    # Find benchmarks_dir in the path
    path_parts = vnnlib_path_obj.parts

    # Find the index of benchmarks_dir
    try:
        bench_idx = path_parts.index(benchmarks_dir)
        # The next part is the benchmark name
        if bench_idx + 1 < len(path_parts):
            return path_parts[bench_idx + 1]
    except ValueError:
        pass

    # Fallback: use the parent directory name
    return vnnlib_path_obj.parent.name


def compare_tensors(
    tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Compare two PyTorch tensors with tolerance.

    :param tensor1: First tensor
    :param tensor2: Second tensor
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :return: True if tensors are close, False otherwise
    """
    if tensor1.shape != tensor2.shape:
        return False
    return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)  # type: ignore[no-any-return]


def _compare_pth_values(val1, val2, rtol: float, atol: float) -> bool:
    """Compare two values from pth files.

    :param val1: First value
    :param val2: Second value
    :param rtol: Relative tolerance for tensors
    :param atol: Absolute tolerance for tensors
    :return: True if values match, False otherwise
    """
    # Handle tensor values
    if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
        return compare_tensors(val1, val2, rtol, atol)

    # Handle list of tensors
    if isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return False
        for t1, t2 in zip(val1, val2, strict=False):
            if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
                if not compare_tensors(t1, t2, rtol, atol):
                    return False
            else:
                if t1 != t2:
                    return False
        return True

    # Direct comparison for other types
    return val1 == val2  # type: ignore[no-any-return]


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
    return all(_compare_pth_values(data1[key], data2[key], rtol, atol) for key in data1)


def compare_output_folders(
    folder1: str, folder2: str, rtol: float = 1e-5, atol: float = 1e-8
) -> tuple[bool, list[str]]:
    """Compare two output folders recursively.

    :param folder1: Path to first folder
    :param folder2: Path to second folder
    :param rtol: Relative tolerance for tensor comparison
    :param atol: Absolute tolerance for tensor comparison
    :return: Tuple of (success, list of mismatches)
    """
    mismatches = []

    # Check both folders exist
    folder1_path = Path(folder1)
    folder2_path = Path(folder2)
    if not folder1_path.exists():
        mismatches.append(f"Missing folder: {folder1}")
        return False, mismatches
    if not folder2_path.exists():
        mismatches.append(f"Missing folder: {folder2}")
        return False, mismatches

    # Get all .pth files in both folders (recursively)
    def get_pth_files(folder_path):
        pth_files = []
        for pth_file in folder_path.rglob("*.pth"):
            rel_path = pth_file.relative_to(folder_path)
            pth_files.append(str(rel_path))
        return sorted(pth_files)

    files1 = get_pth_files(folder1_path)
    files2 = get_pth_files(folder2_path)

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
        path1 = str(folder1_path / rel_path)
        path2 = str(folder2_path / rel_path)
        if not compare_pth_files(path1, path2, rtol, atol):
            mismatches.append(f"Mismatch: {rel_path}")

    return len(mismatches) == 0, mismatches
