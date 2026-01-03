"""Test correctness by comparing type-based processors with AST.

This module verifies that type-based optimized processors produce
identical results to the general AST-based approach.
Supports testing both PyTorch and NumPy backends.

Usage::

    # Test all files with both backends
    pytest test_correctness.py -v

    # Test with specific backend
    pytest test_correctness.py -v --backend torch

    # Test sample of files
    pytest test_correctness.py -v --sample-size 10

    # Backward compatible script execution
    python test_correctness.py --backend torch --sample-size 100
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from torchvnnlib import TorchVNNLIB
from torchvnnlib.tests.utils import find_all_vnnlib_files, find_benchmarks_folders


def load_data(file_path: str, backend: str):
    """Load data from either .pth or .npz file.

    :param file_path: Path to data file
    :param backend: Backend type ('torch' or 'numpy')
    :return: Loaded data dictionary
    """
    if backend == "torch":
        return torch.load(file_path, weights_only=False)
    npz_data = np.load(file_path, allow_pickle=True)
    return {
        "input": npz_data["input"],
        "output": npz_data["output"].tolist(),
    }


def arrays_equal(arr1, arr2):
    """Check if two arrays are equal (works for both torch and numpy).

    :param arr1: First array
    :param arr2: Second array
    :return: True if arrays are exactly equal
    """
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        return np.array_equal(arr1, arr2)
    if TORCH_AVAILABLE and isinstance(arr1, torch.Tensor):
        if isinstance(arr2, np.ndarray):
            arr2 = torch.from_numpy(arr2)
        return torch.equal(arr1, arr2)
    return np.array_equal(np.array(arr1), np.array(arr2))


def compute_max_diff(arr1, arr2):
    """Compute maximum absolute difference between two arrays.

    :param arr1: First array
    :param arr2: Second array
    :return: Maximum absolute difference
    """
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        return np.abs(arr1 - arr2).max()
    if TORCH_AVAILABLE and isinstance(arr1, torch.Tensor):
        if isinstance(arr2, np.ndarray):
            arr2 = torch.from_numpy(arr2)
        return (arr1 - arr2).abs().max().item()
    return np.abs(np.array(arr1) - np.array(arr2)).max()


def _get_data_files(base_dir: str, file_ext: str) -> list[str]:
    """Get all data files from directory.

    :param base_dir: Base directory to search
    :param file_ext: File extension to match
    :return: List of relative file paths
    """
    base_path = Path(base_dir)
    files = []
    for file_path in base_path.rglob(f"*{file_ext}"):
        rel_path = str(file_path.relative_to(base_path))
        files.append(rel_path)
    return files


def _compare_file_data(type_file: str, type_data: dict, ast_data: dict) -> list[str]:
    """Compare type-based and AST data for a single file.

    :param type_file: File identifier for error messages
    :param type_data: Type-based conversion data
    :param ast_data: AST conversion data
    :return: List of mismatches found
    """
    mismatches = []

    # Compare inputs
    if not arrays_equal(type_data["input"], ast_data["input"]):
        max_diff = compute_max_diff(type_data["input"], ast_data["input"])
        mismatches.append(f"{type_file}: Input mismatch, max_diff={max_diff:.6e}")

    # Compare outputs
    if len(type_data["output"]) != len(ast_data["output"]):
        type_count = len(type_data["output"])
        ast_count = len(ast_data["output"])
        mismatches.append(f"{type_file}: Output count, Type={type_count}, AST={ast_count}")
    else:
        for i, (type_out, ast_out) in enumerate(
            zip(type_data["output"], ast_data["output"], strict=False)
        ):
            if not arrays_equal(type_out, ast_out):
                max_diff = compute_max_diff(type_out, ast_out)
                mismatches.append(f"{type_file}: Output[{i}] mismatch, max_diff={max_diff:.6e}")

    return mismatches


def compare_results(
    type_path: str, ast_path: str, backend: str, verbose: bool = False
) -> tuple[bool, list[str]]:
    """Compare type-based results with AST results.

    :param type_path: Path to type-based conversion results
    :param ast_path: Path to AST conversion results
    :param backend: Backend type ('torch' or 'numpy')
    :param verbose: Print detailed comparison info
    :return: Tuple of (all_match, mismatches)
    """
    file_ext = ".pth" if backend == "torch" else ".npz"
    type_files = _get_data_files(type_path, file_ext)
    ast_files = _get_data_files(ast_path, file_ext)
    type_files.sort()
    ast_files.sort()

    mismatches = []

    # Check file counts match
    if len(type_files) != len(ast_files):
        mismatches.append(f"File count: Type={len(type_files)}, AST={len(ast_files)}")
        return False, mismatches

    # Compare each file
    type_base = Path(type_path)
    ast_base = Path(ast_path)
    for type_file, ast_file in zip(type_files, ast_files, strict=False):
        if type_file != ast_file:
            mismatches.append(f"File name mismatch: {type_file} vs {ast_file}")
            continue

        type_data = load_data(str(type_base / type_file), backend)
        ast_data = load_data(str(ast_base / ast_file), backend)
        mismatches.extend(_compare_file_data(type_file, type_data, ast_data))

    if verbose and not mismatches:
        print(f"  All {len(type_files)} properties match exactly")

    return len(mismatches) == 0, mismatches


def get_all_vnnlib_files(sample_size=None):
    """Collect all vnnlib files for parametrized testing.

    :param sample_size: Number of files to sample (None for all)
    :return: List of vnnlib file paths
    """
    test_dir = Path(__file__).parent
    # benchmarks dir is at tests/test_benchmarks/benchmarks
    benchmarks_dir = test_dir / "benchmarks"

    if not benchmarks_dir.exists():
        return []

    from torchvnnlib.tests.utils import find_all_vnnlib_files, find_benchmarks_folders

    benchmark_folders = find_benchmarks_folders(str(benchmarks_dir))
    vnnlib_files = find_all_vnnlib_files(benchmark_folders)

    if sample_size:
        import random

        random.seed(42)
        vnnlib_files = random.sample(vnnlib_files, min(sample_size, len(vnnlib_files)))

    return [str(f) for f in vnnlib_files]


@pytest.mark.benchmark
@pytest.mark.parametrize("vnnlib_file", get_all_vnnlib_files())
def test_file_correctness_parametrized(vnnlib_file, backend, torch_available):
    """Test correctness for a single VNN-LIB file and backend.

    Compares type-based (optimized) results with AST-based (known-correct) results.

    :param vnnlib_file: Path to VNN-LIB file
    :param backend: Backend to use ('torch' or 'numpy')
    :param torch_available: Fixture indicating if torch is available
    """
    if backend == "torch" and not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    basename = Path(vnnlib_file).stem
    type_path = f"/tmp/test_correctness_{basename}_type_{backend}"
    ast_path = f"/tmp/test_correctness_{basename}_ast_{backend}"

    # Convert with type-based
    try:
        converter_type = TorchVNNLIB(verbose=False, detect_fast_type=True, output_format=backend)
        converter_type.convert(vnnlib_file, target_folder_path=type_path)
    except (OSError, RuntimeError, ValueError) as e:
        pytest.fail(f"Type-based conversion failed: {e}")

    # Convert with AST
    try:
        converter_ast = TorchVNNLIB(verbose=False, detect_fast_type=False, output_format=backend)
        converter_ast.convert(vnnlib_file, target_folder_path=ast_path)
    except (OSError, RuntimeError, ValueError) as e:
        pytest.fail(f"AST conversion failed: {e}")

    # Compare results
    all_match, mismatches = compare_results(type_path, ast_path, backend, verbose=False)

    # Cleanup
    type_path_obj = Path(type_path)
    ast_path_obj = Path(ast_path)

    if type_path_obj.exists():
        shutil.rmtree(type_path_obj)
    if ast_path_obj.exists():
        shutil.rmtree(ast_path_obj)

    # Assert results match
    if not all_match:
        mismatch_str = "\n  ".join(mismatches[:3])
        pytest.fail(f"Type-based vs AST mismatch:\n  {mismatch_str}")


def _convert_vnnlib(
    vnnlib_path: str, output_path: str, backend: str, detect_fast: bool, verbose: bool
) -> bool:
    """Convert a VNNLIB file with specified options.

    :param vnnlib_path: Path to VNNLIB file
    :param output_path: Output directory path
    :param backend: Backend to use
    :param detect_fast: Whether to detect fast type
    :param verbose: Print verbose output
    :return: True if conversion succeeded, False otherwise
    """
    label = "Type-based" if detect_fast else "AST"
    try:
        converter = TorchVNNLIB(verbose=False, detect_fast_type=detect_fast, output_format=backend)
        converter.convert(vnnlib_path, target_folder_path=output_path)
        if verbose:
            print(f"  {label}: OK")
        return True
    except (OSError, RuntimeError, ValueError) as e:
        print(f"  {label} FAILED: {e}")
        return False


def _check_correctness_results(all_match: bool, mismatches: list[str], verbose: bool) -> None:
    """Check and print correctness results.

    :param all_match: Whether all results match
    :param mismatches: List of mismatches found
    :param verbose: Print detailed info
    """
    if not all_match:
        if verbose:
            print("  MISMATCH:")
            for mismatch in mismatches:
                print(f"    {mismatch}")
    elif verbose:
        print("  MATCH: Type-based and AST produce identical results")


def check_file_correctness(vnnlib_path: str, backend: str = "torch", verbose: bool = False) -> bool:
    """Test correctness for a single VNN-LIB file.

    :param vnnlib_path: Path to VNN-LIB file
    :param backend: Backend to use ('torch' or 'numpy')
    :param verbose: Print detailed info
    :return: True if results match, False otherwise
    """
    basename = Path(vnnlib_path).stem
    type_path = f"/tmp/test_correctness_{basename}_type_{backend}"
    ast_path = f"/tmp/test_correctness_{basename}_ast_{backend}"

    if verbose:
        print(f"\nTesting: {vnnlib_path} (backend: {backend})")
        print("=" * 70)

    # Convert with type-based
    if not _convert_vnnlib(vnnlib_path, type_path, backend, detect_fast=True, verbose=verbose):
        return False

    # Convert with AST
    if not _convert_vnnlib(vnnlib_path, ast_path, backend, detect_fast=False, verbose=verbose):
        return False

    # Compare results
    all_match, mismatches = compare_results(type_path, ast_path, backend, verbose=verbose)

    # Cleanup
    type_path_obj = Path(type_path)
    ast_path_obj = Path(ast_path)
    if type_path_obj.exists():
        shutil.rmtree(type_path_obj)
    if ast_path_obj.exists():
        shutil.rmtree(ast_path_obj)

    _check_correctness_results(all_match, mismatches, verbose)
    return all_match


def _sample_vnnlib_files(vnnlib_files: list[str], sample_size: int | None) -> list[str]:
    """Sample vnnlib files if sample_size is specified.

    :param vnnlib_files: List of vnnlib file paths
    :param sample_size: Number of files to sample (None for all)
    :return: Sampled list of vnnlib files
    """
    if not sample_size:
        return vnnlib_files
    import random

    random.seed(42)
    return random.sample(vnnlib_files, min(sample_size, len(vnnlib_files)))


def _test_file_and_report(
    i: int, total: int, vnnlib_path: str, backend: str, verbose: bool
) -> tuple[bool, bool]:
    """Test a single file and report results.

    :param i: Current file index (1-based)
    :param total: Total number of files
    :param vnnlib_path: Path to vnnlib file
    :param backend: Backend to use
    :param verbose: Print verbose output
    :return: Tuple of (passed, file_failed) where passed=True if test passed
    """
    file_start = time.perf_counter()

    if not verbose:
        print(f"[{i}/{total}] ", end="", flush=True)

    try:
        if check_file_correctness(vnnlib_path, backend=backend, verbose=verbose):
            elapsed = time.perf_counter() - file_start
            if not verbose:
                print(f"OK ({elapsed:.2f}s) - {Path(vnnlib_path).name}")
            return True, False

        elapsed = time.perf_counter() - file_start
        if not verbose:
            print(f"MISMATCH ({elapsed:.2f}s) - {Path(vnnlib_path).name}")
        return False, True
    except (OSError, RuntimeError, ValueError) as e:
        elapsed = time.perf_counter() - file_start
        if not verbose:
            print(f"ERROR ({elapsed:.2f}s) - {Path(vnnlib_path).name}: {e}")
        return False, True


def _print_correctness_summary(
    passed: int, failed: int, total: int, elapsed: float, failed_files: list[str], backend: str
) -> None:
    """Print correctness test summary.

    :param passed: Number of passed tests
    :param failed: Number of failed tests
    :param total: Total number of tests
    :param elapsed: Total elapsed time
    :param failed_files: List of failed file paths
    :param backend: Backend used
    """
    print("\n" + "=" * 70)
    print(f"CORRECTNESS TEST SUMMARY ({backend} backend)")
    print("=" * 70)
    print(f"Total files: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average time: {elapsed / max(total, 1):.4f}s per file")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  {f}")


def check_all_benchmarks(
    benchmarks_dir: str = "benchmarks",
    sample_size: int | None = None,
    backend: str = "torch",
    verbose: bool = False,
):
    """Test correctness for all benchmark files.

    :param benchmarks_dir: Directory containing benchmarks
    :param sample_size: Number of files to sample (None for all)
    :param backend: Backend to use ('torch' or 'numpy')
    :param verbose: Print detailed info
    :return: True if all tests passed, False otherwise
    """
    benchmark_folders = find_benchmarks_folders(benchmarks_dir)
    vnnlib_files = find_all_vnnlib_files(benchmark_folders)
    vnnlib_files = _sample_vnnlib_files(vnnlib_files, sample_size)

    print(f"Testing correctness of type-based processors vs AST ({backend} backend)")
    print(f"Total files: {len(vnnlib_files)}")
    print("=" * 70)

    passed = 0
    failed = 0
    failed_files = []

    start_time = time.perf_counter()

    for i, vnnlib_path in enumerate(vnnlib_files, 1):
        test_passed, file_failed = _test_file_and_report(
            i, len(vnnlib_files), vnnlib_path, backend, verbose
        )
        if test_passed:
            passed += 1
        else:
            failed += 1
            if file_failed:
                failed_files.append(vnnlib_path)

    elapsed = time.perf_counter() - start_time
    _print_correctness_summary(passed, failed, len(vnnlib_files), elapsed, failed_files, backend)

    return failed == 0


def main():
    """Backward compatible entry point supporting pytest execution."""
    parser = argparse.ArgumentParser(
        description="Test TorchVNNLib correctness with different backends"
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "numpy", "both"],
        default="torch",
        help="Backend to test (torch, numpy, or both)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of files to sample (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed test information")
    parser.add_argument("vnnlib_file", nargs="?", help="Test specific VNN-LIB file")

    args = parser.parse_args()

    if args.backend == "torch" and not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available, cannot test torch backend")
        sys.exit(1)

    # Build pytest arguments
    pytest_args = [__file__, "-v"]

    if args.verbose:
        pytest_args.append("-s")

    if args.backend != "both":
        pytest_args.append(f"--backend={args.backend}")

    if args.sample_size:
        pytest_args.append(f"--sample-size={args.sample_size}")

    if args.vnnlib_file:
        # Filter to specific file
        basename = Path(args.vnnlib_file).stem
        pytest_args.append(f"-k {basename}")

    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
