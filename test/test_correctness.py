#!/usr/bin/env python3
"""Test correctness by comparing type-based processors with AST.

This script verifies that type-based optimized processors produce
identical results to the general AST-based approach.
Supports testing both PyTorch and NumPy backends.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from torchvnnlib import TorchVNNLIB
from utils import find_all_vnnlib_files, find_benchmarks_folders


def load_data(file_path: str, backend: str):
    """Load data from either .pth or .npz file.

    :param file_path: Path to data file
    :param backend: Backend type ('torch' or 'numpy')
    :return: Loaded data dictionary
    """
    if backend == "torch":
        return torch.load(file_path, weights_only=False)
    else:
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
    elif TORCH_AVAILABLE and isinstance(arr1, torch.Tensor):
        if isinstance(arr2, np.ndarray):
            arr2 = torch.from_numpy(arr2)
        return torch.equal(arr1, arr2)
    else:
        return np.array_equal(np.array(arr1), np.array(arr2))


def compute_max_diff(arr1, arr2):
    """Compute maximum absolute difference between two arrays.

    :param arr1: First array
    :param arr2: Second array
    :return: Maximum absolute difference
    """
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        return np.abs(arr1 - arr2).max()
    elif TORCH_AVAILABLE and isinstance(arr1, torch.Tensor):
        if isinstance(arr2, np.ndarray):
            arr2 = torch.from_numpy(arr2)
        return (arr1 - arr2).abs().max().item()
    else:
        return np.abs(np.array(arr1) - np.array(arr2)).max()


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
    mismatches = []
    file_ext = ".pth" if backend == "torch" else ".npz"

    # Get all property files
    type_files = []
    for root, dirs, files in os.walk(type_path):
        for f in files:
            if f.endswith(file_ext):
                rel_path = os.path.relpath(os.path.join(root, f), type_path)
                type_files.append(rel_path)

    ast_files = []
    for root, dirs, files in os.walk(ast_path):
        for f in files:
            if f.endswith(file_ext):
                rel_path = os.path.relpath(os.path.join(root, f), ast_path)
                ast_files.append(rel_path)

    # Check file counts match
    if len(type_files) != len(ast_files):
        mismatches.append(f"File count: Type={len(type_files)}, AST={len(ast_files)}")
        return False, mismatches

    type_files.sort()
    ast_files.sort()

    # Compare each file
    for type_file, ast_file in zip(type_files, ast_files):
        if type_file != ast_file:
            mismatches.append(f"File name mismatch: {type_file} vs {ast_file}")
            continue

        type_data = load_data(os.path.join(type_path, type_file), backend)
        ast_data = load_data(os.path.join(ast_path, ast_file), backend)

        # Compare inputs
        if not arrays_equal(type_data["input"], ast_data["input"]):
            max_diff = compute_max_diff(type_data["input"], ast_data["input"])
            mismatches.append(f"{type_file}: Input mismatch, max_diff={max_diff:.6e}")

        # Compare outputs
        if len(type_data["output"]) != len(ast_data["output"]):
            mismatches.append(
                f"{type_file}: Output count, Type={len(type_data['output'])}, AST={len(ast_data['output'])}"
            )
        else:
            for i, (type_out, ast_out) in enumerate(
                zip(type_data["output"], ast_data["output"])
            ):
                if not arrays_equal(type_out, ast_out):
                    max_diff = compute_max_diff(type_out, ast_out)
                    mismatches.append(
                        f"{type_file}: Output[{i}] mismatch, max_diff={max_diff:.6e}"
                    )

    if verbose and not mismatches:
        print(f"  All {len(type_files)} properties match exactly")

    return len(mismatches) == 0, mismatches


def test_file_correctness(
    vnnlib_path: str, backend: str = "torch", verbose: bool = False
) -> bool:
    """Test correctness for a single VNN-LIB file.

    :param vnnlib_path: Path to VNN-LIB file
    :param backend: Backend to use ('torch' or 'numpy')
    :param verbose: Print detailed info
    :return: True if results match, False otherwise
    """
    basename = os.path.basename(vnnlib_path).replace(".vnnlib", "")
    type_path = f"/tmp/test_correctness_{basename}_type_{backend}"
    ast_path = f"/tmp/test_correctness_{basename}_ast_{backend}"

    if verbose:
        print(f"\nTesting: {vnnlib_path} (backend: {backend})")
        print("=" * 70)

    # Convert with type-based
    try:
        converter_type = TorchVNNLIB(
            verbose=False, detect_fast_type=True, output_format=backend
        )
        converter_type.convert(vnnlib_path, target_folder_path=type_path)
        if verbose:
            print("  Type-based: OK")
    except Exception as e:
        print(f"  Type-based FAILED: {e}")
        return False

    # Convert with AST
    try:
        converter_ast = TorchVNNLIB(
            verbose=False, detect_fast_type=False, output_format=backend
        )
        converter_ast.convert(vnnlib_path, target_folder_path=ast_path)
        if verbose:
            print("  AST: OK")
    except Exception as e:
        print(f"  AST FAILED: {e}")
        return False

    # Compare results
    all_match, mismatches = compare_results(
        type_path, ast_path, backend, verbose=verbose
    )

    # Cleanup
    import shutil

    if os.path.exists(type_path):
        shutil.rmtree(type_path)
    if os.path.exists(ast_path):
        shutil.rmtree(ast_path)

    if not all_match:
        if verbose:
            print("  MISMATCH:")
            for mismatch in mismatches:
                print(f"    {mismatch}")
        return False

    if verbose:
        print("  MATCH: Type-based and AST produce identical results")

    return True


def test_all_benchmarks(
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
    os.chdir(os.path.dirname(__file__))

    # Get benchmark folders first
    benchmark_folders = find_benchmarks_folders(benchmarks_dir)
    vnnlib_files = find_all_vnnlib_files(benchmark_folders)

    if sample_size:
        import random

        random.seed(42)
        vnnlib_files = random.sample(vnnlib_files, min(sample_size, len(vnnlib_files)))

    print(f"Testing correctness of type-based processors vs AST ({backend} backend)")
    print(f"Total files: {len(vnnlib_files)}")
    print("=" * 70)

    passed = 0
    failed = 0
    failed_files = []

    start_time = time.perf_counter()

    for i, vnnlib_path in enumerate(vnnlib_files, 1):
        file_start = time.perf_counter()

        if not verbose:
            print(f"[{i}/{len(vnnlib_files)}] ", end="", flush=True)

        try:
            if test_file_correctness(vnnlib_path, backend=backend, verbose=verbose):
                passed += 1
                elapsed = time.perf_counter() - file_start
                if not verbose:
                    print(f"OK ({elapsed:.2f}s) - {os.path.basename(vnnlib_path)}")
            else:
                failed += 1
                failed_files.append(vnnlib_path)
                elapsed = time.perf_counter() - file_start
                if not verbose:
                    print(
                        f"MISMATCH ({elapsed:.2f}s) - {os.path.basename(vnnlib_path)}"
                    )
        except Exception as e:
            failed += 1
            failed_files.append(vnnlib_path)
            elapsed = time.perf_counter() - file_start
            if not verbose:
                print(f"ERROR ({elapsed:.2f}s) - {os.path.basename(vnnlib_path)}: {e}")

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print(f"CORRECTNESS TEST SUMMARY ({backend} backend)")
    print("=" * 70)
    print(f"Total files: {len(vnnlib_files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average time: {elapsed/len(vnnlib_files):.4f}s per file")

    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  {f}")

    return failed == 0


def main():
    """Main test runner."""
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
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed test information"
    )
    parser.add_argument("vnnlib_file", nargs="?", help="Test specific VNN-LIB file")

    args = parser.parse_args()

    if args.backend == "torch" and not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available, cannot test torch backend")
        sys.exit(1)

    try:
        if args.vnnlib_file:
            # Test specific file
            print(f"Testing single file: {args.vnnlib_file}")
            if args.backend == "both":
                print("\n" + "=" * 70)
                print("Testing with PyTorch backend")
                print("=" * 70)
                success_torch = test_file_correctness(
                    args.vnnlib_file, backend="torch", verbose=args.verbose
                )

                print("\n" + "=" * 70)
                print("Testing with NumPy backend")
                print("=" * 70)
                success_numpy = test_file_correctness(
                    args.vnnlib_file, backend="numpy", verbose=args.verbose
                )

                success = success_torch and success_numpy
            else:
                success = test_file_correctness(
                    args.vnnlib_file, backend=args.backend, verbose=args.verbose
                )
        else:
            # Test all benchmarks
            if args.backend == "both":
                print("Testing with PyTorch backend first")
                success_torch = test_all_benchmarks(
                    sample_size=args.sample_size,
                    backend="torch",
                    verbose=args.verbose,
                )

                print("\n" + "=" * 70)
                print("Testing with NumPy backend")
                print("=" * 70)
                success_numpy = test_all_benchmarks(
                    sample_size=args.sample_size,
                    backend="numpy",
                    verbose=args.verbose,
                )

                success = success_torch and success_numpy
            else:
                success = test_all_benchmarks(
                    sample_size=args.sample_size,
                    backend=args.backend,
                    verbose=args.verbose,
                )

        if success:
            print("\n" + "=" * 70)
            print("SUCCESS: All tests passed")
            print("=" * 70)
            sys.exit(0)
        else:
            print("\n" + "=" * 70)
            print("FAILURE: Some tests failed")
            print("=" * 70)
            sys.exit(1)

    except Exception as e:
        print(f"\nERROR: Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
