#!/usr/bin/env python3
"""Test correctness by comparing type-based processors with AST.

This script verifies that type-based optimized processors produce
identical results to the general AST-based approach.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from torchvnnlib import TorchVNNLIB
from utils import find_all_vnnlib_files, find_benchmarks_folders


def compare_results(
    type_path: str, ast_path: str, verbose: bool = False
) -> tuple[bool, list[str]]:
    """Compare type-based results with AST results.

    Args:
        type_path: Path to type-based conversion results
        ast_path: Path to AST conversion results
        verbose: Print detailed comparison info

    Returns:
        Tuple of (all_match, mismatches)
        all_match: True if results match exactly
        mismatches: List of mismatch descriptions
    """
    mismatches = []

    # Get all property files
    type_files = []
    for root, dirs, files in os.walk(type_path):
        for f in files:
            if f.endswith(".pth"):
                rel_path = os.path.relpath(os.path.join(root, f), type_path)
                type_files.append(rel_path)

    ast_files = []
    for root, dirs, files in os.walk(ast_path):
        for f in files:
            if f.endswith(".pth"):
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

        type_data = torch.load(os.path.join(type_path, type_file), weights_only=False)
        ast_data = torch.load(os.path.join(ast_path, ast_file), weights_only=False)

        # Compare inputs
        if not torch.equal(type_data["input"], ast_data["input"]):
            max_diff = (type_data["input"] - ast_data["input"]).abs().max()
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
                if not torch.equal(type_out, ast_out):
                    max_diff = (type_out - ast_out).abs().max()
                    mismatches.append(
                        f"{type_file}: Output[{i}] mismatch, max_diff={max_diff:.6e}"
                    )

    if verbose and not mismatches:
        print(f"  All {len(type_files)} properties match exactly")

    return len(mismatches) == 0, mismatches


def test_file_correctness(vnnlib_path: str, verbose: bool = False) -> bool:
    """Test correctness for a single VNN-LIB file.

    Args:
        vnnlib_path: Path to VNN-LIB file
        verbose: Print detailed info

    Returns:
        True if results match, False otherwise
    """
    basename = os.path.basename(vnnlib_path).replace(".vnnlib", "")
    type_path = f"/tmp/test_correctness_{basename}_type"
    ast_path = f"/tmp/test_correctness_{basename}_ast"

    if verbose:
        print(f"\nTesting: {vnnlib_path}")
        print("=" * 70)

    # Convert with type-based
    try:
        converter_type = TorchVNNLIB(verbose=False, detect_fast_type=True)
        converter_type.convert(vnnlib_path, target_folder_path=type_path)
        if verbose:
            print("  Type-based: OK")
    except Exception as e:
        print(f"  Type-based FAILED: {e}")
        return False

    # Convert with AST
    try:
        converter_ast = TorchVNNLIB(verbose=False, detect_fast_type=False)
        converter_ast.convert(vnnlib_path, target_folder_path=ast_path)
        if verbose:
            print("  AST: OK")
    except Exception as e:
        print(f"  AST FAILED: {e}")
        return False

    # Compare results
    all_match, mismatches = compare_results(type_path, ast_path, verbose=verbose)

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
    benchmarks_dir: str = "benchmarks", sample_size: int = None, verbose: bool = False
):
    """Test correctness for all benchmark files.

    Args:
        benchmarks_dir: Directory containing benchmarks
        sample_size: Number of files to sample (None for all)
        verbose: Print detailed info
    """
    os.chdir(os.path.dirname(__file__))

    # Get benchmark folders first
    benchmark_folders = find_benchmarks_folders(benchmarks_dir)
    vnnlib_files = find_all_vnnlib_files(benchmark_folders)

    if sample_size:
        import random

        random.seed(42)
        vnnlib_files = random.sample(vnnlib_files, min(sample_size, len(vnnlib_files)))

    print("Testing correctness of type-based processors vs AST")
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
            if test_file_correctness(vnnlib_path, verbose=verbose):
                passed += 1
                elapsed = time.perf_counter() - file_start
                if not verbose:
                    print(f"OK ({elapsed:.2f}s) - {os.path.basename(vnnlib_path)}")
            else:
                failed += 1
                failed_files.append(vnnlib_path)
                elapsed = time.perf_counter() - file_start
                if not verbose:
                    print(f"MISMATCH ({elapsed:.2f}s) - {os.path.basename(vnnlib_path)}")
        except Exception as e:
            failed += 1
            failed_files.append(vnnlib_path)
            elapsed = time.perf_counter() - file_start
            if not verbose:
                print(f"ERROR ({elapsed:.2f}s) - {os.path.basename(vnnlib_path)}: {e}")

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print("CORRECTNESS TEST SUMMARY")
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


if __name__ == "__main__":
    # Test all benchmarks with default settings
    # To test specific file: python test_correctness.py <vnnlib_file>
    # To test with sampling: modify sample_size parameter below

    if len(sys.argv) > 1:
        # Test specific file
        vnnlib_file = sys.argv[1]
        print(f"Testing single file: {vnnlib_file}")
        success = test_file_correctness(vnnlib_file, verbose=True)
    else:
        # Test all benchmarks (use sample_size=None for all, or set to number for sampling)
        success = test_all_benchmarks(sample_size=None, verbose=False)

    sys.exit(0 if success else 1)
