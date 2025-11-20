"""Quickly classify all VNN-LIB files without full conversion.

This script only performs type detection, not conversion, so it's very fast.
"""

import os
import time
from collections import defaultdict
from pathlib import Path

from torchvnnlib.torchvnnlib.ast._preprocess import preprocess_vnnlib
from torchvnnlib.torchvnnlib.fast_type._fast_type_detect import fast_detect_type


def classify_file(vnnlib_path):
    """Classify a single VNN-LIB file."""
    try:
        with open(vnnlib_path) as f:
            lines = f.readlines()

        # Preprocess
        processed_lines, n_inputs, n_outputs = preprocess_vnnlib(lines)

        # Detect type
        vnnlib_type = fast_detect_type(processed_lines, verbose=False)

        return vnnlib_type, None

    except Exception as e:
        return None, str(e)


def classify_benchmark_dir(benchmark_dir):
    """Classify all VNN-LIB files in a benchmark directory."""
    vnnlib_files = sorted(Path(benchmark_dir).glob("**/*.vnnlib"))

    if not vnnlib_files:
        return {}

    results = {}
    for vnnlib_path in vnnlib_files:
        vnnlib_type, error = classify_file(str(vnnlib_path))

        results[vnnlib_path.name] = {
            "type": vnnlib_type,
            "error": error,
            "path": str(vnnlib_path),
        }

    return results


def main():
    os.chdir(os.path.dirname(__file__))

    benchmarks_dir = Path("benchmarks")

    if not benchmarks_dir.exists():
        print(f"Error: benchmarks directory not found at {benchmarks_dir}")
        return

    # Get all benchmark subdirectories
    benchmark_dirs = sorted([d for d in benchmarks_dir.iterdir() if d.is_dir()])

    print("=" * 80)
    print("VNN-LIB Type Classification")
    print("=" * 80)
    print(f"Scanning {len(benchmark_dirs)} benchmark directories...")
    print()

    overall_stats = defaultdict(int)
    benchmark_results = {}

    start_time = time.perf_counter()

    for benchmark_dir in benchmark_dirs:
        benchmark_name = benchmark_dir.name
        results = classify_benchmark_dir(benchmark_dir)

        if not results:
            continue

        benchmark_results[benchmark_name] = results

        # Count types
        type_counts = defaultdict(int)
        errors = 0

        for filename, info in results.items():
            if info["error"]:
                errors += 1
            else:
                type_counts[info["type"].name] += 1
                overall_stats[info["type"].name] += 1

        # Print benchmark summary
        total = len(results)
        print(f"{benchmark_name}:")
        print(f"  Total files: {total}")

        if errors > 0:
            print(f"  Errors: {errors}")

        print(f"  Type distribution:")
        for type_name in sorted(type_counts.keys()):
            count = type_counts[type_name]
            pct = 100 * count / total
            print(f"    {type_name:10s}: {count:4d} files ({pct:5.1f}%)")
        print()

    elapsed = time.perf_counter() - start_time

    # Overall summary
    total_files = sum(overall_stats.values())

    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Classification completed in {elapsed:.2f}s")
    print(f"Total files classified: {total_files}")
    print()
    print("Overall type distribution:")
    for type_name in sorted(overall_stats.keys()):
        count = overall_stats[type_name]
        pct = 100 * count / total_files if total_files > 0 else 0
        print(f"  {type_name:10s}: {count:4d} files ({pct:5.1f}%)")
    print("=" * 80)

    # Save detailed results to file
    output_file = "classification_results.txt"

    print("=" * 80)
    print("Detailed Classification Results")
    print("=" * 80)

    for benchmark_name in sorted(benchmark_results.keys()):
        results = benchmark_results[benchmark_name]
        print(f"{benchmark_name}:")
        print("-" * 80)

        for filename in sorted(results.keys()):
            info = results[filename]
            if info["error"]:
                print(f"  {filename:50s} ERROR: {info['error']}")
            else:
                type_name = info["type"].name
                print(f"  {filename:50s} {type_name:10s}")

    print(f"Detailed results saved to {output_file}")


if __name__ == "__main__":
    main()
