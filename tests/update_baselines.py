#!/usr/bin/env python3
"""Update baseline outputs for torchvnnlib regression testing.

This script copies outputs from results/ to baselines/ to update golden references.

Workflow:
    1. Run test_torchvnnlib.py to generate results/
    2. Run this script to copy results/ -> baselines/
    3. Run test_torchvnnlib_regression.py to verify

Usage:
    python update_baselines.py                     # Copy all results to baselines
    python update_baselines.py --benchmark acasxu  # Copy specific benchmark
    python update_baselines.py --dry-run           # Show what would be copied
"""

import argparse
import shutil
import sys
from pathlib import Path


def copy_baseline(
    result_path: Path, baseline_path: Path, dry_run: bool = False
) -> tuple[bool, int]:
    """Copy a single result folder to baseline.

    :param result_path: Source path in results/
    :param baseline_path: Destination path in baselines/
    :param dry_run: If True, only show what would be copied
    :return: Tuple of (success, file_count)
    """
    if dry_run:
        if result_path.exists():
            pth_count = sum(1 for _ in result_path.rglob("*.pth"))
            print(
                f"  [DRY-RUN] Would copy: {result_path} -> {baseline_path} ({pth_count} .pth files)"
            )
            return True, pth_count
        print(f"  [DRY-RUN] Would skip (missing): {result_path}")
        return False, 0

    # Check source exists
    if not result_path.exists():
        print(f"  [SKIP] Missing result: {result_path}")
        return False, 0

    # Remove old baseline if exists
    if baseline_path.exists():
        shutil.rmtree(baseline_path)

    # Copy result to baseline
    shutil.copytree(result_path, baseline_path)

    # Count .pth files copied
    pth_count = sum(1 for _ in baseline_path.rglob("*.pth"))
    print(f"  [OK] Copied: {baseline_path} ({pth_count} .pth files)")
    return True, pth_count


def update_benchmark(
    benchmark_name: str,
    results_dir: Path,
    baselines_dir: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Update baselines for one benchmark.

    :param benchmark_name: Name of benchmark
    :param results_dir: Root results directory
    :param baselines_dir: Root baselines directory
    :param dry_run: If True, only show what would be copied
    :return: Tuple of (success_count, total_count)
    """
    result_bench_dir = results_dir / benchmark_name
    baseline_bench_dir = baselines_dir / benchmark_name

    if not result_bench_dir.exists():
        print(f"  [SKIP] No results: {benchmark_name}")
        return 0, 0

    # Create baseline benchmark directory
    if not dry_run:
        baseline_bench_dir.mkdir(parents=True, exist_ok=True)

    # Copy each property subdirectory
    success = 0
    total = 0

    for property_dir in sorted(result_bench_dir.iterdir()):
        if not property_dir.is_dir():
            continue

        total += 1
        baseline_property_path = baseline_bench_dir / property_dir.name
        copied, _ = copy_baseline(property_dir, baseline_property_path, dry_run)
        if copied:
            success += 1

    return success, total


def find_benchmarks_to_update(
    results_dir: Path, benchmark_arg: str | None
) -> tuple[list[str], int]:
    """Find which benchmarks to update.

    :param results_dir: Root results directory
    :param benchmark_arg: Specific benchmark name or None for all
    :return: Tuple of (benchmark_list, exit_code) where exit_code is 0 on success
    """
    if benchmark_arg:
        benchmark_path = results_dir / benchmark_arg
        if not benchmark_path.exists():
            print(f"Error: Benchmark '{benchmark_arg}' not found in results/")
            print(f"\\nAvailable benchmarks in {results_dir}:")
            for bench in sorted(results_dir.iterdir()):
                if bench.is_dir():
                    print(f"  - {bench.name}")
            return [], 1
        return [benchmark_arg], 0

    benchmarks = [bench.name for bench in sorted(results_dir.iterdir()) if bench.is_dir()]
    if not benchmarks:
        print("No benchmarks found in results/")
        return [], 1

    return benchmarks, 0


def main():
    """Update baselines from results directory."""
    parser = argparse.ArgumentParser(
        description="Update baselines by copying results/ to baselines/",
        epilog="Example: python update_baselines.py --benchmark acasxu_2023",
    )
    parser.add_argument(
        "--benchmark",
        help="Update specific benchmark only (e.g., acasxu_2023)",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without making changes",
    )
    args = parser.parse_args()

    # Setup paths
    test_dir = Path(__file__).parent
    results_dir = test_dir / "results"
    baselines_dir = test_dir / "baselines"

    # Check results directory exists
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("\\nRun test_torchvnnlib.py first to generate results/")
        return 1

    # Create baselines directory if missing
    if not args.dry_run:
        baselines_dir.mkdir(exist_ok=True)

    # Find benchmarks to update
    benchmarks_to_update, exit_code = find_benchmarks_to_update(results_dir, args.benchmark)
    if exit_code != 0:
        return exit_code

    # Update baselines
    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Updating baselines from results/")
    print(f"Benchmarks: {len(benchmarks_to_update)}")
    print("=" * 80)

    total_success = 0
    total_count = 0

    for benchmark_name in benchmarks_to_update:
        print(f"\\n{benchmark_name}:")
        success, count = update_benchmark(
            benchmark_name, results_dir, baselines_dir, dry_run=args.dry_run
        )
        total_success += success
        total_count += count

        if count > 0:
            print(f"  {success}/{count} properties copied")

    print("\\n" + "=" * 80)
    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Baseline update complete!")
    print(f"Total: {total_success}/{total_count} properties copied")

    if not args.dry_run and total_success > 0:
        print(f"\\nBaselines updated in: {baselines_dir}")
        print("Run test_torchvnnlib_regression.py to verify")

    return 0


if __name__ == "__main__":
    sys.exit(main())
