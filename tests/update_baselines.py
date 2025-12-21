#!/usr/bin/env python3
"""Update baseline outputs for torchvnnlib regression testing.

This script regenerates all baseline outputs using AST-only conversion
(no fast-type optimizations) to ensure consistent golden references.

Usage:
    python update_baselines.py                    # Update all benchmarks
    python update_baselines.py --benchmark acasxu # Update specific benchmark
    python update_baselines.py --dry-run          # Show what would be updated
"""

import argparse
import shutil
import sys
from pathlib import Path

from torchvnnlib import TorchVNNLIB

# Import utilities (reuse existing code)
sys.path.insert(0, str(Path(__file__).parent))
from utils import find_all_vnnlib_files, get_benchmark_name


def update_baseline(vnnlib_path: str, baselines_dir: Path, dry_run: bool = False):
    """Update baseline for a single vnnlib file.

    :param vnnlib_path: Path to vnnlib file
    :param baselines_dir: Baselines directory path
    :param dry_run: If True, only show what would be updated
    """
    benchmark_name = get_benchmark_name(vnnlib_path)
    vnnlib_file = Path(vnnlib_path)

    # Baseline path: baselines/[benchmark]/[property_name]/
    baseline_path = baselines_dir / benchmark_name / vnnlib_file.stem

    if dry_run:
        print(f"  [DRY-RUN] Would update: {baseline_path}")
        return

    # Remove old baseline if exists
    if baseline_path.exists():
        shutil.rmtree(baseline_path)

    # Create new baseline using AST-only conversion (no fast optimizations)
    converter = TorchVNNLIB(detect_fast_type=False, verbose=False)
    converter.convert(vnnlib_path, str(baseline_path))

    # Count .pth files created
    pth_count = sum(1 for _ in baseline_path.rglob("*.pth"))
    print(f"  ✓ Updated: {baseline_path} ({pth_count} .pth files)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update baseline outputs for torchvnnlib regression testing",
        epilog="Example: python update_baselines.py --benchmark acasxu_2023 --limit 5",
    )
    parser.add_argument(
        "--benchmark",
        help="Update specific benchmark only (e.g., acasxu_2023)",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max files per benchmark (default: 20)",
    )
    args = parser.parse_args()

    # Setup paths
    test_dir = Path(__file__).parent
    benchmarks_dir = test_dir / "benchmarks"
    baselines_dir = test_dir / "baselines"

    # Create baselines directory if missing
    if not args.dry_run:
        baselines_dir.mkdir(exist_ok=True)

    # Find vnnlib files
    if args.benchmark:
        benchmark_path = benchmarks_dir / args.benchmark
        if not benchmark_path.exists():
            print(f"Error: Benchmark '{args.benchmark}' not found")
            print(f"\nAvailable benchmarks in {benchmarks_dir}:")
            for bench in sorted(benchmarks_dir.iterdir()):
                if bench.is_dir():
                    print(f"  - {bench.name}")
            return 1
        vnnlib_files = find_all_vnnlib_files([str(benchmark_path)], num_limit=args.limit)
    else:
        vnnlib_files = find_all_vnnlib_files([str(benchmarks_dir)], num_limit=args.limit)

    if not vnnlib_files:
        print("No vnnlib files found")
        return 1

    # Update baselines
    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Updating {len(vnnlib_files)} baselines...")
    print("=" * 80)

    for i, vnnlib_path in enumerate(vnnlib_files, 1):
        print(f"[{i}/{len(vnnlib_files)}] {Path(vnnlib_path).name}")
        try:
            update_baseline(vnnlib_path, baselines_dir, dry_run=args.dry_run)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")

    print("=" * 80)
    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Baseline update complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
