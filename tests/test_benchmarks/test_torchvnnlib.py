"""TorchVNNLib Converter Test Suite.

Tests the TorchVNNLib converter on all VNNComp benchmark vnnlib files
and collects conversion statistics including type detection and performance metrics.

Outputs are saved to results/{benchmark_name}/{property_name}/ for later comparison
with baselines.

Usage::

    # Run all benchmarks (pytest style)
    pytest test_torchvnnlib.py -v -s

    # Run specific benchmark
    pytest test_torchvnnlib.py::test_benchmark_conversion -k acasxu -v -s

    # Backward compatible script execution
    python test_torchvnnlib.py
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import pytest

from torchvnnlib import TorchVNNLIB
from torchvnnlib.tests.utils import (
    find_all_vnnlib_files,
    find_benchmarks_folders,
)


def get_all_benchmarks():
    """Collect all benchmark directories for parametrization.

    :return: List of benchmark directory paths
    """
    test_dir = Path(__file__).parent
    # benchmarks dir is at tests/test_benchmarks/benchmarks
    benchmarks_dir = test_dir / "benchmarks"

    if not benchmarks_dir.exists():
        return []

    benchmark_dirs = find_benchmarks_folders(str(benchmarks_dir))
    return [str(b) for b in benchmark_dirs]


# Global statistics for session summary
_benchmark_stats: defaultdict[str, dict[str, int | float | defaultdict[str, int]]] = defaultdict(
    lambda: {
        "total": 0,
        "success": 0,
        "total_time": 0.0,
        "type_counts": defaultdict(int),
        "fast_type_counts": defaultdict(int),
    }
)
_overall_stats = {"success_count": 0, "total_count": 0, "overall_time": 0.0}


@pytest.mark.benchmark
@pytest.mark.parametrize("benchmark_dir", get_all_benchmarks())
def test_benchmark_conversion(benchmark_dir, test_dir, results_dir):
    """Test conversion for one benchmark directory.

    Outputs are saved to results/{benchmark_name}/{property_name}/

    :param benchmark_dir: Path to benchmark directory
    :param test_dir: Test directory fixture
    :param results_dir: Results directory fixture
    """
    benchmark_name = Path(benchmark_dir).name
    vnnlib_files = find_all_vnnlib_files([benchmark_dir])

    if not vnnlib_files:
        pytest.skip(f"No vnnlib files found in {benchmark_name}")

    print(f"\n{'=' * 80}")
    print(f"BENCHMARK: {benchmark_name}")
    print(f"Files: {len(vnnlib_files)}")
    print(f"{'=' * 80}")

    converter = TorchVNNLIB()
    success_count = 0
    failed = []

    bench_start = time.perf_counter()

    for i, vnnlib_path in enumerate(vnnlib_files, 1):
        file_start = time.perf_counter()

        try:
            # Output to results/{benchmark_name}/{property_name}/
            vnnlib_file = Path(vnnlib_path)
            property_name = vnnlib_file.stem
            output_folder = results_dir / benchmark_name / property_name
            converter.convert(vnnlib_path, target_folder_path=str(output_folder))

            success_count += 1

            # Collect statistics
            if vnnlib_path in converter.conversion_stats:
                stats = converter.conversion_stats[vnnlib_path]
                _benchmark_stats[benchmark_name]["success"] += 1
                _benchmark_stats[benchmark_name]["total_time"] += stats["time"]
                _benchmark_stats[benchmark_name]["type_counts"][stats["type"].name] += 1
                if stats["used_fast"]:
                    _benchmark_stats[benchmark_name]["fast_type_counts"][stats["type"].name] += 1

            elapsed = time.perf_counter() - file_start
            print(f"  [{i}/{len(vnnlib_files)}] OK ({elapsed:.2f}s) - {Path(vnnlib_path).name}")

        except (OSError, RuntimeError, ValueError) as e:
            failed.append(Path(vnnlib_path).name)
            elapsed = time.perf_counter() - file_start
            filename = Path(vnnlib_path).name
            print(f"  [{i}/{len(vnnlib_files)}] FAILED ({elapsed:.2f}s) - {filename}: {e}")

        _benchmark_stats[benchmark_name]["total"] += 1

    bench_elapsed = time.perf_counter() - bench_start
    _overall_stats["success_count"] += success_count
    _overall_stats["total_count"] += len(vnnlib_files)
    _overall_stats["overall_time"] += bench_elapsed

    # Print summary for this benchmark
    total = _benchmark_stats[benchmark_name]["total"]
    print(f"\n  Success: {success_count}/{total}")
    print(f"  Total time: {bench_elapsed:.2f}s")
    print(f"  Average time: {bench_elapsed / total:.4f}s per file")

    # Print type distribution
    type_counts = _benchmark_stats[benchmark_name]["type_counts"]
    fast_counts = _benchmark_stats[benchmark_name]["fast_type_counts"]
    if type_counts:
        print("  Type distribution:")
        for type_name in sorted(type_counts.keys()):
            count = type_counts[type_name]
            fast_count = fast_counts.get(type_name, 0)
            fast_rate = 100.0 * fast_count / count if count > 0 else 0
            stat_str = f"{type_name:12s}: {count:4d} files ({fast_count:4d} used fast,"
            print(f"    {stat_str} {fast_rate:5.1f}%)")

    assert success_count > 0 or not vnnlib_files, f"No files converted in {benchmark_name}"
    if failed:
        print(f"\n  Failed files: {', '.join(failed[:3])}")


def print_summary():
    """Print summary statistics for all benchmarks."""
    if not _benchmark_stats:
        return

    print(f"\n{'=' * 80}")
    print("BENCHMARK ANALYSIS - Fast Type Usage")
    print(f"{'=' * 80}")

    for benchmark_name in sorted(_benchmark_stats.keys()):
        stats = _benchmark_stats[benchmark_name]
        print(f"\n{benchmark_name}:")
        print(f"  Total files: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Total time: {stats['total_time']:.2f}s")
        print(f"  Avg time: {stats['total_time'] / max(stats['total'], 1):.4f}s")
        print("  Type distribution:")
        for type_name, count in sorted(stats["type_counts"].items()):
            fast_count = stats["fast_type_counts"].get(type_name, 0)
            fast_rate = 100.0 * fast_count / count if count > 0 else 0
            stat_str = f"{type_name:12s}: {count:4d} files ({fast_count:4d} used fast,"
            print(f"    {stat_str} {fast_rate:5.1f}%)")

    # Overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")
    total_by_type = defaultdict(int)
    total_fast_by_type = defaultdict(int)
    for stats in _benchmark_stats.values():
        for type_name, count in stats["type_counts"].items():
            total_by_type[type_name] += count
            total_fast_by_type[type_name] += stats["fast_type_counts"].get(type_name, 0)

    print("Overall type distribution:")
    for type_name in sorted(total_by_type.keys()):
        count = total_by_type[type_name]
        fast_count = total_fast_by_type[type_name]
        fast_rate = 100.0 * fast_count / count if count > 0 else 0
        print(f"  {type_name:12s}: {count:4d} files ({fast_count:4d} used fast, {fast_rate:5.1f}%)")

    if _overall_stats["total_count"] > 0:
        print(f"\nTotal conversion time: {_overall_stats['overall_time']:.2f}s")
        avg_time = _overall_stats["overall_time"] / _overall_stats["total_count"]
        print(f"Average time: {avg_time:.4f}s per file")
        success = _overall_stats["success_count"]
        total = _overall_stats["total_count"]
        print(f"Overall success: {success}/{total}")


def main():
    """Backward compatible entry point supporting pytest execution."""
    pytest_args = [__file__, "-v", "-s"]
    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
