"""VNNComp Benchmark Test Runner for TorchVNNLib.

Runs conversion on all VNNComp benchmark vnnlib files and validates results.
"""

import os
import time
from collections import defaultdict

from torchvnnlib import TorchVNNLIB
from utils import find_benchmarks_folders, find_all_vnnlib_files

if __name__ == "__main__":
    dir_name = "benchmarks"

    benchmark_dirs = find_benchmarks_folders(dir_name)
    print(f"Found {len(benchmark_dirs)} benchmark directories")
    vnnlib_paths = find_all_vnnlib_files(benchmark_dirs)
    print(f"Total files: {len(vnnlib_paths)}")
    print("=" * 70)

    # Uncomment to test specific files
    # vnnlib_paths = [
    #     "benchmarks/acasxu/prop_6.vnnlib"
    # ]

    failed_vnnlib_paths = []
    verbose = False

    success_count = 0
    total_count = 0
    converter = TorchVNNLIB()

    # Track statistics by benchmark
    benchmark_stats = defaultdict(
        lambda: {
            "total": 0,
            "success": 0,
            "total_time": 0.0,
            "type_counts": defaultdict(int),
            "fast_type_counts": defaultdict(int),
        }
    )

    overall_start_time = time.perf_counter()

    for i, vnnlib_path in enumerate(vnnlib_paths):
        # Get benchmark name from path
        benchmark_name = (
            vnnlib_path.split(os.sep)[1] if os.sep in vnnlib_path else "unknown"
        )

        print(f"[{i+1}/{len(vnnlib_paths)}] ", end="", flush=True)
        time_start = time.perf_counter()

        try:
            success = False
            # Convert to temporary output folder (will be cleaned up)
            output_folder = f"_temp_test_output_{i}"
            converter.convert(vnnlib_path, target_folder_path=output_folder)

            # Clean up temporary output
            import shutil

            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)

            success = True
            success_count += 1

            # Collect statistics
            if vnnlib_path in converter.conversion_stats:
                stats = converter.conversion_stats[vnnlib_path]
                benchmark_stats[benchmark_name]["total"] += 1
                benchmark_stats[benchmark_name]["success"] += 1
                benchmark_stats[benchmark_name]["total_time"] += stats["time"]
                benchmark_stats[benchmark_name]["type_counts"][stats["type"].name] += 1
                if stats["used_fast"]:
                    benchmark_stats[benchmark_name]["fast_type_counts"][
                        stats["type"].name
                    ] += 1

        except Exception as e:
            failed_vnnlib_paths.append(vnnlib_path)
            benchmark_stats[benchmark_name]["total"] += 1
            success = False
            # raise e  # Uncomment to stop on first error

        elapsed = time.perf_counter() - time_start
        if success:
            print(f"OK ({elapsed:.2f}s) - {os.path.basename(vnnlib_path)}")
        else:
            print(f"FAILED ({elapsed:.2f}s) - {os.path.basename(vnnlib_path)}")
        total_count += 1

    overall_elapsed = time.perf_counter() - overall_start_time

    if failed_vnnlib_paths:
        print(f"\n{len(failed_vnnlib_paths)} failed files:")
        for f in failed_vnnlib_paths:
            print(f"  {f}")

    print(f"\nConversion complete: {success_count}/{total_count} succeeded")
    print(f"Total time: {overall_elapsed:.2f}s")
    print(f"Average time: {overall_elapsed/total_count:.4f}s per file")

    # Print detailed statistics by benchmark
    print("\n" + "=" * 80)
    print("BENCHMARK ANALYSIS - Fast Type Usage")
    print("=" * 80)

    for benchmark_name in sorted(benchmark_stats.keys()):
        stats = benchmark_stats[benchmark_name]
        print(f"\n{benchmark_name}:")
        print(f"  Total files: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Total time: {stats['total_time']:.2f}s")
        print(f"  Avg time: {stats['total_time']/max(stats['total'], 1):.4f}s")
        print(f"  Type distribution:")
        for type_name, count in sorted(stats["type_counts"].items()):
            fast_count = stats["fast_type_counts"].get(type_name, 0)
            fast_rate = 100.0 * fast_count / count if count > 0 else 0
            print(
                f"    {type_name:12s}: {count:4d} files ({fast_count:4d} used fast, {fast_rate:5.1f}%)"
            )

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    total_by_type = defaultdict(int)
    total_fast_by_type = defaultdict(int)
    for stats in benchmark_stats.values():
        for type_name, count in stats["type_counts"].items():
            total_by_type[type_name] += count
            total_fast_by_type[type_name] += stats["fast_type_counts"].get(type_name, 0)

    print("Overall type distribution:")
    for type_name in sorted(total_by_type.keys()):
        count = total_by_type[type_name]
        fast_count = total_fast_by_type[type_name]
        fast_rate = 100.0 * fast_count / count if count > 0 else 0
        print(
            f"  {type_name:12s}: {count:4d} files ({fast_count:4d} used fast, {fast_rate:5.1f}%)"
        )
