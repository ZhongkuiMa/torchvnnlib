#!/usr/bin/env python3
"""
Compare parallel vs sequential processing performance.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":
    vnnlib_path = "benchmarks/vggnet16_2023/spec0_briard.vnnlib"

    if not os.path.exists(vnnlib_path):
        print(f"Error: {vnnlib_path} not found")
        sys.exit(1)

    # print("=" * 60)
    # print("PARALLEL PROCESSING (use_parallel=True)")
    # print("=" * 60)
    # converter_parallel = TorchVNNLIB(verbose=True, use_parallel=True)
    # converter_parallel.convert(vnnlib_path, target_folder_path="results_parallel")

    print("\n" + "=" * 60)
    print("SEQUENTIAL PROCESSING (use_parallel=False)")
    print("=" * 60)
    converter_sequential = TorchVNNLIB(verbose=True, use_parallel=False)
    converter_sequential.convert(vnnlib_path, target_folder_path="results_sequential")

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)
