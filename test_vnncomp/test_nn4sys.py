import time

from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":
    t = time.perf_counter()
    # This file cost times because too many properties.
    converter = TorchVNNLIB()
    # file_path = (
    #     "../../vnncomp2024_benchmarks/benchmarks/nn4sys_2023/vnnlib/"
    #     "cardinality_0_500_128.vnnlib"
    # )
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/nn4sys_2023/vnnlib/"
        "cardinality_0_1_128.vnnlib"
    )
    converter.convert(file_path)
    print(f"Cost time: {time.perf_counter() - t:.2f} seconds")
