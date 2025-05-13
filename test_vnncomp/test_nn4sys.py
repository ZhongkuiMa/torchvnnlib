from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":
    # This file cost times because too many properties.
    converter = TorchVNNLIB()
    # The following example contains a scientific number.
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/nn4sys_2023/vnnlib/"
        "cardinality_0_1000_128.vnnlib"
    )
    # file_path = (
    #     "../../vnncomp2024_benchmarks/benchmarks/nn4sys_2023/vnnlib/"
    #     "cardinality_0_1_128.vnnlib"
    # )
    converter.convert(file_path)
