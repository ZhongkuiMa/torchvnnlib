from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":

    converter = TorchVNNLIB(verbose=True)
    # file_path = (
    #     "../../vnncomp2024_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_6.vnnlib"
    # )
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_7.vnnlib"
    )
    converter.convert(file_path)
