from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":

    converter = TorchVNNLIB(verbose=True)
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/dist_shift_2023/vnnlib/"
        "index966_delta0.13.vnnlib"
    )
    converter.convert(file_path)
