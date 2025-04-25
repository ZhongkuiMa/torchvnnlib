from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":

    converter = TorchVNNLIB(verbose=True)
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/linearizenn/vnnlib/"
        "prop_10_10_0_io.vnnlib"
    )
    converter.convert(file_path)
