import os

from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":

    converter = TorchVNNLIB(verbose=True)
    file_path = (
        "../../../vnncomp2024_benchmarks/benchmarks/tllverifybench_2023/vnnlib/"
        "property_N=8_2.vnnlib"
    )
    # Print the absolute path of the file being converted
    print(f"Converting file: {os.path.abspath(file_path)}")
    converter.convert(file_path)
