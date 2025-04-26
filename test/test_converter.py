from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":

    converter = TorchVNNLIB(verbose=True)
    file_path = "../data/prop_6.vnnlib"
    converter.convert(file_path)
