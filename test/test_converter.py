from torchvnnlib import TorchVNNLIB


def test_example1():
    converter = TorchVNNLIB(verbose=True)
    file_path = "../data/prop_6.vnnlib"
    converter.convert(file_path)


def test_example2():
    converter = TorchVNNLIB(verbose=True)
    file_path = "../data/example.vnnlib"
    converter.convert(file_path)


def test_example3():
    converter = TorchVNNLIB(verbose=True)
    file_path = (
        "../data/TinyImageNet_resnet_medium_prop_idx_57_sidx_9858_eps_0.0039.vnnlib"
    )
    converter.convert(file_path)


if __name__ == "__main__":
    test_example1()
    test_example2()
    test_example3()
