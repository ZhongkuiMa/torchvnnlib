import time

import torch

from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":

    converter = TorchVNNLIB(verbose=True)

    file_path = "prop_6.vnnlib"
    converter.convert(file_path)
    data_path = file_path.replace(".vnnlib", ".pth")
    t = time.perf_counter()
    data = torch.load(data_path)
    t = time.perf_counter() - t
    print(f"Loaded {data_path} ({t:.4f}s)")

    file_path = "TinyImageNet_resnet_medium_prop_idx_57_sidx_9858_eps_0.0039.vnnlib"
    converter.convert(file_path)
    data_path = file_path.replace(".vnnlib", ".pth")
    t = time.perf_counter()
    data = torch.load(data_path)
    t = time.perf_counter() - t
    print(f"Loaded {data_path} ({t:.4f}s)")
