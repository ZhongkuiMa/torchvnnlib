import time

import torch

from ._parse import load_vnnlib


class TorchVNNLIB:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def convert(self, vnnlib_path: str, target_path: str | None = None):
        if self.verbose:
            print(f"Torch vnnlib {vnnlib_path}...")
            t = time.perf_counter()

        result = load_vnnlib(vnnlib_path, verbose=self.verbose)

        if target_path is None:
            target_path = vnnlib_path.replace(".vnnlib", ".pth")
        torch.save(result, target_path)

        if self.verbose:
            print(f"Saved to {target_path} ({time.perf_counter() - t:.4f}s)")
