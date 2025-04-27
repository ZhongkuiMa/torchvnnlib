import os
import time

import torch

from ._to_tensor import *
from .ast import *


class TorchVNNLIB:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def convert(self, vnnlib_path: str, target_folder_path: str | None = None):
        if self.verbose:
            print(f"Torch vnnlib {vnnlib_path}...")
            t = time.perf_counter()

        with open(vnnlib_path, "r") as f:
            lines = f.readlines()

        lines = pre_process_vnnlib(lines)
        tokens_list = tokenize(lines)
        expr = parse(tokens_list)
        expr = optimize(expr)
        expr = flatten(expr)

        # if self.verbose:
        #     print_ast(expr)

        and_properties = convert_to_tensor(expr)

        # Create a folder to save this vnnlib property
        if target_folder_path is None:
            target_folder_path = vnnlib_path.replace(".vnnlib", "")
        if not os.path.exists(target_folder_path):
            os.makedirs(target_folder_path)

        for i, or_properties in enumerate(and_properties):
            if self.verbose:
                print(f"Convert {i+1}/{len(and_properties)} OR groups...")

            # Create a folder for each OR group
            or_folder_path = os.path.join(target_folder_path, f"or_{i}")
            if not os.path.exists(or_folder_path):
                os.makedirs(or_folder_path)

            # Save the input and output constraints
            for j, and_property in enumerate(or_properties):
                if self.verbose:
                    print(f"Convert {j+1}/{len(or_properties)} properties...")

                input_bounds, output_constrs = and_property
                data = {"input": input_bounds, "output": output_constrs}
                # Save the data to a file
                file_name = f"property_{j}.pth"
                file_path = os.path.join(or_folder_path, file_name)
                torch.save(data, file_path)

        if self.verbose:
            print(f"Saved to {target_folder_path} ({time.perf_counter() - t:.4f}s)")
