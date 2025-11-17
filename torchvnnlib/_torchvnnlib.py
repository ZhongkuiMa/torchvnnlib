import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from torch import Tensor

from ._to_tensor import *
from .ast import *


def _save_property_file(
    or_properties: list[tuple[Tensor, list[Tensor]]],
    or_folder_path: str,
    verbose: bool = False
):
    # Save the input and output constraints
    for j, and_property in enumerate(or_properties):
        if verbose:
            print(f"Convert {j+1}/{len(or_properties)} properties...")

        input_bounds, output_constrs = and_property
        data = {"input": input_bounds, "output": output_constrs}
        # Save the data to a file
        file_name = f"sub_prop_{j}.pth"
        file_path = os.path.join(or_folder_path, file_name)
        torch.save(data, file_path)


def _write_property(
    and_properties: list[list[tuple[Tensor, list[Tensor]]]],
    target_folder_path: str | None,
    vnnlib_path: str,
    verbose: bool = False
):
    if target_folder_path is None:
        target_folder_path = vnnlib_path.replace(".vnnlib", "")
    # else:
    #     raise NotImplementedError("Need checking here.")
    #     vnnlib_name = os.path.basename(vnnlib_path).replace(".vnnlib", "")
    #     target_folder_path = os.path.join(target_folder_path, vnnlib_name)

    os.makedirs(target_folder_path, exist_ok=True)

    or_folder_paths = []
    for i, or_properties in enumerate(and_properties):
        or_folder_path = os.path.join(target_folder_path, f"or_group_{i}")
        or_folder_paths.append(or_folder_path)
        os.makedirs(or_folder_path, exist_ok=True)

    save = partial(_save_property_file, verbose=verbose)
    with ThreadPoolExecutor() as executor:
        executor.map(save, and_properties, or_folder_paths)


class TorchVNNLIB:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def convert(self, vnnlib_path: str, target_folder_path: str | None = None):
        if self.verbose:
            print(f"Converting {vnnlib_path}...")
            t_start = time.perf_counter()

        # Read file
        t = time.perf_counter()
        with open(vnnlib_path, "r") as f:
            lines = f.readlines()
        if self.verbose:
            print(f"  Read file: {time.perf_counter() - t:.4f}s")

        # Extract all the statements "(assert <expr>)" from the vnnlib file
        t = time.perf_counter()
        lines, n_inputs, n_outputs = preprocess_vnnlib(lines)
        if self.verbose:
            print(f"  Preprocessing ({n_inputs} inputs, {n_outputs} outputs): {time.perf_counter() - t:.4f}s")

        # Tokenize the lines to get a list of tokens
        t = time.perf_counter()
        tokens_list = tokenize(lines)
        if self.verbose:
            print(f"  Tokenization: {time.perf_counter() - t:.4f}s")

        # Get the expression of each line as a list
        t = time.perf_counter()
        expr = parse(tokens_list)
        if self.verbose:
            print(f"  Parsing ({len(expr.args)} expressions): {time.perf_counter() - t:.4f}s")

        # Optimize expressions
        t = time.perf_counter()
        expr = optimize(expr)
        if self.verbose:
            print(f"  Optimization: {time.perf_counter() - t:.4f}s")

        # Flatten expressions
        t = time.perf_counter()
        expr = flatten(expr)
        if self.verbose:
            print(f"  Flattening: {time.perf_counter() - t:.4f}s")

        # Convert to tensors
        t = time.perf_counter()
        and_properties = convert_to_tensor(expr, n_inputs, n_outputs)
        if self.verbose:
            print(f"  Tensor conversion ({len(and_properties)} AND properties): {time.perf_counter() - t:.4f}s")

        # Write properties to disk
        t = time.perf_counter()
        _write_property(and_properties, target_folder_path, vnnlib_path, self.verbose)
        if self.verbose:
            print(f"  Writing to disk: {time.perf_counter() - t:.4f}s")

        if self.verbose:
            total_time = time.perf_counter() - t_start
            print(f"✓ Total time: {total_time:.4f}s")
