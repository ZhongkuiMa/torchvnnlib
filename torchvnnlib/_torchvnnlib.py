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
    else:
        vnnlib_name = os.path.basename(vnnlib_path).replace(".vnnlib", "")
        target_folder_path = os.path.join(target_folder_path, vnnlib_name)

    os.makedirs(target_folder_path, exist_ok=True)

    or_folder_paths = []
    for i, or_properties in enumerate(and_properties):
        or_folder_path = os.path.join(target_folder_path, f"or_group_{i}")
        or_folder_paths.append(or_folder_path)
        os.makedirs(or_folder_path, exist_ok=True)

    save = partial(_save_property_file, verbose=verbose)
    with ThreadPoolExecutor() as executor:
        executor.map(save, and_properties, or_folder_paths)


# def _write_property(
#     and_properties: list[list[tuple[Tensor, list[Tensor]]]],
#     target_folder_path: str | None,
#     vnnlib_path: str,
#     verbose: bool,
# ):
#     # Create a folder to save this vnnlib property
#     if target_folder_path is None:
#         target_folder_path = vnnlib_path.replace(".vnnlib", "")
#     if not os.path.exists(target_folder_path):
#         os.makedirs(target_folder_path)
#
#     for i, or_properties in enumerate(and_properties):
#         if verbose:
#             print(f"Convert {i+1}/{len(and_properties)} OR groups...")
#
#         # Create a folder for each OR group
#         or_folder_path = os.path.join(target_folder_path, f"or_group_{i}")
#         os.makedirs(or_folder_path, exist_ok=True)
#
#         # Save the input and output constraints
#         for j, and_property in enumerate(or_properties):
#             if verbose:
#                 print(f"Convert {j+1}/{len(or_properties)} properties...")
#
#             input_bounds, output_constrs = and_property
#             data = {"input": input_bounds, "output": output_constrs}
#             # Save the data to a file
#             file_name = f"sub_prop_{j}.pth"
#             file_path = os.path.join(or_folder_path, file_name)
#             torch.save(data, file_path)


class TorchVNNLIB:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def convert(self, vnnlib_path: str, target_folder_path: str | None = None):
        if self.verbose:
            print(f"Torch vnnlib {vnnlib_path}...")
            t = time.perf_counter()

        with open(vnnlib_path, "r") as f:
            lines = f.readlines()

        # Extract all the statements "(assert <expr>)" from the vnnlib file
        lines, n_inputs, n_outputs = preprocess_vnnlib(lines)

        # The following operations can be done in parallel, so we process them in a list

        # Tokenize the lines to get a list of tokens
        tokens_list = tokenize(lines)
        # Get the expression of each line as a list
        # t = time.perf_counter()
        expr = parse(tokens_list)
        # print(f"Parsed {len(expr.args)} expressions in {time.perf_counter() - t:.4f}s")

        # t = time.perf_counter()
        expr = optimize(expr)
        # print(f"Optimized {len(expr.args)} expressions in {time.perf_counter() - t:.4f}s")

        # t = time.perf_counter()
        expr = flatten(expr)
        # print(f"Flattened {len(expr.args)} expressions in {time.perf_counter() - t:.4f}s")

        # if self.verbose:
        #     print_ast(expr)

        # t= time.perf_counter()
        and_properties = convert_to_tensor(expr, n_inputs, n_outputs)
        # print(f"Converted {len(and_properties)} AND properties in {time.perf_counter() - t:.4f}s")

        # t = time.perf_counter()
        _write_property(and_properties, target_folder_path, vnnlib_path, self.verbose)
        # print(f"Written {len(and_properties)} AND properties in {time.perf_counter() - t:.4f}s")

        if self.verbose:
            print(f"Saved to {target_folder_path} ({time.perf_counter() - t:.4f}s)")
