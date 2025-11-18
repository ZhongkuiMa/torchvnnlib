__docformat__ = "restructuredtext"
__all__ = ["TorchVNNLIB"]

import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from torch import Tensor

from ._to_tensor import convert_to_tensor
from .ast import preprocess_vnnlib, tokenize, parse, optimize, flatten
from .fast_type import (
    fast_detect_type,
    parse_simple_patterns,
    process_type1,
    process_type2,
    process_type3,
    process_type4,
    process_type5,
    VNNLIBType,
)


def _save_property_file(
    or_properties: list[tuple[Tensor, list[Tensor]]],
    or_folder_path: str,
    verbose: bool = False,
) -> None:
    """Save OR properties to individual .pth files.

    :param or_properties: List of (input_bounds, output_constraints) tuples
    :param or_folder_path: Directory path to save files
    :param verbose: Print progress messages
    """
    for j, and_property in enumerate(or_properties):
        if verbose:
            print(f"Converting {j+1}/{len(or_properties)} properties")

        input_bounds, output_constrs = and_property
        data = {"input": input_bounds, "output": output_constrs}
        file_name = f"sub_prop_{j}.pth"
        file_path = os.path.join(or_folder_path, file_name)
        torch.save(data, file_path)


def _write_property(
    and_properties: list[list[tuple[Tensor, list[Tensor]]]],
    target_folder_path: str | None,
    vnnlib_path: str,
    verbose: bool = False,
) -> None:
    """Write all properties to organized folder structure.

    :param and_properties: Nested list of properties
    :param target_folder_path: Output directory path
    :param vnnlib_path: Original VNN-LIB file path
    :param verbose: Print progress messages
    """
    if target_folder_path is None:
        target_folder_path = vnnlib_path.replace(".vnnlib", "")

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
    """VNN-LIB to PyTorch tensor converter."""

    def __init__(
        self,
        verbose: bool = False,
        use_parallel: bool = False,
        detect_fast_type: bool = True,
    ) -> None:
        """Initialize TorchVNNLIB converter.

        :param verbose: Print detailed timing information
        :param use_parallel: Use parallel processing where possible
        :param detect_fast_type: Use optimized type-specific processors
        """
        self.verbose = verbose
        self.use_parallel = use_parallel
        self.detect_fast_type = detect_fast_type
        self.conversion_stats: dict[str, dict] = {}

    def _process_by_type(
        self,
        vnnlib_type: VNNLIBType,
        lines: list[str],
        n_inputs: int,
        n_outputs: int,
    ) -> list[list[tuple[Tensor, list[Tensor]]]] | None:
        """Process VNN-LIB file using type-specific processor.

        :param vnnlib_type: Detected VNN-LIB type
        :param lines: Preprocessed assertion lines
        :param n_inputs: Number of input variables
        :param n_outputs: Number of output variables
        :return: Processed properties or None if type requires AST processing
        """
        t = time.perf_counter()

        if vnnlib_type == VNNLIBType.TYPE1:
            parsed_data = parse_simple_patterns(lines, verbose=self.verbose)
            and_properties = process_type1(
                parsed_data["simple_input_bounds"],
                parsed_data["simple_output_constrs"],
                parsed_data["complex_lines"],
                n_inputs,
                n_outputs,
                verbose=self.verbose,
                simple_output_bounds=parsed_data["simple_output_bounds"],
            )
            if self.verbose:
                print(f"Type1 processing: {time.perf_counter() - t:.4f}s")
            return and_properties

        elif vnnlib_type == VNNLIBType.TYPE2:
            parsed_data = parse_simple_patterns(lines, verbose=self.verbose)
            and_properties = process_type2(
                lines,
                n_inputs,
                n_outputs,
                verbose=self.verbose,
                parsed_data=parsed_data,
            )
            if self.verbose:
                print(f"Type2 processing: {time.perf_counter() - t:.4f}s")
            return and_properties

        elif vnnlib_type == VNNLIBType.TYPE3:
            parsed_data = parse_simple_patterns(lines, verbose=self.verbose)
            and_properties = process_type3(
                lines,
                n_inputs,
                n_outputs,
                verbose=self.verbose,
                parsed_data=parsed_data,
            )
            if self.verbose:
                print(f"Type3 processing: {time.perf_counter() - t:.4f}s")
            return and_properties

        elif vnnlib_type == VNNLIBType.TYPE4:
            and_properties = process_type4(
                lines, n_inputs, n_outputs, verbose=self.verbose
            )
            if self.verbose:
                print(f"Type4 processing: {time.perf_counter() - t:.4f}s")
            return and_properties

        elif vnnlib_type == VNNLIBType.TYPE5:
            and_properties = process_type5(
                lines, n_inputs, n_outputs, verbose=self.verbose
            )
            if self.verbose:
                print(f"Type5 processing: {time.perf_counter() - t:.4f}s")
            return and_properties

        else:
            if self.verbose:
                print("Complex structure detected, using AST processing")
            return None

    def convert(self, vnnlib_path: str, target_folder_path: str | None = None) -> None:
        """Convert VNN-LIB file to PyTorch tensors.

        :param vnnlib_path: Path to .vnnlib file
        :param target_folder_path: Output directory path
        """
        if self.verbose:
            print(f"Converting {vnnlib_path}")
        t_start = time.perf_counter()

        t = time.perf_counter()
        with open(vnnlib_path, "r") as f:
            lines = f.readlines()
        if self.verbose:
            print(f"Read file: {time.perf_counter() - t:.4f}s")

        t = time.perf_counter()
        lines, n_inputs, n_outputs = preprocess_vnnlib(lines)
        if self.verbose:
            print(
                f"Preprocessing: {n_inputs} inputs, {n_outputs} outputs, {time.perf_counter() - t:.4f}s"
            )

        use_type_processor = False
        detected_type = VNNLIBType.COMPLEX

        if self.detect_fast_type:
            vnnlib_type = fast_detect_type(lines, verbose=self.verbose)
            detected_type = vnnlib_type

            try:
                and_properties = self._process_by_type(
                    vnnlib_type, lines, n_inputs, n_outputs
                )
                use_type_processor = and_properties is not None
            except (ValueError, RuntimeError) as e:
                if self.verbose:
                    print(f"Type processor failed, fallback to AST: {str(e)[:100]}")
                use_type_processor = False

        if not use_type_processor:
            t = time.perf_counter()
            tokens_list = tokenize(
                lines, verbose=self.verbose, use_parallel=self.use_parallel
            )
            if self.verbose:
                print(f"Tokenization: {time.perf_counter() - t:.4f}s")

            t = time.perf_counter()
            expr = parse(
                tokens_list, verbose=self.verbose, use_parallel=self.use_parallel
            )
            if self.verbose:
                print(
                    f"Parsing: {len(expr.args)} expressions, {time.perf_counter() - t:.4f}s"
                )

            t = time.perf_counter()
            expr = optimize(expr, verbose=self.verbose, use_parallel=self.use_parallel)
            if self.verbose:
                print(f"Optimization: {time.perf_counter() - t:.4f}s")

            t = time.perf_counter()
            expr = flatten(expr)
            if self.verbose:
                print(f"Flattening: {time.perf_counter() - t:.4f}s")

            t = time.perf_counter()
            and_properties = convert_to_tensor(
                expr,
                n_inputs,
                n_outputs,
                verbose=self.verbose,
                use_parallel=self.use_parallel,
            )
            if self.verbose:
                print(
                    f"Tensor conversion: {len(and_properties)} AND properties, {time.perf_counter() - t:.4f}s"
                )

        t = time.perf_counter()
        _write_property(and_properties, target_folder_path, vnnlib_path, self.verbose)
        if self.verbose:
            print(f"Writing to disk: {time.perf_counter() - t:.4f}s")

        total_time = time.perf_counter() - t_start
        if self.verbose:
            print(f"Total time: {total_time:.4f}s")

        self.conversion_stats[vnnlib_path] = {
            "type": detected_type,
            "used_fast": use_type_processor,
            "time": total_time,
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
        }
