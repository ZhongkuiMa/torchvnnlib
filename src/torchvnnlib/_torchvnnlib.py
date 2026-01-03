__docformat__ = "restructuredtext"
__all__ = ["TorchVNNLIB"]

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import cast

from torchvnnlib._backend import Backend, TensorLike, get_backend
from torchvnnlib._to_tensor import convert_to_tensor
from torchvnnlib.ast import (
    And,
    Or,
    flatten,
    optimize,
    parse,
    preprocess_vnnlib,
    tokenize,
)
from torchvnnlib.fast_type import (
    VNNLIBType,
    fast_detect_type,
    parse_simple_patterns,
    process_type1,
    process_type2,
    process_type3,
    process_type4,
    process_type5,
)


def _save_property_file(
    or_properties: list[tuple[TensorLike, list[TensorLike]]],
    or_folder_path: str,
    backend: Backend,
    verbose: bool = False,
) -> None:
    """Save OR properties to individual files.

    :param or_properties: List of (input_bounds, output_constraints) tuples
    :param or_folder_path: Directory path to save files
    :param backend: Backend instance for saving operations
    :param verbose: Print progress messages
    """
    for j, and_property in enumerate(or_properties):
        if verbose:
            print(f"Converting {j + 1}/{len(or_properties)} properties")

        input_bounds, output_constrs = and_property
        data = {"input": input_bounds, "output": output_constrs}
        file_name = f"sub_prop_{j}{backend.file_extension}"
        file_path = str(Path(or_folder_path) / file_name)
        backend.save(data, file_path)


def _write_property(
    and_properties: list[list[tuple[TensorLike, list[TensorLike]]]],
    target_folder_path: str | None,
    vnnlib_path: str,
    backend: Backend,
    verbose: bool = False,
) -> None:
    """Write all properties to organized folder structure.

    :param and_properties: Nested list of properties
    :param target_folder_path: Output directory path
    :param vnnlib_path: Original VNN-LIB file path
    :param backend: Backend instance for saving operations
    :param verbose: Print progress messages
    """
    if target_folder_path is None:
        target_folder_path = str(Path(vnnlib_path).with_suffix(""))

    target_path = Path(target_folder_path)
    target_path.mkdir(parents=True, exist_ok=True)

    or_folder_paths = []
    for i, _or_properties in enumerate(and_properties):
        or_folder_path = str(target_path / f"or_group_{i}")
        or_folder_paths.append(or_folder_path)
        Path(or_folder_path).mkdir(parents=True, exist_ok=True)

    save = partial(_save_property_file, backend=backend, verbose=verbose)
    with ThreadPoolExecutor() as executor:
        executor.map(save, and_properties, or_folder_paths)


class TorchVNNLIB:
    """VNN-LIB to tensor converter supporting PyTorch and NumPy backends."""

    def __init__(
        self,
        verbose: bool = False,
        use_parallel: bool = False,
        detect_fast_type: bool = True,
        output_format: str = "torch",
    ) -> None:
        """Initialize TorchVNNLIB converter.

        :param verbose: Print detailed timing information
        :param use_parallel: Use parallel processing where possible
        :param detect_fast_type: Use optimized type-specific processors
        :param output_format: Output format ('torch' for .pth or 'numpy' for .npz)
        """
        self.verbose = verbose
        self.use_parallel = use_parallel
        self.detect_fast_type = detect_fast_type
        self.output_format = output_format
        self.backend = get_backend(output_format)
        self.conversion_stats: dict[str, dict] = {}

    def _process_type1(
        self, lines: list[str], n_inputs: int, n_outputs: int, t: float
    ) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
        """Process TYPE1 VNNLIB."""
        parsed_data = parse_simple_patterns(lines, verbose=self.verbose)
        and_properties = process_type1(
            parsed_data["simple_input_bounds"],
            parsed_data["simple_output_constrs"],
            parsed_data["complex_lines"],
            n_inputs,
            n_outputs,
            self.backend,
            verbose=self.verbose,
            simple_output_bounds=parsed_data["simple_output_bounds"],
        )
        if self.verbose:
            print(f"Type1 processing: {time.perf_counter() - t:.4f}s")
        return and_properties

    def _process_type234(
        self,
        processor_func,
        lines: list[str],
        n_inputs: int,
        n_outputs: int,
        type_name: str,
        t: float,
        use_parsed_data: bool = False,
    ) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
        """Process TYPE2, TYPE3, or TYPE4 VNNLIB."""
        if use_parsed_data:
            parsed_data = parse_simple_patterns(lines, verbose=self.verbose)
            and_properties = processor_func(
                lines,
                n_inputs,
                n_outputs,
                self.backend,
                verbose=self.verbose,
                parsed_data=parsed_data,
            )
        else:
            and_properties = processor_func(
                lines, n_inputs, n_outputs, self.backend, verbose=self.verbose
            )
        if self.verbose:
            print(f"{type_name} processing: {time.perf_counter() - t:.4f}s")
        return and_properties  # type: ignore[no-any-return]

    def _process_by_type(
        self,
        vnnlib_type: VNNLIBType,
        lines: list[str],
        n_inputs: int,
        n_outputs: int,
    ) -> list[list[tuple[TensorLike, list[TensorLike]]]] | None:
        """Process VNN-LIB file using type-specific processor."""
        t = time.perf_counter()

        if vnnlib_type == VNNLIBType.TYPE1:
            return self._process_type1(lines, n_inputs, n_outputs, t)
        if vnnlib_type == VNNLIBType.TYPE2:
            return self._process_type234(
                process_type2, lines, n_inputs, n_outputs, "Type2", t, use_parsed_data=True
            )
        if vnnlib_type == VNNLIBType.TYPE3:
            return self._process_type234(
                process_type3, lines, n_inputs, n_outputs, "Type3", t, use_parsed_data=True
            )
        if vnnlib_type == VNNLIBType.TYPE4:
            return self._process_type234(process_type4, lines, n_inputs, n_outputs, "Type4", t)
        if vnnlib_type == VNNLIBType.TYPE5:
            return self._process_type234(process_type5, lines, n_inputs, n_outputs, "Type5", t)

        if self.verbose:
            print("Complex structure detected, using AST processing")
        return None

    def _process_ast(
        self, lines: list[str], n_inputs: int, n_outputs: int
    ) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
        """Process using AST pipeline (tokenize, parse, optimize, flatten, convert)."""
        t = time.perf_counter()
        tokens_list = tokenize(lines, verbose=self.verbose, use_parallel=self.use_parallel)
        if self.verbose:
            print(f"Tokenization: {time.perf_counter() - t:.4f}s")

        t = time.perf_counter()
        expr = parse(tokens_list, verbose=self.verbose, use_parallel=self.use_parallel)
        if self.verbose:
            nary_expr = cast(And | Or, expr)
            print(f"Parsing: {len(nary_expr.args)} expressions, {time.perf_counter() - t:.4f}s")

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
            self.backend,
            verbose=self.verbose,
            use_parallel=self.use_parallel,
        )
        if self.verbose:
            elapsed = time.perf_counter() - t
            num_props = len(and_properties)
            print(f"Tensor conversion: {num_props} AND properties, {elapsed:.4f}s")
        return and_properties

    def convert(self, vnnlib_path: str, target_folder_path: str | None = None) -> None:
        """Convert VNN-LIB file to tensor data.

        :param vnnlib_path: Path to .vnnlib file
        :param target_folder_path: Output directory path
        """
        if self.verbose:
            print(f"Converting {vnnlib_path}")
            print(f"Output format: {self.output_format}")
        t_start = time.perf_counter()

        t = time.perf_counter()
        vnnlib_file = Path(vnnlib_path)
        with vnnlib_file.open() as f:
            lines = f.readlines()
        if self.verbose:
            print(f"Read file: {time.perf_counter() - t:.4f}s")

        t = time.perf_counter()
        lines, n_inputs, n_outputs = preprocess_vnnlib(lines)
        if self.verbose:
            elapsed = time.perf_counter() - t
            print(f"Preprocessing: {n_inputs} inputs, {n_outputs} outputs, {elapsed:.4f}s")

        use_type_processor = False
        detected_type = VNNLIBType.COMPLEX

        if self.detect_fast_type:
            vnnlib_type = fast_detect_type(lines, verbose=self.verbose)
            detected_type = vnnlib_type

            try:
                and_properties = self._process_by_type(vnnlib_type, lines, n_inputs, n_outputs)
                use_type_processor = and_properties is not None
            except (ValueError, RuntimeError) as e:
                if self.verbose:
                    print(f"Type processor failed, fallback to AST: {str(e)[:100]}")
                use_type_processor = False

        if not use_type_processor:
            and_properties = self._process_ast(lines, n_inputs, n_outputs)

        # Type narrowing: ensure and_properties is not None
        assert and_properties is not None, (
            "and_properties must be assigned by type processor or AST"
        )
        t = time.perf_counter()
        _write_property(and_properties, target_folder_path, vnnlib_path, self.backend, self.verbose)
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
            "output_format": self.output_format,
        }
