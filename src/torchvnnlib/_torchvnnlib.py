"""Main TorchVNNLIB class for loading and converting VNN-LIB files."""

__docformat__ = "restructuredtext"
__all__ = ["ConversionStats", "TorchVNNLIB"]

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Literal, TypedDict, cast

from torchvnnlib._backend import Backend, TensorLike, get_backend
from torchvnnlib._logging import _enable_verbose
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

_logger = logging.getLogger(__name__)


class ConversionStats(TypedDict):
    """Schema for per-file conversion statistics recorded by :class:`TorchVNNLIB`.

    :param type: Detected VNN-LIB structural type.
    :param used_fast: Whether the fast-type processor was used.
    :param time: Total conversion time in seconds.
    :param n_inputs: Number of input variables.
    :param n_outputs: Number of output variables.
    :param output_format: Backend format (``"torch"`` or ``"numpy"``).
    :param fallback_reason: Reason for AST fallback, if any.
    """

    type: str
    used_fast: bool
    time: float
    n_inputs: int
    n_outputs: int
    output_format: str
    fallback_reason: str | None


def _save_property_file(
    or_properties: list[tuple[TensorLike, list[TensorLike]]],
    or_folder_path: str,
    backend: Backend,
) -> None:
    """Save one OR group to per-property files inside ``or_folder_path``.

    :param or_properties: Properties as ``(input_bounds, output_constraints)`` tuples.
    :param or_folder_path: Destination directory.
    :param backend: Backend used for serialization.
    """
    for j, and_property in enumerate(or_properties):
        _logger.info(f"Converting {j + 1}/{len(or_properties)} properties")
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
) -> None:
    """Write all OR groups to ``or_group_<i>/`` subfolders of the target.

    :param and_properties: One inner list per OR group.
    :param target_folder_path: Output directory; defaults to vnnlib stem.
    :param vnnlib_path: Source VNN-LIB path (used to derive default target).
    :param backend: Backend used for serialization.
    """
    if target_folder_path is None:
        target_folder_path = str(Path(vnnlib_path).with_suffix(""))

    target_path = Path(target_folder_path)
    target_path.mkdir(parents=True, exist_ok=True)

    or_folder_paths = []
    for i in range(len(and_properties)):
        or_folder_path = str(target_path / f"or_group_{i}")
        or_folder_paths.append(or_folder_path)
        Path(or_folder_path).mkdir(parents=True, exist_ok=True)

    save = partial(_save_property_file, backend=backend)
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

        :param verbose: Print detailed timing information.

        :param use_parallel: Use parallel processing where possible.

        :param detect_fast_type: Use optimized type-specific processors.

        :param output_format: Output format ('torch' for .pth or 'numpy' for .npz).

        """
        self.verbose = verbose
        if verbose:
            _enable_verbose()
        self.use_parallel = use_parallel
        self.detect_fast_type = detect_fast_type
        self.output_format = output_format
        self.backend = get_backend(cast(Literal["torch", "numpy"], output_format))
        self.conversion_stats: dict[str, ConversionStats] = {}

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
        _logger.info(f"  Type1 processing: {time.perf_counter() - t:.4f}s")
        return and_properties

    def _process_type234(
        self,
        processor_func: Callable[..., list[list[tuple[TensorLike, list[TensorLike]]]]],
        lines: list[str],
        n_inputs: int,
        n_outputs: int,
        use_parsed_data: bool,
    ) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
        """Dispatch to a TYPE2/3/4/5 processor; caller logs elapsed time."""
        if use_parsed_data:
            parsed_data = parse_simple_patterns(lines, verbose=self.verbose)
            return processor_func(
                lines,
                n_inputs,
                n_outputs,
                self.backend,
                verbose=self.verbose,
                parsed_data=parsed_data,
            )
        return processor_func(lines, n_inputs, n_outputs, self.backend, verbose=self.verbose)

    # Dispatch table: VNNLIBType -> (processor, needs_parse_simple_patterns).
    # ``TYPE1`` is special-cased (it owns its parse_simple_patterns call inside
    # ``_process_type1``); the rest share the ``_process_type234`` scaffolding.
    _TYPE234_DISPATCH: ClassVar[dict[VNNLIBType, tuple[Callable[..., Any], bool]]] = {
        VNNLIBType.TYPE2: (process_type2, True),
        VNNLIBType.TYPE3: (process_type3, True),
        VNNLIBType.TYPE4: (process_type4, False),
        VNNLIBType.TYPE5: (process_type5, False),
    }

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

        handler = self._TYPE234_DISPATCH.get(vnnlib_type)
        if handler is not None:
            processor_func, use_parsed_data = handler
            result = self._process_type234(
                processor_func, lines, n_inputs, n_outputs, use_parsed_data
            )
            _logger.info(f"  {vnnlib_type.name} processing: {time.perf_counter() - t:.4f}s")
            return result

        _logger.info("Complex structure detected, using AST processing")
        return None

    def _process_ast(
        self, lines: list[str], n_inputs: int, n_outputs: int
    ) -> list[list[tuple[TensorLike, list[TensorLike]]]]:
        """Process using AST pipeline (tokenize, parse, optimize, flatten, convert)."""
        t = time.perf_counter()
        tokens_list = tokenize(lines, verbose=self.verbose, use_parallel=self.use_parallel)
        _logger.info(f"  Tokenization: {time.perf_counter() - t:.4f}s")

        t = time.perf_counter()
        expr = parse(tokens_list, verbose=self.verbose, use_parallel=self.use_parallel)
        if not isinstance(expr, And | Or):
            raise ValueError(f"Expected And or Or expression, got {type(expr).__name__}: {expr}")
        nary_expr = cast(And | Or, expr)
        _logger.info(
            f"  Parsing: {len(nary_expr.args)} expressions, {time.perf_counter() - t:.4f}s"
        )

        t = time.perf_counter()
        expr = optimize(expr, verbose=self.verbose, use_parallel=self.use_parallel)
        _logger.info(f"  Optimization: {time.perf_counter() - t:.4f}s")

        t = time.perf_counter()
        expr = flatten(expr)
        _logger.info(f"  Flattening: {time.perf_counter() - t:.4f}s")

        t = time.perf_counter()
        and_properties = convert_to_tensor(
            expr,
            n_inputs,
            n_outputs,
            self.backend,
            verbose=self.verbose,
            use_parallel=self.use_parallel,
        )
        elapsed = time.perf_counter() - t
        num_props = len(and_properties)
        _logger.info(f"  Tensor conversion: {num_props} AND properties, {elapsed:.4f}s")
        return and_properties

    def convert(self, vnnlib_path: str, target_folder_path: str | None = None) -> None:
        """Convert VNN-LIB file to tensor data.

        :param vnnlib_path: Path to .vnnlib file.

        :param target_folder_path: Output directory path.

        """
        _logger.info(f"TorchVNNLIB: converting {vnnlib_path}")
        _logger.info(f"  Output format: {self.output_format}")
        t_start = time.perf_counter()

        t = time.perf_counter()
        vnnlib_file = Path(vnnlib_path)
        with vnnlib_file.open() as f:
            lines = f.readlines()
        _logger.info(f"  Read file: {time.perf_counter() - t:.4f}s")

        t = time.perf_counter()
        lines, n_inputs, n_outputs = preprocess_vnnlib(lines)
        elapsed = time.perf_counter() - t
        _logger.info(f"  Preprocessing: {n_inputs} inputs, {n_outputs} outputs, {elapsed:.4f}s")

        use_type_processor = False
        detected_type = VNNLIBType.COMPLEX
        fallback_reason: str | None = None

        if self.detect_fast_type:
            vnnlib_type = fast_detect_type(lines, verbose=self.verbose)
            detected_type = vnnlib_type

            and_properties = None
            try:
                and_properties = self._process_by_type(vnnlib_type, lines, n_inputs, n_outputs)
                use_type_processor = and_properties is not None
            except Exception as e:  # noqa: BLE001
                fallback_reason = str(e)[:100]
                _logger.warning(f"  Type processor failed, fallback to AST: {fallback_reason}")
                use_type_processor = False

        if not use_type_processor:
            and_properties = self._process_ast(lines, n_inputs, n_outputs)

        if and_properties is None:
            raise RuntimeError(
                f"Internal error: neither fast-path nor AST produced properties for {vnnlib_path}"
            )
        t = time.perf_counter()
        _write_property(and_properties, target_folder_path, vnnlib_path, self.backend)
        _logger.info(f"  Writing to disk: {time.perf_counter() - t:.4f}s")

        total_time = time.perf_counter() - t_start
        _logger.info(f"  Total: {total_time:.4f}s")

        self.conversion_stats[vnnlib_path] = {
            "type": str(detected_type),
            "used_fast": use_type_processor,
            "time": total_time,
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "output_format": self.output_format,
            "fallback_reason": fallback_reason,
        }
