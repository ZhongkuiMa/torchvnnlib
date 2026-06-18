---
type: LAW
note: "Normative. Code obeys this; change only via deliberate revision. Greater authority than functional code."
---

> **Law**. Conventions/rules for torchvnnlib. Code obeys this. Change via deliberate revision.

# Torchvnnlib Conventions

This file defines style and documentation conventions for the torchvnnlib package.
Use it as a **checklist** — when writing or reviewing code, check each item below
one by one.

---

## 1. Module Docstrings

Every `.py` file begins with a module docstring.

### Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 1.1 | **First line**: short summary of the module's purpose (one sentence) | ☐ |
| 1.2 | **Extended description** (optional): 1-2 paragraphs after a blank line, covering architecture or usage | ☐ |
| 1.3 | **Format**: ReST plain text, no `:param:` or `:return:` tags at module level | ☐ |
| 1.4 | Always followed by `__docformat__ = "restructuredtext"` | ☐ |
| 1.5 | **No author, date, or version lines** — git history is authoritative | ☐ |
| 1.6 | **No non-ASCII characters** in docstrings — use ASCII equivalents | ☐ |

### Patterns

| File type | Style | Example |
|-----------|-------|---------|
| Entry point (`_torchvnnlib.py`) | One line summarizing the converter | `"""VNN-LIB to tensor conversion engine."""` |
| AST module (`ast/_expr.py`) | One line | `"""AST expression node classes."""` |
| Backend module (`_backend.py`) | One line | `"""Backend implementations for tensor operations."""` |
| fast_type module (`fast_type/_utils.py`) | One line | `"""Utility functions for fast type detection."""` |
| `__init__.py` | Summary of package with listed exports | `"""VNN-LIB format conversion..."""` |

---

## 2. Class Docstrings

### 2.1 Structure

```python
class Expr(ABC):
    """
    Abstract base class for VNN-LIB expression AST nodes.

    All expression types (constants, variables, binary ops, n-ary ops)
    inherit from this class and implement evaluation and printing.

    :param <attr>: Description of class-level attributes.
    """
```

### 2.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 2.1 | **First line**: describes what the class represents, ends with period | ☐ |
| 2.2 | Constructor parameters documented in class docstring with `:param name:` — capitalized, ends with period | ☐ |
| 2.3 | `:raises ValueError:` in class docstring when `__init__` validates input format | ☐ |
| 2.4 | ABC methods document the contract subclasses must fulfill | ☐ |
| 2.5 | No docstring on `__init__` of simple data-holding classes (class docstring covers it) | ☐ |
| 2.6 | AST node classes include `__slots__` — document which slots are defined | ☐ |
| 2.7 | `__repr__`, `__eq__`, `__hash__` overrides do not need docstrings (standard Python semantics) | ☐ |

### 2.3 Good examples

```python
class Var(Expr):
    """Variable reference node (``X_i`` for inputs, ``Y_i`` for outputs).

    :param name: Variable name in ``X_<idx>`` or ``Y_<idx>`` format.
    :raises ValueError: If name does not match the expected format.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        if not re.match(r"^[XY]_\d+$", name):
            raise ValueError(f"Invalid variable name: {name}")
        self.name = name
```

```python
class TorchVNNLIB:
    """VNN-LIB to PyTorch tensor converter.

    Parses VNN-LIB format specifications and converts them to
    executable PyTorch tensor operations.

    :param verbose: Whether to print conversion progress and statistics.
    """
```

---

## 3. Method/Function Docstrings

### 3.1 Structure

```python
def parse(self, vnnlib_str: str) -> list[Expr]:
    """
    Parse VNN-LIB specification string into AST.

    Tokenizes the input, then builds an abstract syntax tree of
    expression nodes representing the constraints.

    :param vnnlib_str: Raw VNN-LIB specification string.
    :return: List of parsed constraint expressions.
    :raises ParseError: If the VNN-LIB string is malformed.
    """
```

### 3.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 3.1 | **First line**: imperative mood, describes what the method computes, ends with period | ☐ |
| 3.2 | Use `:param name:`, `:return:`, and `:raises ExceptionType:` tags — no `:type:` tags | ☐ |
| 3.3 | `:param` descriptions: **capitalized, end with period**, describe semantics | ☐ |
| 3.4 | `:return` description: **capitalized, end with period** | ☐ |
| 3.5 | `:raises` descriptions: **capitalized, end with period**; describe the trigger condition | ☐ |
| 3.6 | Private helpers may use a single-line docstring without `:param:` tags | ☐ |
| 3.7 | Abstract methods must have docstrings describing the contract | ☐ |
| 3.8 | **No non-ASCII characters** in docstrings | ☐ |

---

## 4. Inline Comments

| # | Rule | Pass/Fail |
|---|------|-----------|
| 4.1 | Comment **why**, not what — the code already says what | ☐ |
| 4.2 | Only add comments when the reasoning is non-obvious (parsing edge cases, optimization rationale) | ☐ |
| 4.3 | AST construction comments: explain non-trivial grammar reductions or precedence handling | ☐ |
| 4.4 | No commented-out code — delete it | ☐ |
| 4.5 | `# TODO:` comments require an associated issue reference (enforced by ruff TD001) | ☐ |

---

## 5. Naming Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 5.1 | **Classes**: PascalCase — `Expr`, `Cst`, `Var`, `BinaryOp`, `NaryOp`, `Add`, `And`, `Or`, `Backend`, `TorchBackend`, `NumpyBackend`, `TorchVNNLIB` | ☐ |
| 5.2 | **AST operator class names**: match VNN-LIB operator names — `Add`, `Sub`, `Mul`, `Div`, `And`, `Or`, `Eq`, `Leq`, `Geq` | ☐ |
| 5.3 | **Functions/methods**: snake_case — `parse`, `flatten`, `optimize`, `convert_to_tensor`, `preprocess_vnnlib` | ☐ |
| 5.4 | **Private functions**: `_` prefix — `_tokenize`, `_preprocess`, `_check_inputs` | ☐ |
| 5.5 | **Private modules**: `_` prefix — `_torchvnnlib.py`, `_backend.py`, `_to_tensor.py`, `_expr.py`, `_parse.py`, `_tokenize.py`, `_flatten.py`, `_optimize.py` | ☐ |
| 5.6 | **Constants**: UPPER_CASE — `_MAX_ITER`, `_CONVERGE_TOL` | ☐ |
| 5.7 | **Enum members**: UPPER_CASE — `TYPE1`, `TYPE2`, `TYPE3`, `TYPE4`, `TYPE5` | ☐ |
| 5.8 | **Processing modules**: `_type<N>_processor.py` — `_type1_processor.py`, `_type2_processor.py` | ☐ |

---

## 6. Code Style

| # | Rule | Pass/Fail |
|---|------|-----------|
| 6.1 | **100-char line length** (enforced by ruff) | ☐ |
| 6.2 | **Double quotes** for strings and docstrings | ☐ |
| 6.3 | **Absolute imports only** — `from torchvnnlib.ast._expr import Var` | ☐ |
| 6.4 | `__docformat__ = "restructuredtext"` after module docstring, before imports | ☐ |
| 6.5 | `__all__` in every source module, alphabetically sorted | ☐ |
| 6.6 | **Import order**: stdlib → third-party (`numpy`, `torch` [optional]) → first-party (`torchvnnlib.*`). Groups separated by blank lines. | ☐ |
| 6.7 | `import numpy as np`; `from numpy import ndarray` (for type annotations) | ☐ |
| 6.8 | **McCabe complexity ≤ 10** (enforced by ruff C90) | ☐ |
| 6.9 | **Only import what you use** — clean up unused imports (enforced by ruff F401) | ☐ |
| 6.10 | **No string annotations** when type is already imported | ☐ |
| 6.11 | `from __future__ import annotations` at top of files using forward references | ☐ |

---

## 7. AST Node Class Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 7.1 | AST nodes inherit from `Expr` (abstract base) | ☐ |
| 7.2 | Leaf nodes (`Cst`, `Var`) store a single value | ☐ |
| 7.3 | `BinaryOp` nodes store `left: Expr` and `right: Expr` | ☐ |
| 7.4 | `NaryOp` nodes store `args: list[Expr]` | ☐ |
| 7.5 | Concrete operator classes (`Add`, `And`, `Or`) inherit from `BinaryOp` or `NaryOp` | ☐ |
| 7.6 | All AST nodes define `__repr__` returning valid VNN-LIB syntax for round-trip verification | ☐ |
| 7.7 | `__eq__` and `__hash__` based on structural equality (node type + children). Concrete operators may delegate to parent `__hash__` rather than redefining identically | ☐ |
| 7.8 | `NaryOp` implements `__iter__` delegating to `self.args` for `for sub_expr in nary_expr` syntax | ☐ |
| 7.9 | `Expr` base class caches `_has_input_vars` / `_has_output_vars` via lazy evaluation; `Var` sets these directly | ☐ |

---

## 8. Backend ABC Conventions

The `Backend` ABC defines a **storage/serialization** interface — not an arithmetic
interface. Concrete backends (`TorchBackend`, `NumpyBackend`) provide tensor creation,
saving, and stacking operations used by the conversion pipeline.

| # | Rule | Pass/Fail |
|---|------|-----------|
| 8.1 | `Backend` is an ABC defining tensor creation and serialization methods | ☐ |
| 8.2 | `TorchBackend` implements ops using `torch.*` functions (guarded by `try/except ImportError`) | ☐ |
| 8.3 | `NumpyBackend` implements ops using `numpy.*` functions (always available) | ☐ |
| 8.4 | Backend methods: `zeros(shape, dtype)`, `full(shape, fill_value, dtype)`, `tensor(data)`, `stack(tensors)`, `isnan(tensor)`, `where(condition, x, y)`, `save(path, tensor)`, `name`, `file_extension` | ☐ |
| 8.5 | `TensorLike` type alias for backend-agnostic tensor type (`torch.Tensor | np.ndarray[Any, Any]`) | ☐ |
| 8.6 | All backend methods are stateless — pure functions of their inputs | ☐ |
| 8.7 | `get_backend(name: str) -> Backend` factory function dispatches by name (`"torch"`, `"numpy"`); raises `ValueError` for unknown names, `ImportError` if torch is requested but not installed | ☐ |

---

## 9. fast_type Processor Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 9.1 | Each type (1-5) has its own processor module: `_type<N>_processor.py` | ☐ |
| 9.2 | Processors are standalone — each handles one VNN-LIB constraint type | ☐ |
| 9.3 | `_fast_type_detect.py` detects the VNN-LIB type (returns `VNNLIBType` enum). Dispatch to processors happens externally in the calling code | ☐ |
| 9.4 | Shared utilities in `fast_type/_utils.py` — common tensor operations used by 2+ processors | ☐ |
| 9.5 | `VNNLIBType` enum in `fast_type/_enums.py` defines the 5 type constants | ☐ |
| 9.6 | `@unique` decorator on the `VNNLIBType` enum | ☐ |

---

## 10. Optional Dependency Handling

| # | Rule | Pass/Fail |
|---|------|-----------|
| 10.1 | torch is optional — imports guarded with `try/except ImportError` | ☐ |
| 10.2 | `if TYPE_CHECKING:` block for torch imports used only for type annotations | ☐ |
| 10.3 | Numpy is required — imported at module level without guards | ☐ |
| 10.4 | Fallback behavior when torch is unavailable: operations that require torch raise `ImportError` with a clear message | ☐ |
| 10.5 | Tests use `pytest.importorskip("torch")` for torch-dependent tests | ☐ |

---

## 11. TorchVNNLIB Class Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 11.1 | Constructor takes `verbose: bool = False, use_parallel: bool = True` | ☐ |
| 11.2 | Main entry point: `convert(vnnlib_path: str, ...)` — takes a file path, reads the file internally, then runs the pipeline | ☐ |
| 11.3 | `conversion_stats: dict` attribute tracks per-stage timing and resource metrics | ☐ |
| 11.4 | Conversion flow: preprocess → type-detect → [fast-path or tokenize → parse → optimize → flatten → convert_to_tensor]. Type detection happens BEFORE parsing (not after), enabling fast-path skip for simple constraint types | ☐ |
| 11.5 | Each conversion stage has a corresponding method or delegate module | ☐ |
| 11.6 | ThreadPoolExecutor used for parallel processing when `use_parallel=True`; each stage timed with `time.perf_counter()` when `verbose=True` | ☐ |

---

## 12. Parallel Processing and Verbose Timing

These cross-cutting patterns appear in most pipeline stages.

| # | Rule | Pass/Fail |
|---|------|-----------|
| 12.1 | Functions accept `verbose: bool = False` and `use_parallel: bool = True` keyword arguments | ☐ |
| 12.2 | When `use_parallel=True`, use `concurrent.futures.ThreadPoolExecutor` for parallel work distribution | ☐ |
| 12.3 | Timing pattern: `t = time.perf_counter() if verbose else None` at start, `if verbose: print(f"... {time.perf_counter() - t:.4f}s")` at end | ☐ |
| 12.4 | Timing messages include the stage name and elapsed seconds with 4 decimal places | ☐ |
| 12.5 | `typing.cast()` used for type narrowing of AST nodes — `cast(And, expr)`, `cast(Or, expr)` — when the type is known from context but mypy cannot infer it | ☐ |

---

## 13. Test Style

### 13.1 Directory Layout

```
tests/
├── test_arch/                 # architecture/import enforcement
├── test_benchmarks/           # integration tests (opt-in)
└── test_units/
    ├── test_ast/
    │   ├── __init__.py
    │   └── test_<concern>.py
    ├── test_fast_type/
    │   ├── __init__.py
    │   └── test_<concern>.py
    └── test_torchvnnlib/
        ├── __init__.py
        └── test_<concern>.py
```

### 13.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 13.1 | **Test file naming**: `test_<concern>.py` — `test_expr.py`, `test_parse.py`, `test_backend.py` | ☐ |
| 13.2 | `__init__.py` at leaf `test_<pkg>/` level only (collision avoidance) | ☐ |
| 13.3 | **No pytest markers** except `@pytest.mark.parametrize` | ☐ |
| 13.4 | `pytest.importorskip("torch")` at the top of torch-dependent test files | ☐ |
| 13.5 | AST round-trip tests: parse → repr → re-parse, verify structural equality | ☐ |
| 13.6 | Backend tests: parametrize over `[TorchBackend(), NumpyBackend()]` for cross-backend consistency | ☐ |
| 13.7 | VNN-LIB fixtures defined as string constants in test files or conftest | ☐ |
| 13.8 | Test module docstrings: 1-3 lines max summarizing what the file validates | ☐ |
| 13.9 | **Default test suite**: `pytest` runs `tests/test_units/` by default. Benchmark/integration tests are opt-in | ☐ |
| 13.10 | **No `@pytest.mark.skip`** in committed code — use conditional early return with `[REVIEW]` comment for unavailable hardware | ☐ |

---

## 14. Enum Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 14.1 | **IntEnum with `@unique`**: All enums use `IntEnum` with `@unique` decorator (e.g., `VNNLIBType` in `fast_type/_enums.py`) | ☐ |
| 14.2 | **Placement**: Subpackage-local enums in `<subfolder>/_enums.py` | ☐ |
| 14.3 | **Class naming**: PascalCase with categorical suffix — `Type`, `Mode`, `Status`, `Strategy`. Never suffix with `Enum` | ☐ |
| 14.4 | **Member naming**: `UPPER_SNAKE_CASE`, 1-3 words. Must be unique within the class | ☐ |
| 14.5 | **Custom `__repr__`**: IntEnum classes define `__repr__` returning `f"{self.name}"` | ☐ |
| 14.6 | **Member docstrings**: Every enum member has a one-line ReST docstring after the value assignment | ☐ |
| 14.7 | **Module boilerplate**: `__docformat__ = "restructuredtext"`, `__all__` alphabetically sorted listing enum classes | ☐ |

---

## 15. Constants Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 15.1 | **Naming**: `UPPER_SNAKE_CASE`, 2-4 words. Use prefixes (`DEFAULT_`, `MAX_`, `MIN_`) and suffixes for clarity | ☐ |
| 15.2 | **Scope levels**: Place at narrowest scope — function-level → file-level → subfolder `_constants.py` → package-level. Promote when a second consumer at broader scope appears | ☐ |
| 15.3 | **Extraction trigger**: Extract a literal when it appears 2+ times. Never duplicate a constant across files | ☐ |
| 15.4 | **When NOT to extract**: Self-documenting single-use values, test data, function defaults already named by the parameter, `0`/`1`/`-1` for indexing | ☐ |
| 15.5 | **Type annotations**: Annotate only when the type is not obvious from the literal | ☐ |
| 15.6 | **Frozen collections**: Use `frozenset` or `tuple` for constant collections — never mutable `list` or `set` | ☐ |

---

## 16. Architecture Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 16.1 | **`__init__.py` facade pattern**: import from private `_*.py` modules, re-export public names via `__all__` | ☐ |
| 16.2 | **ast/ subpackage**: depends only on stdlib (no numpy, no torch) | ☐ |
| 16.3 | **fast_type/ subpackage**: depends on numpy and ast. Type1 processor may import from root modules (`_backend`, `_to_tensor`) for complex expressions; Type2-5 import only from `_utils` and `_backend` | ☐ |
| 16.4 | **Root modules** (`_backend.py`, `_to_tensor.py`, `_torchvnnlib.py`): depend on ast and fast_type | ☐ |
| 16.5 | **Dependency flow**: `ast/ → fast_type/ → root modules` (no circular imports). Lazy imports inside function bodies permitted to break circular deps between `_fast_type_detect` and processors | ☐ |

---

## 17. Logging Conventions

Pipeline tool: use `logging` package with `_enable_verbose()` helper.

### Setup

```python
import logging

_logger = logging.getLogger(__name__)


def _enable_verbose() -> None:
    """Configure package-level logger for console output."""
    pkg_logger = logging.getLogger("torchvnnlib")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.DEBUG)


class TorchVNNLIB:
    def __init__(self, verbose: bool = False, ...):
        if verbose:
            _enable_verbose()
```

### Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 17.1 | **One `_enable_verbose()` per package** — defined in main orchestrator module, configures package-level logger | ☐ |
| 17.2 | **Module-level `_logger`** — every file with output uses `_logger = logging.getLogger(__name__)` | ☐ |
| 17.3 | **Direct `_logger.info(f"...")` calls** — no `isEnabledFor` guards, no `%`-formatting | ☐ |
| 17.4 | **f-strings for all log messages** — `_logger.info(f"  Stage: {elapsed:.4f}s")` | ☐ |
| 17.5 | **Output format**: first line `ToolName: action`, stage lines `  Stage: description (0.XXXXs)`, final line `  Total: 0.XXXXs` | ☐ |
| 17.6 | **`warnings.warn()` for recoverable errors** — never `logger.warning()`. Warnings independent of verbose flag | ☐ |
| 17.7 | **`raise ValueError/RuntimeError` for fatal errors** — never `logger.error()` | ☐ |
| 17.8 | **No `print()` for diagnostic output** — except `ast/_print.py` debug utility which is intentional | ☐ |
| 17.9 | **Timing**: `t = time.perf_counter()` always called (nanosecond overhead) | ☐ |
