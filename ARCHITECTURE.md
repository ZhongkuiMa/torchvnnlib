---
type: DESCRIPTION
note: "Descriptive. Mirrors current code; update to follow code changes. < functional code."
---

> **This file IS**: the package tree, module responsibilities, and data flow of torchvnnlib. **It is NOT**: a user guide or API reference — see README.md.

# TorchVNNLIB Architecture

VNN-LIB to tensor converter with two processing paths: fast type-specific regex processors (Types 1-5) and a general AST pipeline.

## Package Tree

```
src/torchvnnlib/
├── _torchvnnlib.py    # TorchVNNLIB class — entry point, dispatches to fast or AST path
├── _backend.py        # Backend ABC — abstracts torch/numpy ops (modify when: adding backend)
├── _to_tensor.py      # AST-to-tensor conversion (modify when: changing output format)
├── ast/               # General parser — tokenize, parse, optimize, flatten
│   ├── _expr.py       # Expression node classes (Var, Cst, Add, Mul, And, Or, Leq, Geq...)
│   ├── _tokenize.py   # S-expression tokenizer
│   ├── _parse.py      # Token stream to AST
│   ├── _optimize.py   # AST simplification passes
│   ├── _flatten.py    # Normalize nested And/Or into flat structure
│   ├── _preprocess.py # Variable declaration extraction (n_inputs, n_outputs)
│   └── _print.py      # AST pretty-printer (debugging)
└── fast_type/         # Regex-based processors for common patterns (10-100x faster)
    ├── _fast_type_detect.py  # VNNLIBType enum + detection logic
    ├── _type1_processor.py   # Simple bounds + simple output constraints
    ├── _type2_processor.py   # Simple bounds + OR-grouped output
    ├── _type3_processor.py   # OR-grouped bounds + simple output
    ├── _type4_processor.py   # OR-grouped bounds + OR-grouped output
    ├── _type5_processor.py   # Top-level OR wrapping complete properties
    └── _utils.py             # Shared regex patterns and parsing helpers
```

## Modification Map

| Intent | Primary Modify | Follow-ups | Avoid | Constraints | Failure Signal |
|--------|---------------|------------|-------|-------------|----------------|
| Add fast type processor | `fast_type/_typeN_processor.py` (new) | `_fast_type_detect.py` (enum + detection), `_torchvnnlib.py` (dispatch), `fast_type/__init__.py` (export) | `ast/` | Must use `Backend` interface (enforced) | `pytest tests/` fails |
| Add AST expression type | `ast/_expr.py` | `_tokenize.py`, `_parse.py`, `_optimize.py`, `_flatten.py`, `_to_tensor.py`, `ast/__init__.py` | `fast_type/` | Must be immutable dataclass (observed) | `pytest tests/` fails |
| Add output backend | `_backend.py` | `_torchvnnlib.py` (get_backend routing) | `ast/`, `fast_type/` | Must implement `Backend` ABC (enforced) | `mypy` type errors |
| Change output structure | `_to_tensor.py` | `fast_type/_type*_processor.py` (must produce same structure) | `ast/` | Both paths must produce identical output (observed) | Tests comparing fast vs AST fail |
| Fix parsing bug | `ast/_parse.py` or `ast/_tokenize.py` | None | `fast_type/` (separate code path) | — | `pytest tests/` fails |

## Dependency Rules

| Rule | Source | Failure |
|------|--------|---------|
| `fast_type/` imports only from `_backend.py` and `_utils.py` (not `ast/`) | (observed) | Coupling fast path to AST defeats performance purpose |
| `ast/` modules import only from `_expr.py` and each other | (observed) | Circular imports at runtime |
| No raw `torch.*` or `np.*` in processors — use `Backend` | (enforced) | Backend switching breaks |
| Absolute imports only (no relative) | (enforced) | `ruff check` TID failure |

## Common Mistakes

| Mistake | Detection Signal | Fix |
|---------|-----------------|-----|
| Using `torch.tensor()` directly in a processor | `mypy` or fails with numpy backend | Use `backend.tensor()` |
| Fast processor output shape differs from AST path | Benchmark tests fail on specific .vnnlib files | Match `list[list[tuple[TensorLike, list[TensorLike]]]]` structure |
| Adding export to `__init__.py` without `__all__` entry | `ruff check` F401 (unused import in `__init__`) | Add to both import and `__all__` |

## Processing Flow

```
.vnnlib file
    │
    ├─► preprocess_vnnlib() → lines, n_inputs, n_outputs
    │
    ├─► fast_detect_type() → VNNLIBType
    │       │
    │       ├─► TYPE1-5: process_typeN() → and_properties
    │       │
    │       └─► COMPLEX: fallback to AST
    │
    └─► AST path: tokenize → parse → optimize → flatten → convert_to_tensor
                                                              │
                                                              ▼
                                                        and_properties
    │
    └─► _write_property() → .pth/.npz files on disk
```

## Conventions

- All modules: `__docformat__ = "restructuredtext"` and `__all__` required
- Private modules prefixed with `_` (public API only through `__init__.py`)
- Expression nodes in `ast/_expr.py` are frozen dataclasses
- `TensorLike` type alias used everywhere (resolves to `torch.Tensor | np.ndarray`)

## Related Documents

- [README.md](README.md) — usage, API, output format
- [CONTRIBUTING.md](CONTRIBUTING.md) — development workflow
- Root [ARCHITECTURE.md](../ARCHITECTURE.md) — rover system architecture
