---
type: DESCRIPTION
note: "Descriptive. Mirrors current code; update to follow code changes. < functional code."
---

> **This file IS**: the contributor setup and workflow guide specific to torchvnnlib. **It is NOT**: project-wide conventions — see root CONTRIBUTING.md.

# Contributing to TorchVNNLIB

Shared conventions (imports, formatting, docstrings) are in the root [CONTRIBUTING.md](../CONTRIBUTING.md). This file covers torchvnnlib-specific workflow.

## Setup

```bash
cd torchvnnlib
pip install -e ".[dev]"
pre-commit install
```

## Checks

```bash
pre-commit run --all-files  # lint, format, type-check
pytest tests/ -v            # tests
```

## Workflow

1. Create branch from `main`
2. Make changes
3. Run checks (above)
4. Commit and push

## Adding a Fast Type Processor

Most contributions add a new type-specific processor (Type N):

1. Create `src/torchvnnlib/fast_type/_typeN_processor.py` with a `process_typeN()` function
2. Add detection logic in `src/torchvnnlib/fast_type/_fast_type_detect.py` (extend `VNNLIBType` enum and `fast_detect_type()`)
3. Register in `src/torchvnnlib/_torchvnnlib.py` (`_process_by_type` method)
4. Export from `src/torchvnnlib/fast_type/__init__.py`
5. Add tests in `tests/test_units/`
6. Verify: `pytest tests/ -v`

## Adding an AST Operation

1. Define expression class in `src/torchvnnlib/ast/_expr.py`
2. Add tokenization in `_tokenize.py`, parsing in `_parse.py`
3. Handle optimization in `_optimize.py` and flattening in `_flatten.py`
4. Add tensor conversion in `src/torchvnnlib/_to_tensor.py`
5. Export from `src/torchvnnlib/ast/__init__.py`

## Constraints

| Rule | Details |
|------|---------|
| Absolute imports only | `from torchvnnlib.ast._expr import ...` (no relative) |
| `__docformat__` + `__all__` | Required in every module |
| Private modules | Prefix with `_` (e.g., `_type1_processor.py`) |
| Backend abstraction | All tensor ops go through `Backend` interface, never raw torch/numpy |
| Type hints | Required on all public functions |
| McCabe complexity <= 10 | Enforced by ruff C90 |
