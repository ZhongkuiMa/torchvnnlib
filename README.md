# TorchVNNLIB

[![Build Status](https://github.com/ZhongkuiMa/torchvnnlib/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/torchvnnlib/actions/workflows/unit-tests.yml)

Parse VNN-LIB verification properties into package-owned values. The legacy
VNN-LIB 1.0 API produces ordered PyTorch tensors or NumPy arrays; the 2.0 API
produces an immutable typed query representation backed by the official parser.

## Installation

```bash
git clone https://github.com/ZhongkuiMa/torchvnnlib.git
cd torchvnnlib
pip install -e ".[dev]"
```

Requirements: Python 3.11+, numpy. Optional: torch (install with `pip install -e ".[torch]"`).

## Quick Start

```python
from torchvnnlib import TorchVNNLIB
import torch

converter = TorchVNNLIB(output_format="torch", detect_fast_type=True)
properties = converter.parse("property.vnnlib")
input_bounds, output_constraints = properties[0][0]
print(input_bounds.shape)          # (n_inputs, 2)
print(len(output_constraints))     # inequality tensors
```

VNN-LIB 2.0 uses a separate API because named networks, shaped tensors, and
general query expressions cannot be represented by the legacy tensor tuple:

```python
version = TorchVNNLIB.detect_version("query.vnnlib.gz")
query = TorchVNNLIB.parse_query("query.vnnlib.gz")
print(version, query.networks, query.branches, query.capabilities)
```

`parse_query()` accepts plain or gzip-compressed input and returns only frozen
TorchVNNLIB dataclasses; objects from the upstream `vnnlib` parser are not
exposed. A version declaration must be the first non-comment statement;
malformed or late declarations are rejected. Query execution and network
binding belong to the consuming verifier.

For NumPy output, use `output_format="numpy"` and `numpy.load(..., allow_pickle=True)`.

## API

```python
TorchVNNLIB(
    verbose=False,          # print timing info
    use_parallel=False,     # multi-threaded conversion
    detect_fast_type=True,  # use optimized type-specific processors
    output_format="torch",  # "torch" (.pth) or "numpy" (.npz)
)

converter.convert(
    vnnlib_path,              # path to .vnnlib file
    target_folder_path=None,  # output directory (default: vnnlib stem)
)
```

`parse()` is the canonical API and preserves source order without filesystem
round trips. `convert()` delegates to `parse()` and writes the same tensors
when persistent artifacts are required.

Each output file contains two keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `"input"` | `(n_inputs, 2)` | Per-variable `[lower, upper]` bounds |
| `"output"` | list of `(n_constraints, 1+n_vars)` | Linear inequalities: `b + Ax >= 0` |

OR-disjunctions produce one subdirectory per group.

## Type Detection

With `detect_fast_type=True`, legacy input selects optimized processors for the
five common bound/disjunction shapes and falls back to the AST otherwise. Set
it to `False` to force AST parsing.

## Limitations

The legacy `parse()` API has these restrictions:

- Input variables must have both lower and upper bounds (closed intervals)
- Only linear output constraints are supported
- Variables must follow the `X_i` / `Y_i` naming convention
- No if-then-else control flow

`parse_query()` parses VNN-LIB 2.0 syntax but does not itself execute networks
or claim that a downstream verifier supports every parsed capability.

Implementation structure and contribution rules live in
[ARCHITECTURE.md](ARCHITECTURE.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
