# TorchVNNLIB

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/torchvnnlib/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/torchvnnlib/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/ZhongkuiMa/torchvnnlib/branch/main/graph/badge.svg)](https://codecov.io/gh/ZhongkuiMa/torchvnnlib)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

Convert VNN-LIB verification property files (`.vnnlib`) into PyTorch tensors (`.pth`) or NumPy arrays (`.npz`) for repeated loading in neural network verification pipelines.

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
converter.convert("property.vnnlib", target_folder_path="output")

data = torch.load("output/or_group_0/sub_prop_0.pth")
print(data["input"].shape)   # (n_inputs, 2) — per-variable [lower, upper] bounds
print(len(data["output"]))   # list of (n_constraints, 1+n_vars) inequality tensors
```

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

Each output file contains two keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `"input"` | `(n_inputs, 2)` | Per-variable `[lower, upper]` bounds |
| `"output"` | list of `(n_constraints, 1+n_vars)` | Linear inequalities: `b + Ax >= 0` |

OR-disjunctions produce one subdirectory per group:

```
output/
├── or_group_0/
│   ├── sub_prop_0.pth
│   └── sub_prop_1.pth
└── or_group_1/
    └── sub_prop_0.pth
```

## Type Detection

With `detect_fast_type=True`, the converter identifies common VNN-LIB patterns and selects optimized processors:

| Type | Pattern |
|------|---------|
| 1 | Simple input bounds + simple output constraints |
| 2 | Simple input bounds + OR-grouped output constraints |
| 3 | OR-grouped input bounds + simple output constraints |
| 4 | OR-grouped input bounds + OR-grouped output constraints |
| 5 | Top-level OR wrapping complete properties |
| Other | Non-standard — falls back to AST-based parsing |

Set `detect_fast_type=False` to force AST-based parsing for all files.

## Limitations

- Input variables must have both lower and upper bounds (closed intervals)
- Only linear output constraints are supported
- Variables must follow the `X_i` / `Y_i` naming convention
- No if-then-else control flow

## Project Structure

```
src/torchvnnlib/
├── _torchvnnlib.py      # TorchVNNLIB class (main entry point)
├── _backend.py          # Backend abstraction (torch/numpy)
├── _to_tensor.py        # AST-to-tensor conversion
├── ast/                 # Parser pipeline: tokenize, parse, optimize, flatten
└── fast_type/           # Type-specific optimized processors (Types 1-5)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
