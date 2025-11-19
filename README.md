# TorchVNNLIB

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-required-green.svg)](https://numpy.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20(optional)-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance VNN-LIB converter to PyTorch tensors or NumPy arrays for neural network verification.

## Overview

[VNN-LIB](https://www.vnnlib.org/) is the standard specification format for neural network verification properties, used extensively in [VNN-COMP](https://www.aiverification.org/2025/) and verification research. However, `.vnnlib` files present challenges for modern tensor-based verification tools:

- Slow text parsing
- Inefficient for GPU-accelerated operations
- Cumbersome integration with PyTorch pipelines

**TorchVNNLIB** solves these problems by converting VNN-LIB specifications directly into PyTorch tensors or NumPy arrays, enabling:

- Fast loading via `.pth` (PyTorch) or `.npz` (NumPy) formats
- Native PyTorch integration or lightweight NumPy usage
- GPU-ready constraint representations (PyTorch)
- Standardized property formats
- **No PyTorch dependency required** when using NumPy backend

## Installation

### Requirements

- Python 3.10+
- NumPy (required)
- PyTorch 2.x (optional, only if using `output_format="torch"`)

### Setup

```bash
# Clone or add as submodule
git submodule add <repository-url> torchvnnlib

# Install minimal dependencies (NumPy only)
pip install numpy

# Optional: Install PyTorch for .pth output
pip install torch
```

## Quick Start

### Using NumPy Backend (No PyTorch Required)

```python
from torchvnnlib import TorchVNNLIB

# Initialize converter with NumPy backend
converter = TorchVNNLIB(verbose=True, detect_fast_type=True, output_format="numpy")

# Convert .vnnlib to NumPy arrays
converter.convert("property.vnnlib", target_folder_path="output")

# Load converted arrays
import numpy as np
data = np.load("output/or_group_0/sub_prop_0.npz")

# Access constraints
input_bounds = data["input"]      # Shape: (n_inputs, 2)
output_constraints = data["output"]  # List of arrays
```

### Using PyTorch Backend

```python
from torchvnnlib import TorchVNNLIB

# Initialize converter with PyTorch backend (default)
converter = TorchVNNLIB(verbose=True, detect_fast_type=True, output_format="torch")

# Convert .vnnlib to PyTorch tensors
converter.convert("property.vnnlib", target_folder_path="output")

# Load converted tensors
import torch
data = torch.load("output/or_group_0/sub_prop_0.pth")

# Access constraints
input_bounds = data["input"]      # Shape: (n_inputs, 2)
output_constraints = data["output"]  # List of tensors
```

## Features

### Dual Backend Support

1. **NumPy Backend** - Lightweight, no PyTorch dependency
2. **PyTorch Backend** - Full GPU acceleration support

### Dual Processing Pipelines

1. **Fast Type-Specific Processors** - Optimized for common VNN-LIB patterns (Type1-5)
2. **AST-Based Parser** - Handles complex, non-standard specifications

### Supported VNN-LIB Types

- **Type1**: Simple input bounds and output constraints
- **Type2**: Simple inputs with OR-grouped outputs
- **Type3**: OR-grouped inputs with simple outputs
- **Type4**: OR-grouped inputs and outputs
- **Type5**: Top-level OR wrapping complete properties

### Performance Features

- Parallel processing with ThreadPoolExecutor
- Efficient tensor pre-allocation
- Cached variable indexing
- Type-specific optimizations

## API Reference

### TorchVNNLIB Class

```python
TorchVNNLIB(
    verbose: bool = False,
    use_parallel: bool = False,
    detect_fast_type: bool = True,
    output_format: str = "torch"
)
```

**Parameters:**
- `verbose`: Print detailed timing information
- `use_parallel`: Enable parallel processing
- `detect_fast_type`: Use optimized type-specific processors (recommended)
- `output_format`: Output format - `"torch"` for .pth files or `"numpy"` for .npz files

**Methods:**

#### convert()

```python
converter.convert(
    vnnlib_path: str,
    target_folder_path: str | None = None
)
```

Convert VNN-LIB file to tensor/array data.

**Parameters:**
- `vnnlib_path`: Path to `.vnnlib` file
- `target_folder_path`: Output directory (defaults to `<vnnlib_path>` without extension)

**Output Structure:**
```
output/
├── or_group_0/
│   ├── sub_prop_0.pth (or .npz)
│   ├── sub_prop_1.pth (or .npz)
│   └── ...
├── or_group_1/
│   └── ...
```

Each file contains:
```python
{
    "input": Array,       # Shape: (n_inputs, 2) - [lower, upper] bounds
    "output": list[Array] # List of constraint arrays
}
```

### Output Tensor/Array Format

#### Input Bounds
Shape: `(n_inputs, 2)`
```python
[[lower_0, upper_0],
 [lower_1, upper_1],
 ...]
```

#### Output Constraints
List of arrays, each with shape: `(n_constraints, 1 + n_outputs)` or `(n_constraints, 1 + n_outputs + n_inputs)`

Format: `b + Ax >= 0`
```python
[[b_0, a_00, a_01, ..., a_0n],
 [b_1, a_10, a_11, ..., a_1n],
 ...]
```

Where:
- First column: bias term `b`
- Next `n_outputs` columns: output variable coefficients
- Optional last `n_inputs` columns: input variable coefficients (if constraint involves inputs)

## Examples

### Example 1: Simple Property (Type1) - NumPy

**VNN-LIB:**
```lisp
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

; Simple input bounds
(assert (<= X_0 0.5))
(assert (>= X_0 -0.5))

; Simple output constraint
(assert (<= Y_0 Y_1))
```

**Python:**
```python
from torchvnnlib import TorchVNNLIB
import numpy as np

converter = TorchVNNLIB(output_format="numpy")
converter.convert("property.vnnlib", "output")

data = np.load("output/or_group_0/sub_prop_0.npz")
print(data["input"])   # [[-0.5, 0.5]]
print(data["output"])  # [array([[0.0, -1.0, 1.0]])]
```

### Example 2: Using PyTorch Backend

**Python:**
```python
from torchvnnlib import TorchVNNLIB
import torch

converter = TorchVNNLIB(output_format="torch", verbose=True)
converter.convert("property.vnnlib", "output")

data = torch.load("output/or_group_0/sub_prop_0.pth")
input_bounds = data["input"].cuda()  # GPU acceleration
output_constraints = [c.cuda() for c in data["output"]]
```

### Example 3: Complete Workflow

```python
import os
import numpy as np
from torchvnnlib import TorchVNNLIB

# Convert with timing information
converter = TorchVNNLIB(verbose=True, detect_fast_type=True, output_format="numpy")
converter.convert("acasxu_prop1.vnnlib", "acasxu_prop1_tensors")

# Load and verify
for or_idx in range(len(os.listdir("acasxu_prop1_tensors"))):
    or_folder = f"acasxu_prop1_tensors/or_group_{or_idx}"

    for sub_file in os.listdir(or_folder):
        if not sub_file.endswith(".npz"):
            continue

        data = np.load(os.path.join(or_folder, sub_file))

        input_bounds = data["input"]
        output_constraints = data["output"]

        print(f"Input shape: {input_bounds.shape}")
        print(f"Output constraints: {len(output_constraints)} OR groups")
```

## Architecture

### Processing Pipeline

```
VNN-LIB File
    |
    v
Preprocessing (extract declarations, assertions)
    |
    v
Type Detection (fast pattern matching)
    |
    +----> Fast Type Processor (Type1-5)
    |           |
    |           v
    |      Direct Tensor/Array Conversion
    |
    +----> AST-Based Processor (Complex)
                |
                v
           Tokenize -> Parse -> Optimize -> Flatten
                |
                v
           Tensor/Array Conversion
    |
    v
Save as .pth or .npz files
```

### Module Structure

```
torchvnnlib/
├── __init__.py           # Public API
├── _backend.py           # Backend abstraction (torch/numpy)
├── _torchvnnlib.py       # Main converter class
├── _to_tensor.py         # AST to tensor conversion
├── ast/                  # AST-based parsing
│   ├── _expr.py         # Expression classes
│   ├── _tokenize.py     # Tokenizer
│   ├── _parse.py        # Parser
│   ├── _optimize.py     # AST optimizer
│   ├── _flatten.py      # Property flattener
│   └── _preprocess.py   # Preprocessor
└── fast_type/           # Type-specific processors
    ├── _fast_type_detect.py
    ├── _type1_processor.py
    ├── _type2_processor.py
    ├── _type3_processor.py
    ├── _type4_processor.py
    ├── _type5_processor.py
    └── _utils.py
```

## VNN-LIB Format

### Supported Operations

**Declaration:**
```lisp
(declare-const X_i Real)  ; Input variables
(declare-const Y_i Real)  ; Output variables
```

**Assertions:**
```lisp
(assert (<= X_0 1.0))     ; Upper bound
(assert (>= X_0 -1.0))    ; Lower bound
(assert (<= Y_0 Y_1))     ; Output constraint
```

**Logical:**
```lisp
(and expr1 expr2 ...)     ; Conjunction
(or expr1 expr2 ...)      ; Disjunction
```

**Arithmetic:**
```lisp
(+ expr1 expr2 ...)       ; Addition
(- expr1 expr2)           ; Subtraction
(* coef var)              ; Multiplication
```

### Constraint Format

All constraints are normalized to: **b + Ax >= 0**

Where:
- `b`: bias/constant term
- `A`: coefficient matrix
- `x`: variable vector (outputs then inputs)

## Testing

```bash
cd torchvnnlib
python test/test_basic.py
```

## Performance Notes

- Fast type detection reduces processing time by 10-100x for common patterns
- Parallel processing enabled by default for multi-core systems
- `.pth` loading is orders of magnitude faster than parsing `.vnnlib`
- `.npz` loading is lightweight and doesn't require PyTorch
- Pre-allocated arrays minimize memory allocations
- NumPy backend has significantly smaller memory footprint

## Limitations

- Input variables must have closed bounds (both lower and upper)
- Only linear constraints supported
- Variable names must follow `X_i`, `Y_i` convention
- Division operations are rarely used and may have limited support

## Contributing

Contributions welcome! Areas for improvement:

- Support for additional VNN-LIB patterns
- Further performance optimizations
- Extended test coverage
- Documentation enhancements
