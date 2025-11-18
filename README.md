# TorchVNNLIB

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance VNN-LIB to PyTorch tensor converter for neural network verification.

## Overview

[VNN-LIB](https://www.vnnlib.org/) is the standard specification format for neural network verification properties, used extensively in [VNN-COMP](https://www.aiverification.org/2025/) and verification research. However, `.vnnlib` files present challenges for modern tensor-based verification tools:

- Slow text parsing
- Inefficient for GPU-accelerated operations
- Cumbersome integration with PyTorch pipelines

**TorchVNNLIB** solves these problems by converting VNN-LIB specifications directly into PyTorch tensors, enabling:

- Fast loading via `.pth` format
- Native PyTorch integration
- GPU-ready constraint representations
- Standardized property formats

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.x

### Setup

```bash
# Clone or add as submodule
git submodule add <repository-url> torchvnnlib

# Install dependencies
pip install torch
```

## Quick Start

```python
from torchvnnlib import TorchVNNLIB

# Initialize converter
converter = TorchVNNLIB(verbose=True, detect_fast_type=True)

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
    detect_fast_type: bool = True
)
```

**Parameters:**
- `verbose`: Print detailed timing information
- `use_parallel`: Enable parallel processing
- `detect_fast_type`: Use optimized type-specific processors (recommended)

**Methods:**

#### convert()

```python
converter.convert(
    vnnlib_path: str,
    target_folder_path: str | None = None
)
```

Convert VNN-LIB file to PyTorch tensors.

**Parameters:**
- `vnnlib_path`: Path to `.vnnlib` file
- `target_folder_path`: Output directory (defaults to `<vnnlib_path>` without extension)

**Output Structure:**
```
output/
├── or_group_0/
│   ├── sub_prop_0.pth
│   ├── sub_prop_1.pth
│   └── ...
├── or_group_1/
│   └── ...
```

Each `.pth` file contains:
```python
{
    "input": Tensor,       # Shape: (n_inputs, 2) - [lower, upper] bounds
    "output": list[Tensor] # List of constraint tensors
}
```

### Output Tensor Format

#### Input Bounds
Shape: `(n_inputs, 2)`
```python
[[lower_0, upper_0],
 [lower_1, upper_1],
 ...]
```

#### Output Constraints
List of tensors, each with shape: `(n_constraints, 1 + n_outputs)` or `(n_constraints, 1 + n_outputs + n_inputs)`

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

### Example 1: Simple Property (Type1)

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

**Output Structure:**
```
output/
├── or_group_0/
│   └── sub_prop_0.pth
```

**sub_prop_0.pth:**
```python
{
    "input": tensor([[-0.5, 0.5]]),
    "output": [tensor([[0.0, -1.0, 1.0]])]
}
```

### Example 2: OR-Grouped Outputs (Type2)

**VNN-LIB:**
```lisp
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

; Simple input bounds
(assert (<= X_0 0.5))
(assert (>= X_0 -0.5))

; OR-grouped output constraints
(assert (or
    (and (<= Y_0 Y_1))
    (and (<= Y_0 0.5))
))
```

**Output Structure:**
```
output/
├── or_group_0/
│   └── sub_prop_0.pth
```

**sub_prop_0.pth:**
```python
{
    "input": tensor([[-0.5, 0.5]]),
    "output": [
        tensor([[0.0, -1.0, 1.0]]),
        tensor([[0.5, -1.0, 0.0]])
    ]
}
```

### Example 3: OR-Grouped Inputs (Type3)

**VNN-LIB:**
```lisp
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

; OR-grouped input constraints
(assert (or
    (and (<= X_0 0.3) (>= X_0 -0.5))
    (and (<= X_0 0.5) (>= X_0 -0.3))
))

; Simple output constraint
(assert (<= Y_0 Y_1))
```

**Output Structure:**
```
output/
├── or_group_0/
│   ├── sub_prop_0.pth
│   └── sub_prop_1.pth
```

**sub_prop_0.pth:**
```python
{
    "input": tensor([[-0.5, 0.3]]),
    "output": [tensor([[0.0, -1.0, 1.0]])]
}
```

**sub_prop_1.pth:**
```python
{
    "input": tensor([[-0.3, 0.5]]),
    "output": [tensor([[0.0, -1.0, 1.0]])]
}
```

### Example 4: OR-Grouped Inputs and Outputs (Type4)

**VNN-LIB:**
```lisp
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

; OR-grouped input constraints
(assert (or
    (and (<= X_0 0.3) (>= X_0 -0.5))
    (and (<= X_0 0.5) (>= X_0 -0.3))
))

; OR-grouped output constraints
(assert (or
    (and (<= Y_0 Y_1))
    (and (<= Y_0 0.5))
))
```

**Output Structure:**
```
output/
├── or_group_0/
│   ├── sub_prop_0.pth
│   └── sub_prop_1.pth
```

**sub_prop_0.pth:**
```python
{
    "input": tensor([[-0.5, 0.3]]),
    "output": [
        tensor([[0.0, -1.0, 1.0]]),
        tensor([[0.5, -1.0, 0.0]])
    ]
}
```

**sub_prop_1.pth:**
```python
{
    "input": tensor([[-0.3, 0.5]]),
    "output": [
        tensor([[0.0, -1.0, 1.0]]),
        tensor([[0.5, -1.0, 0.0]])
    ]
}
```

### Example 5: Top-Level OR (Type5)

**VNN-LIB:**
```lisp
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

; Top-level OR wrapping complete properties
(assert (or
    (and (<= X_0 0.3) (>= X_0 -0.5) (<= Y_0 Y_1))
    (and (<= X_0 0.5) (>= X_0 -0.3) (<= Y_0 0.5))
))
```

**Output Structure:**
```
output/
├── or_group_0/
│   ├── sub_prop_0.pth
│   └── sub_prop_1.pth
```

**sub_prop_0.pth:**
```python
{
    "input": tensor([[-0.5, 0.3]]),
    "output": [tensor([[0.0, -1.0, 1.0]])]
}
```

**sub_prop_1.pth:**
```python
{
    "input": tensor([[-0.3, 0.5]]),
    "output": [tensor([[0.5, -1.0, 0.0]])]
}
```

### Example 6: Complete Workflow

```python
import os
import torch
from torchvnnlib import TorchVNNLIB

# Convert with timing information
converter = TorchVNNLIB(verbose=True, detect_fast_type=True)
converter.convert("acasxu_prop1.vnnlib", "acasxu_prop1_tensors")

# Load and verify
for or_idx in range(len(os.listdir("acasxu_prop1_tensors"))):
    or_folder = f"acasxu_prop1_tensors/or_group_{or_idx}"

    for sub_file in os.listdir(or_folder):
        data = torch.load(os.path.join(or_folder, sub_file))

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
    |      Direct Tensor Conversion
    |
    +----> AST-Based Processor (Complex)
                |
                v
           Tokenize -> Parse -> Optimize -> Flatten
                |
                v
           Tensor Conversion
    |
    v
Save as .pth files
```

### Module Structure

```
torchvnnlib/
├── __init__.py           # Public API
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
- Pre-allocated tensors minimize memory allocations

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
