# TorchVNNLIB

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy 2.2](https://img.shields.io/badge/NumPy-2.2-green.svg)](https://numpy.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20(optional)-orange.svg)](https://pytorch.org/)
[![VNN-COMP 2024](https://img.shields.io/badge/VNN--COMP-2024-orange.svg)](https://sites.google.com/view/vnn2024)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

High-performance VNN-LIB to PyTorch tensor and NumPy array converter for neural network verification.

**Tested on all VNN-COMP 2024 benchmarks with 100% conversion success rate.**

## Overview

[VNN-LIB](https://www.vnnlib.org/) is the standard specification format for neural network verification properties, used extensively in [VNN-COMP](https://sites.google.com/view/vnn2024) and verification research. However, text-based `.vnnlib` files present significant performance challenges for modern tensor-based verification tools:

- Slow text parsing for large property files
- Inefficient for GPU-accelerated operations
- Cumbersome integration with PyTorch verification pipelines
- Repeated parsing overhead for batch processing

TorchVNNLIB solves these problems by converting VNN-LIB specifications into efficient binary formats:

- **PyTorch Backend**: `.pth` files with native GPU-ready tensors
- **NumPy Backend**: `.npz` files with lightweight arrays (no PyTorch dependency)
- **10-100x faster** loading compared to text parsing
- Standardized constraint representation
- Dual processing pipelines optimized for common VNN-LIB patterns

## Motivation

Neural network verification tools like α,β-CROWN, ERAN, and Marabou require repeated loading of verification properties. For benchmarks with hundreds of properties, parsing overhead becomes significant. TorchVNNLIB pre-processes VNN-LIB files once into binary tensor/array formats, enabling:

- Fast repeated loading during verification
- Direct GPU transfer for PyTorch-based verifiers
- Efficient batch processing of properties
- Integration with existing PyTorch verification codebases

## Features

- **Dual Backend Support**: PyTorch tensors or NumPy arrays
- **No PyTorch Dependency Required**: Use NumPy backend for lightweight deployment
- **Optimized Processing**: Type-specific fast processors for common VNN-LIB patterns (Type 1-5)
- **AST-Based Parser**: General-purpose parser for complex specifications
- **Parallel Processing**: Multi-threaded conversion for large benchmark suites
- **Standardized Output**: Consistent constraint representation across all backends
- **Production Ready**: Tested on all VNN-COMP 2024 benchmarks

## Installation

### Requirements

- Python 3.10 or higher
- numpy 2.2.4
- torch 2.5.1 (optional, only required for PyTorch backend)

### Setup

```bash
# Minimal installation (NumPy backend only)
pip install numpy==2.2.4

# Full installation (PyTorch backend)
pip install numpy==2.2.4 torch==2.5.1
```

## Quick Start

### NumPy Backend (No PyTorch Required)

```python
from torchvnnlib import TorchVNNLIB
import numpy as np

# Initialize converter with NumPy backend
converter = TorchVNNLIB(
    verbose=True,
    detect_fast_type=True,
    output_format="numpy"
)

# Convert VNN-LIB to NumPy arrays
converter.convert("property.vnnlib", target_folder_path="output")

# Load converted arrays
data = np.load("output/or_group_0/sub_prop_0.npz")

# Access constraints
input_bounds = data["input"]          # Shape: (n_inputs, 2)
output_constraints = data["output"]   # List of constraint arrays
```

### PyTorch Backend (GPU-Ready)

```python
from torchvnnlib import TorchVNNLIB
import torch

# Initialize converter with PyTorch backend
converter = TorchVNNLIB(
    verbose=True,
    detect_fast_type=True,
    output_format="torch"
)

# Convert VNN-LIB to PyTorch tensors
converter.convert("property.vnnlib", target_folder_path="output")

# Load converted tensors
data = torch.load("output/or_group_0/sub_prop_0.pth")

# Access constraints (GPU-ready)
input_bounds = data["input"].cuda()
output_constraints = [c.cuda() for c in data["output"]]
```

## VNN-LIB Type Support

TorchVNNLIB uses optimized type-specific processors for common VNN-LIB patterns:

- **Type 1**: Simple input bounds with simple output constraints
- **Type 2**: Simple input bounds with OR-grouped output constraints
- **Type 3**: OR-grouped input bounds with simple output constraints
- **Type 4**: OR-grouped input bounds with OR-grouped output constraints
- **Type 5**: Top-level OR wrapping complete properties

For complex or non-standard specifications, the AST-based parser provides general-purpose conversion.

## Output Format

### Directory Structure

```
output/
├── or_group_0/
│   ├── sub_prop_0.pth (or .npz)
│   ├── sub_prop_1.pth (or .npz)
│   └── ...
├── or_group_1/
│   ├── sub_prop_0.pth (or .npz)
│   └── ...
```

Each file contains a dictionary with:
- `"input"`: Input bounds array/tensor
- `"output"`: List of output constraint arrays/tensors

### Input Bounds Format

Shape: `(n_inputs, 2)`

```python
[[lower_bound_0, upper_bound_0],
 [lower_bound_1, upper_bound_1],
 ...]
```

Each row specifies `[lower, upper]` bounds for one input variable.

### Output Constraints Format

List of arrays/tensors with constraint format: **b + Ax ≥ 0**

Each constraint array has shape `(n_constraints, 1 + n_vars)` where:
- Column 0: Bias term `b`
- Columns 1 to `n_outputs`: Output variable coefficients
- Columns `n_outputs+1` to end: Input variable coefficients (if applicable)

Example constraint array:
```python
[[b_0, a_00, a_01, ..., a_0n],
 [b_1, a_10, a_11, ..., a_1n],
 ...]
```

Each row represents one linear inequality: `b_i + a_i0*y_0 + a_i1*y_1 + ... ≥ 0`

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

Main converter class for VNN-LIB to tensor/array conversion.

**Parameters**:
- `verbose` (bool): Print detailed timing and processing information
- `use_parallel` (bool): Enable parallel processing for multiple properties
- `detect_fast_type` (bool): Use optimized type-specific processors (recommended: True)
- `output_format` (str): Output format - `"torch"` for .pth or `"numpy"` for .npz

**Methods**:

#### convert()

```python
converter.convert(
    vnnlib_path: str,
    target_folder_path: str | None = None
)
```

Convert VNN-LIB file to tensor/array format.

**Parameters**:
- `vnnlib_path` (str): Path to .vnnlib input file
- `target_folder_path` (str | None): Output directory (default: `<vnnlib_path>` without extension)

**Returns**: None (writes files to disk)

## Examples

### Example 1: Simple Property Conversion

**VNN-LIB specification:**
```lisp
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

; Input bounds
(assert (<= X_0 0.5))
(assert (>= X_0 -0.5))

; Output constraint: Y_0 <= Y_1
(assert (<= Y_0 Y_1))
```

**Python conversion:**
```python
from torchvnnlib import TorchVNNLIB
import numpy as np

converter = TorchVNNLIB(output_format="numpy", verbose=True)
converter.convert("simple_property.vnnlib", "output")

data = np.load("output/or_group_0/sub_prop_0.npz")
print(f"Input bounds: {data['input']}")      # [[-0.5, 0.5]]
print(f"Output constraints: {data['output']}")  # [array([[0.0, -1.0, 1.0]])]
```

### Example 2: Batch Processing

```python
from torchvnnlib import TorchVNNLIB
from pathlib import Path

converter = TorchVNNLIB(
    output_format="numpy",
    use_parallel=True,
    detect_fast_type=True
)

# Convert all VNN-LIB files in a directory
vnnlib_dir = Path("benchmarks/acasxu/vnnlib")
for vnnlib_file in vnnlib_dir.glob("*.vnnlib"):
    output_dir = f"converted/{vnnlib_file.stem}"
    converter.convert(str(vnnlib_file), output_dir)
    print(f"Converted {vnnlib_file.name}")
```

### Example 3: Integration with Verification Workflow

```python
import torch
from torchvnnlib import TorchVNNLIB

# Convert once
converter = TorchVNNLIB(output_format="torch")
converter.convert("property.vnnlib", "property_tensors")

# Load multiple times during verification
for iteration in range(100):
    data = torch.load("property_tensors/or_group_0/sub_prop_0.pth")
    input_bounds = data["input"].cuda()
    output_constraints = [c.cuda() for c in data["output"]]

    # Run verification with GPU-accelerated bounds
    # verify_network(model, input_bounds, output_constraints)
```

## VNN-LIB Syntax Support

### Declarations

```lisp
(declare-const X_i Real)  ; Input variables (X_0, X_1, ...)
(declare-const Y_i Real)  ; Output variables (Y_0, Y_1, ...)
```

### Assertions

```lisp
(assert (<= X_0 1.0))     ; Upper bound
(assert (>= X_0 -1.0))    ; Lower bound
(assert (<= Y_0 Y_1))     ; Output constraint
```

### Logical Operations

```lisp
(and expr1 expr2 ...)     ; Conjunction
(or expr1 expr2 ...)      ; Disjunction
```

### Arithmetic Operations

```lisp
(+ expr1 expr2 ...)       ; Addition
(- expr1 expr2)           ; Subtraction
(* coefficient variable) ; Scalar multiplication
```

All constraints are normalized to the canonical form: **b + Ax ≥ 0**

## Architecture

### Processing Pipeline

```
VNN-LIB File
    │
    ├─> Preprocessing (extract declarations, assertions)
    │
    ├─> Type Detection (pattern matching for Type 1-5)
    │
    ├─────> Fast Type Processor (Type 1-5)
    │           │
    │           └─> Direct Tensor/Array Conversion
    │
    └─────> AST-Based Processor (Complex/General)
                │
                ├─> Tokenization
                ├─> Parsing
                ├─> AST Optimization
                └─> Flattening to Constraints
    │
    └─> Save as .pth or .npz files
```

### Module Structure

```
torchvnnlib/
├── __init__.py              # Public API
├── _backend.py              # Backend abstraction (torch/numpy)
├── _torchvnnlib.py          # Main converter class
├── _to_tensor.py            # AST to tensor conversion
├── ast/                     # AST-based parsing pipeline
│   ├── _expr.py             # Expression AST nodes
│   ├── _tokenize.py         # Tokenizer
│   ├── _parse.py            # Parser
│   ├── _optimize.py         # AST optimization
│   ├── _flatten.py          # Property flattening
│   └── _preprocess.py       # Preprocessing utilities
└── fast_type/               # Type-specific optimized processors
    ├── _fast_type_detect.py # Type detection
    ├── _type1_processor.py  # Type 1 processor
    ├── _type2_processor.py  # Type 2 processor
    ├── _type3_processor.py  # Type 3 processor
    ├── _type4_processor.py  # Type 4 processor
    ├── _type5_processor.py  # Type 5 processor
    └── _utils.py            # Shared utilities
```

## Testing and Validation

### VNN-COMP 2024 Benchmarks

TorchVNNLIB has been tested on all VNN-COMP 2024 benchmarks:

- **Total Benchmarks**: 23
- **Total Properties**: 1000+ VNN-LIB files
- **Conversion Success Rate**: 100%
- **Type Coverage**: Type 1-5 plus complex general specifications
- **Backend Testing**: Both PyTorch and NumPy backends validated

### Run Tests

```bash
cd torchvnnlib/test
python test_basic.py
```

## Performance Characteristics

- **Fast Type Processing**: 10-100x faster than AST-based parsing for common patterns
- **Binary Loading**: Orders of magnitude faster than text parsing
- **Memory Efficient**: Pre-allocated tensors/arrays minimize allocations
- **NumPy Backend**: Significantly smaller memory footprint than PyTorch
- **GPU Transfer**: Direct .pth loading enables immediate GPU operations

**Benchmark**: Converting 100 ACAS Xu properties takes approximately 2-3 seconds on modern hardware.

## Known Limitations

- Input variables must have both lower and upper bounds (closed intervals)
- Only linear constraints are supported
- Variable naming must follow `X_i` (inputs) and `Y_i` (outputs) convention
- Division operations have limited support
- Control flow constructs (if-then-else) not supported

## Related Projects

- **[SlimONNX](https://github.com/ZhongkuiMa/slimonnx)**: ONNX model optimization for verification. Works together with TorchVNNLIB for complete verification workflows.
- **[ShapeONNX](https://github.com/ZhongkuiMa/shapeonnx)**: Shape inference for ONNX models.
- **[VNN-COMP](https://sites.google.com/view/vnn2024)**: International Verification of Neural Networks Competition.

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Run black formatter on all modified files
4. Test on VNN-COMP benchmarks if applicable
5. Submit a pull request

Direct pushes to main branch are restricted.

## License

MIT License. See LICENSE file for details.

