# TorchVNNLIB: Convert VNN-LIB to PyTorch Tensors

[**VNN-LIB**](https://www.vnnlib.org/) is the global standard for neural network verification benchmarks. It powers [**VNN-COMP (International Verification of Neural Networks Competition)**](https://www.aiverification.org/2025/) and is widely adopted by researchers worldwide. However, `.vnnlib` files can be inefficient to load, and cumbersome to integrate into modern tensor-based computation frameworks.

> A good protocol should not sacrifice high efficiency IO operations. In other word, our tool is just a converter, haha...🫡🫡🫡

**torchvnnlib**—a blazing-fast, lightweight library designed to seamlessly convert `.vnnlib` specifications into PyTorch tensors. Whether you're verifying safety properties, robustness constraints, or any other neural network property, **torchvnnlib** accelerates your workflow, making benchmark handling effortless and computation-ready.

## Why Choose torchvnnlib? 🔥

PyTorch is the backbone of many modern verification tools. But raw `.vnnlib` files present significant hurdles:

- **Slow to parse** – `.vnnlib` files aren't designed for high-speed processing.
- **Inefficient for tensor operations** – Traditional representations lack the optimized structure required for tensor operation with GPU acceleration.
- **Not PyTorch-native** – Direct integration with PyTorch-based verification tools is cumbersome.

With **torchvnnlib**, handling VNN-LIB benchmarks becomes effortless, allowing you to focus on what matters—pushing the boundaries of neural network verification!

## Key Features 🏆

- 🔄 **Automated Parsing** – Reads `.vnnlib` files and extracts verification constraints.
- 🧩 **Torch-Compatible Tensors** – Converts constraints into PyTorch tensors for efficient GPU computations.
- 🚀 **Optimized Storage** – Saves processed constraints as `.pth` files for near-instant loading.
- ⚡ **Seamless Integration** – Works smoothly with PyTorch-based verification tools.
- 🛠️ **Minimal Dependencies, Maximum Efficiency** – Designed for speed and ease of use.

## Data Format Overview 💾

**torchvnnlib** focuses on **linear inequalities**, the fundamental building blocks of verification constraints. Each `.vnnlib` file consists of:

1. **Pre-conditions on the inputs** – Defining the valid input space.
2. **Expected constraints on the outputs** – Ensuring correct network behavior.

### Example

#### VNN-LIB Specification

```lisp
; ACAS Xu property 6

(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)
(declare-const X_3 Real)
(declare-const X_4 Real)

(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(declare-const Y_3 Real)
(declare-const Y_4 Real)

(assert (or
    (and (<= X_0 0.700434925) (>= X_0 -0.129289109) (<= X_1 0.499999896) (>= X_1 0.11140846) (<= X_2 -0.499204121) (>= X_2 -0.499999896) (<= X_3 0.5) (>= X_3 -0.5) (<= X_4 0.5) (>= X_4 -0.5))
    (and (<= X_0 0.700434925) (>= X_0 -0.129289109) (<= X_1 -0.11140846) (>= X_1 -0.499999896) (<= X_2 -0.499204121) (>= X_2 -0.499999896) (<= X_3 0.5) (>= X_3 -0.5) (<= X_4 0.5) (>= X_4 -0.5))
))

(assert (or
    (and (<= Y_1 Y_0))
    (and (<= Y_2 Y_0))
    (and (<= Y_3 Y_0))
    (and (<= Y_4 Y_0))
))
```

#### Converted PyTorch Representation

There are two main components in the converted PyTorch representation:

- **inputs**: A list of tensors representing the input bounds. each tensor has a shape of `(n, 2)`, where `n` is the number of input variables. Each row represents a variable's lower and upper bounds.
- **outputs**: A list of tensors representing the output constraints. All the tensors are "or" clauses. Each tensor represents a set of constraints, which are "and" clauses. The tensor has a shape of `(m, n+1)`, where `m` is the number of "and" clauses and `n` is the number of output variables.

```python
{
    "inputs": [
        torch.tensor([
            [-0.129289109, 0.700434925],
            [0.11140846, 0.499999896],
            [-0.499999896, -0.499204121],
            [-0.5, 0.5],
            [-0.5, 0.5]
        ]),
        torch.tensor([
            [-0.129289109, 0.700434925],
            [-0.499999896, -0.11140846],
            [-0.499999896, -0.499204121],
            [-0.5, 0.5],
            [-0.5, 0.5]
        ])
    ],
    "outputs": [
        torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    ],
}
```

## Installation

You need the following dependencies to run **torchvnnlib**:

- Python 3.10+ (We are using Python 3.12)
- PyTorch (We are using PyTorch 2.5.1. No special functions are used, so 2.x should be fine.)

## Usage

### Example Script

```python
import time
import torch
from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":
    converter = TorchVNNLIB(verbose=True)

    file_path = "prop_6.vnnlib"
    converter.convert(file_path)
    data_path = file_path.replace(".vnnlib", ".pth")
    t = time.perf_counter()
    data = torch.load(data_path)
    t = time.perf_counter() - t
    print(f"Loaded {data_path} ({t:.4f}s)")
```

### Expected Output

```
Torch vnnlib prop_6.vnnlib...
Saved to prop_6.pth (0.0020s)
Loaded prop_6.pth (0.0007s)
Torch vnnlib TinyImageNet_resnet_medium_prop_idx_57_sidx_9858_eps_0.0039.vnnlib...
Saved to TinyImageNet_resnet_medium_prop_idx_57_sidx_9858_eps_0.0039.pth (0.2309s)
Loaded TinyImageNet_resnet_medium_prop_idx_57_sidx_9858_eps_0.0039.pth (0.0005s)
```

By storing constraints as `.pth` files, subsequent loading becomes near-instantaneous, giving your verification pipeline **unmatched efficiency**!

## 🤝 Contributing

We warmly welcome contributions from everyone! Whether it's fixing bugs 🐞, adding features ✨, improving documentation 📚, or just sharing ideas 💡—your input is appreciated!

📌 NOTE: Direct pushes to the `main` branch are restricted. Make sure to fork the repository and submit a Pull Request for any changes!
