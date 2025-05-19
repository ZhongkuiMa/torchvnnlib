# TorchVNNLIB: Convert VNN-LIB to PyTorch Tensors 🔄

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> [VNN-LIB](https://www.vnnlib.org/) is the global standard for neural network verification benchmarks. It powers [VNN-COMP (International Verification of Neural Networks Competition)](https://www.aiverification.org/2025/) and is widely adopted by researchers worldwide.
> It is [Lisp-style](https://lisp-lang.org/style-guide/) and is consistent with the [SMT-LIB](https://smt-lib.org/) format. The VNN-LIB format is a text file with the `.vnnlib` extension. It contains a set of constraints that define the properties of the neural network.
> However, `.vnnlib` files can be inefficient to load, and cumbersome to integrate into modern tensor-based computation frameworks. We need a better way to handle these files and take it as the input and output of the verification tools.

**torchvnnlib**—a blazing-fast, AST-based, lightweight library designed to seamlessly convert `.vnnlib` specifications into PyTorch tensors. Whether you're verifying safety properties, robustness constraints, or any other neural network property, **torchvnnlib** accelerates your workflow, making benchmark handling effortless and computation-ready.

## 📋 Table of Contents
- [Why Choose torchvnnlib?](#why-choose-torchvnnlib-)
- [Key Features](#key-features-)
- [Installation](#installation-)
- [Quick Start](#quick-start-)
- [Usage](#usage-)
- [Documentation](#documentation-)
  - [Involved Operations](#involved-operations)
  - [Cases for Neural Network Verification](#cases-for-neural-network-verification-)
  - [Workflow](#workflow-)
- [Examples](#examples-)
- [Contributing](#contributing-)

## Why Choose torchvnnlib? 🔥

[PyTorch](https://pytorch.org/) is the backbone of many modern verification tools. But raw `.vnnlib` files present significant hurdles:

- **Slow to parse** – `.vnnlib` files aren't designed for high-speed processing.
- **Inefficient for tensor operations** – Traditional representations lack the optimized structure required for tensor operation with GPU acceleration.
- **Not PyTorch-native** – Direct integration with PyTorch-based verification tools is cumbersome.

With **torchvnnlib**, handling VNN-LIB benchmarks becomes effortless, allowing you to focus on what matters—pushing the boundaries of neural network verification!

## Key Features 🏆

Unlock a new level of efficiency and precision in neural network verification with our powerful toolset! Here's what makes us stand out:

- 📜 **One Format to Rule Them All**  
  Say goodbye to messy constraints! We standardize VNN-LIB files into a unified, streamlined format — setting the foundation for faster, smarter verification workflows. Not just a converter — a true game-changer.

- 🧠 **Built for Ultimate Verification Efficiency**  
  Smart transformations mean a single input can verify multiple properties — dramatically reducing computation and speeding up your verification process.

- 🔍 **Developer-Friendly AST**  
  Our lightweight Abstract Syntax Tree (AST) makes constraint manipulation a breeze. Customize, extend, and innovate with ease.

- 🔄 **Fully Automated Parsing**  
  No more manual handling! Instantly read .vnnlib files and extract constraints in a click.

- 🧩 **PyTorch Native**  
  Constraints are output as PyTorch tensors (saved in `.pth` format) — ready to harness the full power of GPUs without extra conversions.

- 🚀 **Lightning-Fast Loading**  
  Save processed constraints as `.pth` files and enjoy near-instant startup times for large-scale data.

- ⚡ **Seamless Integration**  
  Designed to plug and play with your PyTorch-based verification pipelines — no headaches, just results.

- 🛠️ **Minimal Setup, Maximum Impact**  
  Lightweight, dependency-minimal, and blazing fast — because your time and resources deserve the best.

## Installation 🚀

### Dependencies

You need the following dependencies to run **torchvnnlib**:

- 🐍 Python 3.10+ (We are using Python 3.12)
- 🔥 PyTorch (We are using PyTorch 2.5.1. No special functions are used, so 2.x should be fine.)

## Quick Start 🚀

```python
import torch
from torchvnnlib import convert_vnnlib

# Convert single .vnnlib file to PyTorch tensors
properties = convert_vnnlib("path/to/property.vnnlib")

# Access input bounds
input_bounds = properties["input"]  # shape: (n_inputs, 2)

# Access output constraints
output_constraints = properties["output"]  # list or tensor

# Save the converted properties for faster loading
torch.save(properties, "property.pth")

# Load the saved properties
properties = torch.load("property.pth")
```

## Usage 🚀

You can find example usage in the `test` folder.

By storing constraints as `.pth` files, subsequent loading becomes near-instantaneous, giving your verification pipeline **unmatched efficiency**! ⚡

### Test Examples of VNNCOMP'24 🧪

You need to get the repo of [vnncomp2024](https://github.com/ChristopherBrix/vnncomp2024_benchmarks). This repo does not contain the benchmarks folder because it is about 20GB. The testing examples are in the `test_vnncomp` folder. Then you make sure the following folder structure:

```
torchvnnlib/
│   ├── torchvnnlib/
│   ├── README.md
│   └── test_vnncomp/
└── ...
vnncomp2024/
│   ├── benchmarks/
└── ...
```

## Documentation 📚

### Involved Operations

#### Declaration Statement

VNN-LIB format is a [Lisp-style](https://lisp-lang.org/style-guide/) format and is consistent with the [SMT-LIB](https://smt-lib.org/) format. But for neural network verification, we only need to consider a small set of operations. Overall, there are two types of operations: `declear-constant` and `assert`.

- `declare-constant` is used to declare the input and output variables. We take the input and output variables both as constants in neural network verification. There are some fixed formats: all the input variables are named as $X_i$, and the output variables are named as $Y_i$ , where $i$ is the index of the variable. They are both real numbers. So you will see the following format in the VNN-LIB file:

```lisp
(declare-const X_0 Real)
(declare-const X_1 Real)
...
(declare-const Y_0 Real)
(declare-const Y_1 Real)
...
```

#### Assertion Statement

- `assert` is used to declare the constraints. There are two types of constraints: bound and linear constraints. For input variables, we only need and support bound constraints now, i.e., an inequality constraint with only one input variable and one constant. For output variables, we only need and support linear constraints now, i.e., a linear constraint with multiple output variables and constants (sometimes, it involves input variables).

Note that we use the following constraint format:
$$
b + Ax \geq 0 ,
$$
where $b$ is a constant, $A$ is a matrix and $x$ is a vector of the variables, as the unifed format to represent a linear constraint. Note that we need a closed input bounds for the input variables, i.e., the lower bound and upper bound of the input variables. The following example shows the two types of constraints:

```lisp
(assert (<= X_0 0.5))
(assert (>= X_0 -0.5))
(assert (<= Y_0 Y_1))
```

#### Comparison Operations

Commonly, we let the LHS be single variable and the RHS be a more compliacted expression.

- `<=` is used to declare the less than or equal to constraint. It is a binary operation, i.e., it takes two inputs. The first input is the left-hand side of the constraint, and the second input is the right-hand side of the constraint.

- `>=` is used to declare the greater than or equal to constraint. It is a binary operation similar to `<=`.

- `=` is used to declare the equal to constraint. It is a binary operation similar to `<=` and `>=`.

#### Logical Operations

Commonly, we use `and` and `or` operations to declare the logical constraints.

- `and` is used to declare the and operation. It is a N-ary operation, i.e., it can take multiple inputs. The inputs are the constraints that we want to combine.
- `or` is used to declare the or operation. It is a N-ary operation similar to `and`.

#### Arithmetic Operations

Commonly, we use plus and minus operations to declare the linear constraints. The multiplication operation is used for coefficients and the division operation is seldomly used.

- `+` is used to declare the plus operation. We treat it as a N-ary operation, i.e., it can take multiple inputs.
- `-` is used to declare the minus operation. It is a binary operation, i.e., it takes two inputs. *Note that we do not consider negative numbers now and we will directly parse negative numbers.*
- `*` is used to declare the multiplication operation. It is a binary operation similar to `-`.
- `/` is used to declare the division operation. It is a binary operation similar to `-`.

### Cases for Neural Network Verification 🧪

In the typical cases, a VNN-LIB file contains the following parts:

- `declare-constant`: Declaration statements for input and output variables. It treats all the input and output variables as real constants.
- `assert`: Assertion statements for input scalar bounds and output linear constraints. The input bounds are closed bounds and limites the lower and upper bounds of the input variables. The output constraints are linear constraints, which only involve the output variables and constants (in a few times, they will involve input variables).

So we know, there are many assertion statements defining the input and output constraints. Currently, we only consider the following types of constraints:

- **Input constraints**: `<=`, `>=`, or `or` for input variables to declare the *input bounds*. It only involves one input variable and one constant.
- **Output constraints**: `<=`, `>=` for output variables. It may involve multiple output variables and constants (in a few times, they will involve input variables). We only consider the linear constraints now.

#### Common VNN-LIB Files for Neural Network Verification

A VNN-LIB file is a text file with the `.vnnlib` extension. The declaration statements are only for declaring the input and output variables. There may be many assertion statements for the input and output constraints but the logic is that all such assertion statements in the same VNN-LIB file are combined by `and` operation, i.e., they are expected to be true at the same time. Now we introduce the common VNN-LIB files for neural network verification (refer to VNN-COMP). We only consider the abstract logic rather than the concrete text format (omitting the declaration statements).

- Type1: `(and ...input_constraints... ...output_constraints...)`
- Type2: `(and ...input_constraints... (or ...output_constraints...))`
- Type3: `(and (or ...input_constraints...) ...output_constraints...)`
- Type4: `(and (or ...input_constraints...) (or ...output_constraints...))`
- Type5: `(or (and ...input_constraints... ...output_constraints...))`

These types of VNN-LIB files are the most common ones in the neural network verification.

This too aims to transfer to all these types of VNN-LIB files to one standardized format, i.e., the **Type1 and Type2 formats**. The reason is that the neural network verifier is hard to calculate, and we expect the same inputs can be used for multiple output properties. Even though the current VNN-LIB files are created by considering such a logic but not all (at least for the VNN-COMP).

### Workflow 🔄

#### Parsing VNN-LIB Files

We can omit the declaration statements because the neural network needs all inputs and outputs are constrained, and we can infer the number of variables from the constraints. Next, we will convert all the content in a VNN-LIB file to an AST (Abstract Syntax Tree) because it is a totally "and" statement regarding the whole file. This is a good data structure for us to perform more processing including optimization.

#### Flattening Properties

Then we will *flatten* the AST to an "and" list of "or" groups of each single property. This will be a recursive structure of folders and files. Here, we consider that we do not want a single file contain too much data or too many properties.
*Loading a single file with too many properties will be slow and inefficient because we can only verify one property (for one specific input) at a time.*
The current case sometimes needs us remain all inputs and outputs in the same file, and this is not what we expect. We will unify all the properties in the same format as below:

```lisp
(and
    (or 
        (and ...input_constraints... (or ...output_constraints...))
        (and ...input_constraints... ...output_constraints...)
        ...
    )
)
```

So, there will be at most three levels of folders, and *we do not want more*. If more, you should consider to design the properties in a better way.

#### Formatting Properties

We will use a dictionary to store the properties, i.e., two tensors of input bounds and output constraints. The input bounds is a tensor `Tensor` of shape $(n,2)$, where $n$ is the number of input variables and $2$ is the lower and upper bounds of the input variables. The output constraints is a list of tensors `list[Tensor]` of shape $(l, 1+m)$, where each component of the list is the number of OR clauses (this accepts different length tensors), $l$ is the number of AND clauses (number of output constraints), and $m$ is the number of output variables. The first dimension of an output constraints is the constant term and the output constraint has a form of $b + Ax \geq 0$, where $b$ is the constant term, $A$ is the coefficient matrix and $x$ is the variables. If the output constraints involve input variables, it will have a shape of $(k, l, 1+m+n)$, where the input variable will follow after the output variables.

## Examples 🌟

### Example of Type1

If we have the following VNN-LIB file:

```lisp
(declare-const X_0 Real)

(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (<= X_0 0.5))
(assert (>= X_0 -0.5))

(assert (<= Y_0 Y_1))
```

We will have the following PyTorch tensors:

```python
{
    "input": torch.tensor([[-0.5, 0.5]]),
    "output": torch.tensor([[[0.0, -1.0, 1.0]]]),
}
```

### Example of Type2

If we have the following VNN-LIB file:

```lisp
(declare-const X_0 Real)

(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (<= X_0 0.5))
(assert (>= X_0 -0.5))

(assert 
    (or 
        (and (<= Y_0 Y_1))
        (and (<= Y_0 0.5))
    )
)
```

We will have the following PyTorch tensors:

```python
{
    "input": torch.tensor([[-0.5, 0.5]]),
    "output": [
        torch.tensor([[0.0, -1.0, 1.0]]),
        torch.tensor([[0.5, -1.0, 0.0]])
    ],
}
```

### Example of Type4

If we have the following VNN-LIB file:

```lisp
(declare-const X_0 Real)

(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert 
    (or
        (and (<= X_0 0.3) (>= X_0 -0.5))
        (and (<= X_0 0.5) (>= X_0 -0.3))
    )
)

(assert 
    (or 
        (and (<= Y_0 Y_1))
        (and (<= Y_0 0.5))
    )
)
```

We will have two files and each containing following PyTorch tensors:

```python
# File 1
{
    "input": torch.tensor([[-0.5, 0.3]]),
    "output": [
        torch.tensor([[0.0, -1.0, 1.0]]),
        torch.tensor([[0.5, -1.0, 0.0]])
    ],
}

# File 2
{
    "input": torch.tensor([[-0.3, 0.5]]),
    "output": [
        torch.tensor([[0.0, -1.0, 1.0]]),
        torch.tensor([[0.5, -1.0, 0.0]])
    ],
}
```

### Example of Type5

If we have the following VNN-LIB file:

```lisp
(declare-const X_0 Real)

(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert
    (or
        (and (<= X_0 0.3) (>= X_0 -0.5) (<= Y_0 Y_1))
        (and (<= X_0 0.5) (>= X_0 -0.3) (<= Y_0 0.5))
    )
) 
```

We will have two files and each containing following PyTorch tensors:

```python
# File 1
{
    "input": torch.tensor([[-0.5, 0.3]]),
    "output": torch.tensor([[[0.0, -1.0, 1.0]]]),
}

# File 2
{
    "input": torch.tensor([[-0.3, 0.5]]),
    "output": torch.tensor([[[0.5, -1.0, 0.0]]]),
}
```

## Contributing 🤝

We warmly welcome contributions from everyone! Whether it's fixing bugs 🐞, adding features ✨, improving documentation 📚, or just sharing ideas 💡—your input is appreciated!

### How to Contribute

1. 🍴 **Fork the repository**
2. 🔄 **Create a branch** (`git checkout -b feature/amazing-feature`)
3. 💻 **Make your changes** and test them thoroughly
4. 📝 **Commit your changes** (`git commit -m 'Add amazing feature'`)
5. 🚀 **Push to your branch** (`git push origin feature/amazing-feature`)
6. 🔍 **Open a Pull Request** and describe your changes in detail

📌 **NOTE**: Direct pushes to the `main` branch are restricted. Make sure to fork the repository and submit a Pull Request for any changes!

### Code Style

We use [Black](https://github.com/psf/black) for code formatting and encourage comprehensive docstrings for all functions and classes.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
