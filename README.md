# TorchVNNLIB: Convert VNN-LIB to PyTorch Tensors

[**VNN-LIB**](https://www.vnnlib.org/) is the global standard for neural network verification benchmarks. It powers [**VNN-COMP (International Verification of Neural Networks Competition)**](https://www.aiverification.org/2025/) and is widely adopted by researchers worldwide. However, `.vnnlib` files can be inefficient to load, and cumbersome to integrate into modern tensor-based computation frameworks. We need a better way to handle these files and take it as the input and output of the verification tools.

**torchvnnlib**—a blazing-fast, AST-based, lightweight library designed to seamlessly convert `.vnnlib` specifications into PyTorch tensors. Whether you're verifying safety properties, robustness constraints, or any other neural network property, **torchvnnlib** accelerates your workflow, making benchmark handling effortless and computation-ready.

## Why Choose torchvnnlib? 🔥

PyTorch is the backbone of many modern verification tools. But raw `.vnnlib` files present significant hurdles:

- **Slow to parse** – `.vnnlib` files aren't designed for high-speed processing.
- **Inefficient for tensor operations** – Traditional representations lack the optimized structure required for tensor operation with GPU acceleration.
- **Not PyTorch-native** – Direct integration with PyTorch-based verification tools is cumbersome.

With **torchvnnlib**, handling VNN-LIB benchmarks becomes effortless, allowing you to focus on what matters—pushing the boundaries of neural network verification!

## Key Features 🏆

- 📜 **Unified Format** – Standardizes the representation of verification constraints, considering the needed input and output of modern verification tools. This is the **MOST IMPORTANT** feature of this tool. It is not only a converter, but also a standardization tool.
- 🧠 **High Efficient for Verification** - For the same inputs, the verifier needs a lot of efforts to verify the properties. So we transform the VNN-LIB files to a standardized format, which consider one input can be used for multiple properties.
- 🔍 **Universal AST Representation** - We construct a simple Abstract Syntax Tree (AST) to represent a VNN-LIB file. This is very friendly for researchers to process more customized operations.
- 🔄 **Automated Parsing** – Reads `.vnnlib` files and extracts verification constraints.
- 🧩 **Torch-Compatible Tensors** – Converts constraints into PyTorch tensors for efficient GPU computations.
- 🚀 **Optimized Storage** – Saves processed constraints as `.pth` files for near-instant loading.
- ⚡ **Seamless Integration** – Works smoothly with PyTorch-based verification tools.
- 🛠️ **Minimal Dependencies, Maximum Efficiency** – Designed for speed and ease of use.

## Installation 🚀

You need the following dependencies to run **torchvnnlib**:

- 🐍 Python 3.10+ (We are using Python 3.12)
- 🔥 PyTorch (We are using PyTorch 2.5.1. No special functions are used, so 2.x should be fine.)

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

VNN-LIB format is a Lisp-style format and is consistent with the SMT-LIB format. But for neural network verification, we only need to consider a small set of operations. Overall, there are two types of operations: `declear-constant` and `assert`.

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

- `assert` is used to declare the constraints. There are two types of constraints: bound and linear constraints. For input variables, we only need and support bound constraints now, i.e., an inequality constraint with only one input variable and one constant. For output variables, we only need and support linear constraints now, i.e., a linear constraint with multiple output variables and constants (sometimes, it involves input variables). Note that we use
  $$ b + Ax \geq 0, $$
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

## Cases for Neural Network Verification

In the typical cases, a VNN-LIB file contains the following parts:

- `declare-constant`: Declaration statements for input and output variables. It treats all the input and output variables as real constants.
- `assert`: Assertion statements for input scalar bounds and output linear constraints. The input bounds are closed bounds and limites the lower and upper bounds of the input variables. The output constraints are linear constraints, which only involve the output variables and constants (in a few times, they will involve input variables).

So we know, there are many assertion statements defining the input and output constraints. Currently, we only consider the following types of constraints:

- **Input constraints**: `<=`, `>=`, or `or` for input variables to declare the *input bounds*. It only involves one input variable and one constant.
- **Output constraints**: `<=`, `>=` for output variables. It may involve multiple output variables and constants (in a few times, they will involve input variables). We only consider the linear constraints now.

### Common VNN-LIB Files for Neural Network Verification

A VNN-LIB file is a text file with the `.vnnlib` extension. The declaration statements are only for declaring the input and output variables. There may be many assertion statements for the input and output constraints but the logic is that all such assertion statements in the same VNN-LIB file are combined by `and` operation, i.e., they are expected to be true at the same time. Now we introduce the common VNN-LIB files for neural network verification (refer to VNN-COMP). We only consider the abstract logic rather than the concrete text format (omitting the declaration statements).

- Type1: `(and ...input_constraints... ...output_constraints...)`
- Type2: `(and ...input_constraints... (or ...output_constraints...))`
- Type3: `(and (or ...input_constraints...) ...output_constraints...)`
- Type4: `(and (or ...input_constraints...) (or ...output_constraints...))`
- Type5: `(or (and ...input_constraints... ...output_constraints...))`

These types of VNN-LIB files are the most common ones in the neural network verification.

This too aims to transfer to all these types of VNN-LIB files to one standardized format, i.e., the **Type1 and Type2 formats**. The reason is that the neural network verifier is hard to calculate, and we expect the same inputs can be used for multiple output properties. Even though the current VNN-LIB files are created by considering such a logic but not all (at least for the VNN-COMP).

## Workflow 🔄

### Parsing VNN-LIB Files

We can omit the declaration statements because the neural network needs all inputs and outputs are constrained, and we can infer the number of variables from the constraints. Next, we will convert all the content in a VNN-LIB file to an AST (Abstract Syntax Tree) because it is a totally "and" statement regarding the whole file. This is a good data structure for us to perform more processing including optimization.

### Flattening Properties

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

### Formatting Properties

We will use a dictionary to store the properties, i.e., two tensors of input bounds and output constraints. The input bounds is a tensor of shape $(n,2)$, where $n$ is the number of input variables and $2$ is the lower and upper bounds of the input variables. The output constraints is a tensor of shape $(k, l, m+1)$, where $a$ is the number of OR clauses, $b$ is the number of AND clauses (number of output constriants), and $m$ is the number of output variables. The first dimension of an output constraints is the constant term and the output constraint has a form of $b + Ax \geq 0$, where $b$ is the constant term, $A$ is the coefficient matrix and $x$ is the variables. If the output constriants involve input variables, it will have a shape of $(k, l, 1+m+n)$, where the input variable will follow after the output variables.

## Examples 🌟

### Example of Type1

If we have the following VNN-LIB file.

```lisp
(declare-const X_0 Real)

(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (<= X_0 0.5))
(assert (>= X_0 -0.5))

(assert (<= Y_0 Y_1))
```

We will have the following PyTorch tensors.

```python
{
  "input": torch.tensor([[-0.5, 0.5]]),
  "output": torch.tensor([[[0.0, -1.0, 1.0]]]), 
}

```

### Example of Type2

If we have the following VNN-LIB file.

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

We will have the following PyTorch tensors.

```python
{
  "input": torch.tensor([[-0.5, 0.5]]),
  "output": torch.tensor([[[0.0, -1.0, 1.0], [0.5, -1.0, 0.0]]]), 
}

```

### Example of Type4

If we have the following VNN-LIB file.

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

We will have two files and each containing following PyTorch tensors.

```python
{
  "input": torch.tensor([[-0.5, 0.3]]),
  "output": torch.tensor([[[0.0, -1.0, 1.0], [0.5, -1.0, 0.0]]]), 
}
```
```python
{
  "input": torch.tensor([[-0.3, 0.5]]),
  "output": torch.tensor([[[0.0, -1.0, 1.0], [0.5, -1.0, 0.0]]]), 
}
```

### Example of Type5

If we have the following VNN-LIB file.

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

We will have two files and each containing following PyTorch tensors.

```python
{
  "input": torch.tensor([[-0.5, 0.3]]),
  "output": torch.tensor([[[0.0, -1.0, 1.0]]]), 
}
```
```python
{
    "input": torch.tensor([[-0.3, 0.5]]),
    "output": torch.tensor([[[0.5, -1.0, 0.0]]]), 
}
```

## Contributing 🤝

We warmly welcome contributions from everyone! Whether it's fixing bugs 🐞, adding features ✨, improving documentation 📚, or just sharing ideas 💡—your input is appreciated!

📌 NOTE: Direct pushes to the `main` branch are restricted. Make sure to fork the repository and submit a Pull Request for any changes!
