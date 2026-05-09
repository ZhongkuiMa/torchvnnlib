"""AST expression types for VNN-LIB formula representation."""

__docformat__ = "restructuredtext"
__all__ = [
    "Add",
    "And",
    "BinaryOp",
    "Cst",
    "Div",
    "Eq",
    "Expr",
    "Geq",
    "Leq",
    "Mul",
    "NaryOp",
    "Or",
    "Sub",
    "UnaryOp",
    "Var",
]


class Expr:
    """Abstract base class for all VNN-LIB AST expression nodes."""

    def __init__(self):
        """Initialize cached flags for input/output variable presence."""
        self._has_input_vars = None  # Cache for input variable presence
        self._has_output_vars = None  # Cache for output variable presence

    def __repr__(self):
        """Return string representation."""
        raise NotImplementedError("This should be implemented in subclasses")

    def __eq__(self, other):
        """Check equality. Must be implemented in subclasses."""
        raise NotImplementedError("This should be implemented in subclasses")

    def __hash__(self):
        """Hash based on string representation."""
        # Default implementation - subclasses should override
        return hash(self.__repr__())

    @property
    def has_input_vars(self) -> bool:
        """Check if expression contains input variables (X_*)."""
        if self._has_input_vars is None:
            self._has_input_vars = self._compute_has_input_vars()
        return self._has_input_vars  # type: ignore[no-any-return]

    @property
    def has_output_vars(self) -> bool:
        """Check if expression contains output variables (Y_*)."""
        if self._has_output_vars is None:
            self._has_output_vars = self._compute_has_output_vars()
        return self._has_output_vars  # type: ignore[no-any-return]

    def _compute_has_input_vars(self) -> bool:
        """Compute whether expression contains input variables. Override in subclasses."""
        return False

    def _compute_has_output_vars(self) -> bool:
        """Compute whether expression contains output variables. Override in subclasses."""
        return False


class Cst(Expr):
    """Constant literal node in a VNN-LIB expression.

    :param value: The numeric constant value.
    """

    def __init__(self, value: float):
        """Store a constant numeric value.

        :param value: The numeric constant.
        """
        super().__init__()
        self.value = value

    def __repr__(self):
        """Return string representation."""
        return f"{self.value}"

    def __eq__(self, other):
        """Check equality by comparing value."""
        if not isinstance(other, Cst):
            return False
        return self.value == other.value

    def __hash__(self):
        """Hash based on value."""
        return hash(self.value)


class Var(Expr):
    """Variable reference node (``X_i`` for inputs, ``Y_i`` for outputs).

    :param name: Variable name in ``X_<idx>`` or ``Y_<idx>`` format.
    :raises ValueError: If name does not match the expected format.
    """

    def __init__(self, name: str):
        """Store variable name, type, and index.

        :param name: Variable name (e.g., ``X_0``, ``Y_3``).
        :raises ValueError: If name does not match ``X_<idx>`` or ``Y_<idx>``.
        """
        super().__init__()
        self.name = name
        if not ((name.startswith(("X", "Y"))) and name[1] == "_" and name[2:].isdigit()):
            raise ValueError(f"Variable name must start with 'X' or 'Y' but {name}.")

        # Cache variable type and index for performance
        self.var_type = name[0]  # 'X' or 'Y'
        self.index = int(name[2:])  # Parse index once

        # Set cached flags directly since we know the type
        self._has_input_vars = self.var_type == "X"
        self._has_output_vars = self.var_type == "Y"

    def __repr__(self):
        """Return string representation."""
        return f"{self.name}"

    def __eq__(self, other):
        """Check equality by comparing name."""
        if not isinstance(other, Var):
            return False
        return self.name == other.name

    def __hash__(self):
        """Hash based on name."""
        return hash(self.name)

    def _compute_has_input_vars(self) -> bool:
        """Compute whether this variable is an input variable."""
        return self.var_type == "X"

    def _compute_has_output_vars(self) -> bool:
        """Compute whether this variable is an output variable."""
        return self.var_type == "Y"


class UnaryOp(Expr):
    """Abstract base for unary operations with a single operand.

    :param arg: The single operand expression.
    """

    def __init__(self, arg: Expr):
        """Store the operand expression.

        :param arg: The single operand.
        """
        super().__init__()
        self.arg = arg

    def __repr__(self):
        """Return string representation."""
        raise RuntimeError("This should be implemented in subclasses")

    def __eq__(self, other):
        """Check equality by comparing arg."""
        if not isinstance(other, UnaryOp):
            return False
        return self.arg == other.arg

    def __hash__(self):
        """Hash based on arg."""
        return hash(self.arg)

    def _compute_has_input_vars(self) -> bool:
        """Compute whether the operand contains input variables."""
        return self.arg.has_input_vars

    def _compute_has_output_vars(self) -> bool:
        """Compute whether the operand contains output variables."""
        return self.arg.has_output_vars


class BinaryOp(Expr):
    """Abstract base for binary operations with left and right operands.

    :param left: Left operand expression.
    :param right: Right operand expression.
    """

    def __init__(self, left: Expr, right: Expr):
        """Store left and right operand expressions.

        :param left: Left operand.
        :param right: Right operand.
        """
        super().__init__()
        self.left = left
        self.right = right

    def __repr__(self):
        """Return string representation."""
        raise RuntimeError("This should be implemented in subclasses")

    def __eq__(self, other):
        """Check equality by comparing left and right."""
        if not isinstance(other, BinaryOp):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self):
        """Hash based on left and right."""
        return hash((self.left, self.right))

    def _compute_has_input_vars(self) -> bool:
        """Compute whether either operand contains input variables."""
        return self.left.has_input_vars or self.right.has_input_vars

    def _compute_has_output_vars(self) -> bool:
        """Compute whether either operand contains output variables."""
        return self.left.has_output_vars or self.right.has_output_vars


class NaryOp(Expr):
    """Abstract base for n-ary operations with a variable number of operands.

    :param args: List of operand expressions.
    """

    def __init__(self, args: list[Expr]):
        """Store the list of operand expressions.

        :param args: List of operand expressions.
        """
        super().__init__()
        self.args = args

    def __repr__(self):
        """Return string representation."""
        raise RuntimeError("This should be implemented in subclasses")

    def __eq__(self, other):
        """Check equality by comparing all args element-wise."""
        if not isinstance(other, NaryOp):
            return False
        if len(self.args) != len(other.args):
            return False
        return all(self.args[i] == other.args[i] for i in range(len(self.args)))

    def __hash__(self):
        """Hash based on args tuple."""
        return hash(tuple(self.args))

    def __iter__(self):
        """Iterate over operand expressions."""
        return iter(self.args)

    def _compute_has_input_vars(self) -> bool:
        """Compute whether any operand contains input variables."""
        return any(arg.has_input_vars for arg in self.args)

    def _compute_has_output_vars(self) -> bool:
        """Compute whether any operand contains output variables."""
        return any(arg.has_output_vars for arg in self.args)


class Add(NaryOp):
    """N-ary addition node ``(+ a b ...)`` in VNN-LIB S-expression form."""

    def __repr__(self):
        """Return string representation."""
        return f"(+ {' '.join(map(str, self.args))})"

    def __hash__(self):
        """Hash based on args tuple."""
        return hash(tuple(self.args))


class Sub(BinaryOp):
    """Binary subtraction node ``(- left right)`` in VNN-LIB S-expression form."""

    def __repr__(self):
        """Return string representation."""
        return f"(- {self.left} {self.right})"

    def __hash__(self):
        """Hash based on left and right."""
        return hash((self.left, self.right))


class Mul(BinaryOp):
    """Binary multiplication node ``(* left right)`` in VNN-LIB S-expression form."""

    def __repr__(self):
        """Return string representation."""
        return f"(* {self.left} {self.right})"

    def __hash__(self):
        """Hash based on left and right."""
        return hash((self.left, self.right))


class Div(BinaryOp):
    """Binary division node ``(/ left right)`` in VNN-LIB S-expression form."""

    def __repr__(self):
        """Return string representation."""
        return f"(/ {self.left} {self.right})"

    def __hash__(self):
        """Hash based on left and right."""
        return hash((self.left, self.right))


class Eq(BinaryOp):
    """Equality constraint node ``(= left right)`` in VNN-LIB S-expression form."""

    def __repr__(self):
        """Return string representation."""
        return f"(= {self.left} {self.right})"

    def __hash__(self):
        """Hash based on left and right."""
        return hash((self.left, self.right))


class Leq(BinaryOp):
    """Less-than-or-equal constraint ``(<= left right)`` in VNN-LIB S-expression form."""

    def __repr__(self):
        """Return string representation."""
        return f"(<= {self.left} {self.right})"

    def __hash__(self):
        """Hash based on left and right."""
        return hash((self.left, self.right))


class Geq(BinaryOp):
    """Greater-than-or-equal constraint ``(>= left right)`` in VNN-LIB S-expression form."""

    def __repr__(self):
        """Return string representation."""
        return f"(>= {self.left} {self.right})"

    def __hash__(self):
        """Hash based on left and right."""
        return hash((self.left, self.right))


class And(NaryOp):
    """Logical conjunction node ``(and a b ...)`` in VNN-LIB S-expression form."""

    def __repr__(self):
        """Return string representation."""
        return f"(and {' '.join(map(str, self.args))})"

    def __hash__(self):
        """Hash based on args tuple."""
        return hash(tuple(self.args))


class Or(NaryOp):
    """Logical disjunction node ``(or a b ...)`` in VNN-LIB S-expression form."""

    def __repr__(self):
        """Return string representation."""
        return f"(or {' '.join(map(str, self.args))})"

    def __hash__(self):
        """Hash based on args tuple."""
        return hash(tuple(self.args))
