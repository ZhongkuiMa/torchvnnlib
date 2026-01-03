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
    def __init__(self):
        self._has_input_vars = None  # Cache for input variable presence
        self._has_output_vars = None  # Cache for output variable presence

    def __repr__(self):
        raise NotImplementedError("This should be implemented in subclasses")

    def __eq__(self, other):
        raise NotImplementedError("This should be implemented in subclasses")

    def __hash__(self):
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
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __repr__(self):
        return f"{self.value}"

    def __eq__(self, other):
        if not isinstance(other, Cst):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Var(Expr):
    def __init__(self, name: str):
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
        return f"{self.name}"

    def __eq__(self, other):
        if not isinstance(other, Var):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def _compute_has_input_vars(self) -> bool:
        return self.var_type == "X"

    def _compute_has_output_vars(self) -> bool:
        return self.var_type == "Y"


class UnaryOp(Expr):
    def __init__(self, arg: Expr):
        super().__init__()
        self.arg = arg

    def __repr__(self):
        raise RuntimeError("This should be implemented in subclasses")

    def __eq__(self, other):
        if not isinstance(other, UnaryOp):
            return False
        return self.arg == other.arg

    def __hash__(self):
        return hash(self.arg)

    def _compute_has_input_vars(self) -> bool:
        return self.arg.has_input_vars

    def _compute_has_output_vars(self) -> bool:
        return self.arg.has_output_vars


class BinaryOp(Expr):
    def __init__(self, left: Expr, right: Expr):
        super().__init__()
        self.left = left
        self.right = right

    def __repr__(self):
        raise RuntimeError("This should be implemented in subclasses")

    def __eq__(self, other):
        if not isinstance(other, BinaryOp):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash((self.left, self.right))

    def _compute_has_input_vars(self) -> bool:
        return self.left.has_input_vars or self.right.has_input_vars

    def _compute_has_output_vars(self) -> bool:
        return self.left.has_output_vars or self.right.has_output_vars


class NaryOp(Expr):
    def __init__(self, args: list[Expr]):
        super().__init__()
        self.args = args

    def __repr__(self):
        raise RuntimeError("This should be implemented in subclasses")

    def __eq__(self, other):
        if not isinstance(other, NaryOp):
            return False
        if len(self.args) != len(other.args):
            return False
        return all(self.args[i] == other.args[i] for i in range(len(self.args)))

    def __hash__(self):
        return hash(tuple(self.args))

    def __iter__(self):
        return iter(self.args)

    def _compute_has_input_vars(self) -> bool:
        return any(arg.has_input_vars for arg in self.args)

    def _compute_has_output_vars(self) -> bool:
        return any(arg.has_output_vars for arg in self.args)


class Add(NaryOp):
    def __repr__(self):
        return f"(+ {' '.join(map(str, self.args))})"

    def __hash__(self):
        return hash(tuple(self.args))


class Sub(BinaryOp):
    def __repr__(self):
        return f"(- {self.left} {self.right})"

    def __hash__(self):
        return hash((self.left, self.right))


class Mul(BinaryOp):
    def __repr__(self):
        return f"(* {self.left} {self.right})"

    def __hash__(self):
        return hash((self.left, self.right))


class Div(BinaryOp):
    def __repr__(self):
        return f"(/ {self.left} {self.right})"

    def __hash__(self):
        return hash((self.left, self.right))


class Eq(BinaryOp):
    def __repr__(self):
        return f"(= {self.left} {self.right})"

    def __hash__(self):
        return hash((self.left, self.right))


class Leq(BinaryOp):
    def __repr__(self):
        return f"(<= {self.left} {self.right})"

    def __hash__(self):
        return hash((self.left, self.right))


class Geq(BinaryOp):
    def __repr__(self):
        return f"(>= {self.left} {self.right})"

    def __hash__(self):
        return hash((self.left, self.right))


class And(NaryOp):
    def __repr__(self):
        return f"(and {' '.join(map(str, self.args))})"

    def __hash__(self):
        return hash(tuple(self.args))


class Or(NaryOp):
    def __repr__(self):
        return f"(or {' '.join(map(str, self.args))})"

    def __hash__(self):
        return hash(tuple(self.args))
