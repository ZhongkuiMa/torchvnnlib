__docformat__ = "restructuredtext"
__all__ = [
    "Expr",
    "Cst",
    "Var",
    "UnaryOp",
    "BinaryOp",
    "NaryOp",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Eq",
    "Leq",
    "Geq",
    "And",
    "Or",
]


class Expr:
    def __init__(self):
        pass

    def __repr__(self):
        raise NotImplementedError("This should be implemented in subclasses")

    def __eq__(self, other):
        raise NotImplementedError("This should be implemented in subclasses")

    def __hash__(self):
        return hash(self.__repr__())


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
        if not (
            (name.startswith("X") or name.startswith("Y"))
            and name[1] == "_"
            and name[2:].isdigit()
        ):
            raise ValueError(f"Variable name must start with 'X' or 'Y' but {name}.")

    def __repr__(self):
        return f"{self.name}"

    def __eq__(self, other):
        if not isinstance(other, Var):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


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
        for i in range(len(self.args)):
            if self.args[i] != other.args[i]:
                return False
        return True

    def __hash__(self):
        return hash(tuple(self.args))

    def __iter__(self):
        return iter(self.args)


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
