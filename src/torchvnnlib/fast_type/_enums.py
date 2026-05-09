"""Enum types for the fast_type subpackage."""

__docformat__ = "restructuredtext"
__all__ = ["VNNLIBType"]

from enum import IntEnum, unique


@unique
class VNNLIBType(IntEnum):
    """Structural type of a VNN-LIB file, used to select the fast-path processor."""

    TYPE1 = 1
    """Simple inputs + simple outputs: ``(and ...input_constraints... ...output_constraints...)``."""
    TYPE2 = 2
    """OR(AND) inputs + simple outputs: ``(and ...input_constraints... (or ...output_constraints...))``."""
    TYPE3 = 3
    """Simple inputs + OR(AND) outputs: ``(and (or ...input_constraints...) ...output_constraints...)``."""
    TYPE4 = 4
    """OR(AND) inputs + OR(AND) outputs: ``(and (or ...input_constraints...) (or ...output_constraints...))``."""
    TYPE5 = 5
    """Mixed OR(AND) with both X and Y: ``(or (and ...input_constraints... ...output_constraints...))``."""
    COMPLEX = 6
    """Complex structure requiring AST-based processing."""

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.name}"
