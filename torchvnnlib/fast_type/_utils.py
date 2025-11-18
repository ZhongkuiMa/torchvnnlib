"""VNN-LIB Type Detection Module.

This module detects the structural type of VNN-LIB files to enable
type-specific fast-path processing.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "VNNLIBType",
    "SIMPLE_INPUT_BOUND_PATTERN",
    "SIMPLE_OUTPUT_CONSTRAINT_PATTERN",
    "SIMPLE_OUTPUT_BOUND_PATTERN",
]


import re
from enum import Enum


class VNNLIBType(Enum):
    """VNN-LIB file types based on structure."""

    TYPE1 = "Type1: (and ...input_constraints... ...output_constraints...)"
    TYPE2 = "Type2: (and ...input_constraints... (or ...output_constraints...))"
    TYPE3 = "Type3: (and (or ...input_constraints...) ...output_constraints...)"
    TYPE4 = "Type4: (and (or ...input_constraints...) (or ...output_constraints...))"
    TYPE5 = "Type5: (or (and ...input_constraints... ...output_constraints...))"
    COMPLEX = "Complex: Requires AST-based processing"


# Pattern to detect top-level OR/AND structures
TOP_LEVEL_OR_PATTERN = re.compile(r"^\s*\(\s*assert\s+\(\s*or\s+", re.IGNORECASE)
NESTED_OR_PATTERN = re.compile(r"\(\s*or\s+", re.IGNORECASE)
NESTED_AND_PATTERN = re.compile(r"\(\s*and\s+", re.IGNORECASE)

# Pre-compiled regex patterns for simple expressions
# Pattern for simple input bounds: (assert (<=/>=|= X_5 0.123))
SIMPLE_INPUT_BOUND_PATTERN = re.compile(
    r"^\s*\(\s*assert\s+\(\s*(<=|>=|=)\s+(X_)(\d+)\s+([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\)\s*\)\s*$"
)

# Pattern for simple output constraints with two Y variables: (assert (<=/>=  Y_0 Y_1))
SIMPLE_OUTPUT_CONSTRAINT_PATTERN = re.compile(
    r"^\s*\(\s*assert\s+\(\s*(<=|>=)\s+(Y_)(\d+)\s+(Y_)(\d+)\s*\)\s*\)\s*$"
)

# Pattern for simple output bounds with one Y variable: (assert (<=/>=|= Y_5 0.123))
SIMPLE_OUTPUT_BOUND_PATTERN = re.compile(
    r"^\s*\(\s*assert\s+\(\s*(<=|>=|=)\s+(Y_)(\d+)\s+([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\)\s*\)\s*$"
)
