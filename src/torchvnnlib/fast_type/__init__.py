"""Fast type-specific VNN-LIB processors.

Optimized processors for common VNN-LIB patterns (Type1-5).
"""

__docformat__ = "restructuredtext"
__all__ = [
    "INPUT_BOUND_INNER_PATTERN",
    "OUTPUT_BOUND_INNER_PATTERN",
    "OUTPUT_CONSTRAINT_INNER_PATTERN",
    "SIMPLE_INPUT_BOUND_PATTERN",
    "SIMPLE_OUTPUT_BOUND_PATTERN",
    "SIMPLE_OUTPUT_CONSTRAINT_PATTERN",
    "VNNLIBType",
    "convert_simple_input_bounds",
    "fast_detect_and_parse",
    "fast_detect_type",
    "parse_and_block",
    "parse_dual_or_blocks",
    "parse_input_bounds_block",
    "parse_input_or_block",
    "parse_or_block",
    "parse_output_and_block",
    "parse_output_or_block",
    "parse_simple_patterns",
    "process_type1",
    "process_type2",
    "process_type3",
    "process_type4",
    "process_type5",
]

from torchvnnlib.fast_type._fast_type_detect import (
    VNNLIBType,
    fast_detect_and_parse,
    fast_detect_type,
    parse_simple_patterns,
)
from torchvnnlib.fast_type._type1_processor import process_type1
from torchvnnlib.fast_type._type2_processor import process_type2
from torchvnnlib.fast_type._type3_processor import process_type3
from torchvnnlib.fast_type._type4_processor import process_type4
from torchvnnlib.fast_type._type5_processor import process_type5
from torchvnnlib.fast_type._utils import (
    INPUT_BOUND_INNER_PATTERN,
    OUTPUT_BOUND_INNER_PATTERN,
    OUTPUT_CONSTRAINT_INNER_PATTERN,
    SIMPLE_INPUT_BOUND_PATTERN,
    SIMPLE_OUTPUT_BOUND_PATTERN,
    SIMPLE_OUTPUT_CONSTRAINT_PATTERN,
    convert_simple_input_bounds,
    parse_and_block,
    parse_dual_or_blocks,
    parse_input_bounds_block,
    parse_input_or_block,
    parse_or_block,
    parse_output_and_block,
    parse_output_or_block,
)
