"""Fast type-specific VNN-LIB processors.

Optimized processors for common VNN-LIB patterns (Type1-5).
"""

__docformat__ = "restructuredtext"
__all__ = [
    "ParsedPatterns",
    "VNNLIBType",
    "convert_simple_input_bounds",
    "fast_detect_type",
    "iter_and_blocks",
    "parse_and_block",
    "parse_dual_or_blocks",
    "parse_input_bounds_block",
    "parse_input_or_block",
    "parse_output_and_block",
    "parse_output_or_block",
    "parse_simple_patterns",
    "process_type1",
    "process_type2",
    "process_type3",
    "process_type4",
    "process_type5",
]

from torchvnnlib.fast_type._enums import VNNLIBType
from torchvnnlib.fast_type._fast_type_detect import (
    ParsedPatterns,
    fast_detect_type,
    parse_simple_patterns,
)
from torchvnnlib.fast_type._type1_processor import process_type1
from torchvnnlib.fast_type._type2_processor import process_type2
from torchvnnlib.fast_type._type3_processor import process_type3
from torchvnnlib.fast_type._type4_processor import process_type4
from torchvnnlib.fast_type._type5_processor import process_type5
from torchvnnlib.fast_type._utils import (
    convert_simple_input_bounds,
    iter_and_blocks,
    parse_and_block,
    parse_dual_or_blocks,
    parse_input_bounds_block,
    parse_input_or_block,
    parse_output_and_block,
    parse_output_or_block,
)
