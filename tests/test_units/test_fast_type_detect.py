"""Unit tests for _fast_type_detect.py - VNN-LIB type detection and parsing.

Tests type detection (TYPE1-TYPE5), simple pattern parsing, and combined detect+parse operations.
"""

import pytest

from torchvnnlib.fast_type._fast_type_detect import (
    fast_detect_and_parse,
    fast_detect_type,
    parse_simple_patterns,
)
from torchvnnlib.fast_type._utils import VNNLIBType


class TestFastDetectType:
    """Test VNN-LIB type detection."""

    def test_type1_detection(self):
        """Test TYPE1 detection: simple inputs + simple outputs."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (>= Y_0 0.5))",
        ]

        vnnlib_type = fast_detect_type(lines, verbose=False)
        assert vnnlib_type == VNNLIBType.TYPE1, "Should detect TYPE1"

    def test_type2_detection(self):
        """Test TYPE2 detection: simple inputs + OR(AND) outputs."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (or (and (>= Y_0 0.5))))",
        ]

        vnnlib_type = fast_detect_type(lines, verbose=False)
        assert vnnlib_type == VNNLIBType.TYPE2, "Should detect TYPE2"

    def test_type3_detection(self):
        """Test TYPE3 detection: OR(AND) inputs + simple outputs."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (>= X_0 0.0) (<= X_0 1.0))))",
            "(assert (>= Y_0 0.5))",
        ]

        vnnlib_type = fast_detect_type(lines, verbose=False)
        assert vnnlib_type == VNNLIBType.TYPE3, "Should detect TYPE3"

    def test_type4_detection(self):
        """Test TYPE4 detection: OR(AND) inputs + OR(AND) outputs."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (>= X_0 0.0))))",
            "(assert (or (and (>= Y_0 0.5))))",
        ]

        vnnlib_type = fast_detect_type(lines, verbose=False)
        assert vnnlib_type == VNNLIBType.TYPE4, "Should detect TYPE4"

    def test_type5_detection(self):
        """Test TYPE5 detection: mixed OR(AND) with both X and Y."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
            "(assert (or (and (>= X_0 0.0) (>= Y_0 0.5))))",
        ]

        vnnlib_type = fast_detect_type(lines, verbose=False)
        assert vnnlib_type == VNNLIBType.TYPE5, "Should detect TYPE5"

    def test_detect_with_verbose(self, capsys):
        """Test type detection with verbose output."""
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (>= Y_0 0.5))",
        ]

        vnnlib_type = fast_detect_type(lines, verbose=True)

        captured = capsys.readouterr()
        output = captured.out

        assert vnnlib_type == VNNLIBType.TYPE1, "Should detect TYPE1"
        assert "Fast type detection" in output or len(output) > 0, "Should have verbose output"

    def test_unsupported_pattern_raises_error(self):
        """Test that unsupported top-level OR pattern raises ValueError."""
        lines = [
            "(declare-const X_0 Real)",
            "(assert (or (>= X_0 0.0) (>= X_0 1.0)))",  # Top-level OR without AND
        ]

        with pytest.raises(ValueError, match="Unsupported VNN-LIB pattern"):
            fast_detect_type(lines, verbose=False)


class TestParseSimplePatterns:
    """Test simple pattern parsing."""

    def test_parse_simple_input_bounds(self):
        """Test parsing of simple input bound patterns."""
        lines = [
            "(assert (>= X_0 0.5))",
            "(assert (<= X_0 1.5))",
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert len(result["simple_input_bounds"]) == 2, "Should find 2 input bounds"
        assert result["simple_input_bounds"][0] == (
            ">=",
            "X_",
            0,
            0.5,
        ), "First bound should be >= X_0 0.5"
        assert result["simple_input_bounds"][1] == (
            "<=",
            "X_",
            0,
            1.5,
        ), "Second bound should be <= X_0 1.5"

    def test_parse_simple_output_bounds(self):
        """Test parsing of simple output bound patterns."""
        lines = [
            "(assert (>= Y_0 0.2))",
            "(assert (<= Y_1 0.8))",
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert len(result["simple_output_bounds"]) == 2, "Should find 2 output bounds"
        assert result["simple_output_bounds"][0] == (
            ">=",
            "Y_",
            0,
            0.2,
        ), "First bound should be >= Y_0 0.2"
        assert result["simple_output_bounds"][1] == (
            "<=",
            "Y_",
            1,
            0.8,
        ), "Second bound should be <= Y_1 0.8"

    def test_parse_simple_output_constraints(self):
        """Test parsing of simple output constraint patterns (Y vs Y)."""
        lines = [
            "(assert (>= Y_0 Y_1))",
            "(assert (<= Y_1 Y_2))",
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert len(result["simple_output_constrs"]) == 2, "Should find 2 constraints"
        assert result["simple_output_constrs"][0] == (
            ">=",
            "Y_",
            0,
            "Y_",
            1,
        ), "First should be >= Y_0 Y_1"
        assert result["simple_output_constrs"][1] == (
            "<=",
            "Y_",
            1,
            "Y_",
            2,
        ), "Second should be <= Y_1 Y_2"

    def test_parse_mixed_patterns(self):
        """Test parsing with mixed input bounds, output bounds, and constraints."""
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (>= Y_0 Y_1))",
            "(assert (<= Y_0 0.5))",
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert len(result["simple_input_bounds"]) == 2, "Should find 2 input bounds"
        assert len(result["simple_output_constrs"]) == 1, "Should find 1 constraint"
        assert len(result["simple_output_bounds"]) == 1, "Should find 1 output bound"

    def test_parse_equality_bounds(self):
        """Test parsing of equality bounds (Y = value)."""
        lines = [
            "(assert (= Y_0 0.5))",
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert len(result["simple_output_bounds"]) == 1, "Should find 1 equality bound"
        assert result["simple_output_bounds"][0][0] == "=", "Should be equality operator"

    def test_parse_complex_lines_ignored(self):
        """Test that complex patterns are captured as complex_lines."""
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (or (and (>= Y_0 0.5))))",  # Complex pattern
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert len(result["simple_input_bounds"]) == 1, "Should find 1 simple input bound"
        assert len(result["complex_lines"]) == 1, "Should find 1 complex line"
        assert "(or" in result["complex_lines"][0], "Complex line should contain OR pattern"

    def test_parse_empty_lines(self):
        """Test parsing with no assertions."""
        lines = [
            "(declare-const X_0 Real)",
            "(declare-const Y_0 Real)",
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert len(result["simple_input_bounds"]) == 0, "Should find no input bounds"
        assert len(result["simple_output_constrs"]) == 0, "Should find no constraints"
        assert len(result["simple_output_bounds"]) == 0, "Should find no output bounds"
        assert len(result["complex_lines"]) == 2, "Should have 2 complex lines"

    def test_parse_with_verbose(self, capsys):
        """Test parsing with verbose output."""
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (>= Y_0 0.5))",
        ]

        result = parse_simple_patterns(lines, verbose=True)

        captured = capsys.readouterr()
        output = captured.out

        assert len(result["simple_input_bounds"]) == 1, "Should find 1 input bound"
        assert "Parse simple patterns" in output or len(output) > 0, "Should have verbose output"

    def test_parse_scientific_notation(self):
        """Test parsing with scientific notation numbers."""
        lines = [
            "(assert (>= X_0 1.5e-3))",
            "(assert (<= Y_0 2.5e-2))",
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert len(result["simple_input_bounds"]) == 1, "Should find input bound"
        assert len(result["simple_output_bounds"]) == 1, "Should find output bound"

    def test_parse_negative_numbers(self):
        """Test parsing with negative numbers."""
        lines = [
            "(assert (>= X_0 -5.0))",
            "(assert (<= Y_0 -0.5))",
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert result["simple_input_bounds"][0][3] == -5.0, "Should parse negative input bound"
        assert result["simple_output_bounds"][0][3] == -0.5, "Should parse negative output bound"


class TestFastDetectAndParse:
    """Test combined detection and parsing."""

    def test_type1_detect_and_parse(self):
        """Test TYPE1 detection and parsing together."""
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (>= Y_0 0.5))",
        ]

        vnnlib_type, data = fast_detect_and_parse(lines, n_inputs=1, n_outputs=1, verbose=False)

        assert vnnlib_type == VNNLIBType.TYPE1, "Should detect TYPE1"
        assert len(data["simple_input_bounds"]) == 2, "Should parse 2 input bounds"
        assert len(data["simple_output_constrs"]) == 0, "Should have no output constraints"
        assert len(data["complex_lines"]) == 1, "Should have 1 complex line (output bound)"

    def test_type2_detect_and_parse(self):
        """Test TYPE2 detection and parsing together."""
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (or (and (>= Y_0 0.5))))",
        ]

        vnnlib_type, data = fast_detect_and_parse(lines, n_inputs=1, n_outputs=1, verbose=False)

        assert vnnlib_type == VNNLIBType.TYPE2, "Should detect TYPE2"
        assert len(data["simple_input_bounds"]) == 2, "Should parse input bounds"

    def test_type3_detect_and_parse(self):
        """Test TYPE3 detection and parsing together."""
        lines = [
            "(assert (or (and (>= X_0 0.0))))",
            "(assert (>= Y_0 0.5))",
        ]

        vnnlib_type, data = fast_detect_and_parse(lines, n_inputs=1, n_outputs=1, verbose=False)

        assert vnnlib_type == VNNLIBType.TYPE3, "Should detect TYPE3"
        assert len(data["complex_lines"]) >= 1, "Should have complex lines"

    def test_type4_detect_and_parse(self):
        """Test TYPE4 detection and parsing together."""
        lines = [
            "(assert (or (and (>= X_0 0.0))))",
            "(assert (or (and (>= Y_0 0.5))))",
        ]

        vnnlib_type, _data = fast_detect_and_parse(lines, n_inputs=1, n_outputs=1, verbose=False)

        assert vnnlib_type == VNNLIBType.TYPE4, "Should detect TYPE4"

    def test_type5_detect_and_parse(self):
        """Test TYPE5 detection and parsing together."""
        lines = [
            "(assert (or (and (>= X_0 0.0) (>= Y_0 0.5))))",
        ]

        vnnlib_type, _data = fast_detect_and_parse(lines, n_inputs=1, n_outputs=1, verbose=False)

        assert vnnlib_type == VNNLIBType.TYPE5, "Should detect TYPE5"

    def test_detect_and_parse_with_verbose(self, capsys):
        """Test detect+parse with verbose output."""
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (>= Y_0 0.5))",
        ]

        vnnlib_type, _data = fast_detect_and_parse(lines, n_inputs=1, n_outputs=1, verbose=True)

        captured = capsys.readouterr()
        output = captured.out

        assert vnnlib_type == VNNLIBType.TYPE1, "Should detect TYPE1"
        assert "Fast detect+parse" in output or len(output) > 0, "Should have verbose output"

    def test_metadata_in_result(self):
        """Test that metadata is included in result."""
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (<= X_0 1.0))",
            "(assert (>= Y_0 0.5))",
        ]

        _vnnlib_type, data = fast_detect_and_parse(lines, n_inputs=1, n_outputs=1, verbose=False)

        assert "metadata" in data, "Should include metadata"
        assert data["metadata"]["n_lines"] == 3, "Should count 3 lines"
        assert data["metadata"]["has_simple_input"], "Should detect simple input"
        assert data["metadata"]["has_simple_output"], "Should detect simple output"

    def test_realistic_benchmark_pattern(self):
        """Test with realistic VNN-COMP benchmark pattern."""
        lines = [
            "(assert (>= X_0 -1.0))",
            "(assert (<= X_0 1.0))",
            "(assert (>= X_1 0.0))",
            "(assert (<= X_1 2.0))",
            "(assert (>= Y_0 -0.5))",
            "(assert (<= Y_0 0.5))",
        ]

        vnnlib_type, data = fast_detect_and_parse(lines, n_inputs=2, n_outputs=1, verbose=False)

        assert vnnlib_type == VNNLIBType.TYPE1, "Should detect TYPE1"
        assert len(data["simple_input_bounds"]) == 4, "Should parse 4 input bounds"


class TestVNNLIBTypeEnum:
    """Test VNNLIBType enum values."""

    def test_type_enum_values(self):
        """Test that all VNNLIBType values exist."""
        assert hasattr(VNNLIBType, "TYPE1"), "Should have TYPE1"
        assert hasattr(VNNLIBType, "TYPE2"), "Should have TYPE2"
        assert hasattr(VNNLIBType, "TYPE3"), "Should have TYPE3"
        assert hasattr(VNNLIBType, "TYPE4"), "Should have TYPE4"
        assert hasattr(VNNLIBType, "TYPE5"), "Should have TYPE5"
        assert hasattr(VNNLIBType, "COMPLEX"), "Should have COMPLEX"

    def test_type_enum_string_representation(self):
        """Test that enum values have proper string representation."""
        assert VNNLIBType.TYPE1.name == "TYPE1", "TYPE1 name should be TYPE1"
        assert VNNLIBType.TYPE2.name == "TYPE2", "TYPE2 name should be TYPE2"
        assert VNNLIBType.TYPE3.name == "TYPE3", "TYPE3 name should be TYPE3"
        assert VNNLIBType.TYPE4.name == "TYPE4", "TYPE4 name should be TYPE4"
        assert VNNLIBType.TYPE5.name == "TYPE5", "TYPE5 name should be TYPE5"


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_empty_lines_list(self):
        """Test handling of empty lines list."""
        lines = []

        vnnlib_type = fast_detect_type(lines, verbose=False)

        # Should return some default type
        assert isinstance(vnnlib_type, VNNLIBType), "Should return a VNNLIBType"

    def test_mixed_whitespace_in_patterns(self):
        """Test patterns with various whitespace."""
        lines = [
            "( assert   (>=   X_0   0.0   )  )",  # Extra whitespace
            "(assert(<= X_0 1.0))",  # No spaces
        ]

        result = parse_simple_patterns(lines, verbose=False)

        # Should still parse successfully
        assert isinstance(result, dict), "Should return a dictionary"

    def test_multiple_or_blocks(self):
        """Test handling of multiple OR blocks."""
        lines = [
            "(assert (or (and (>= X_0 0.0) (>= Y_0 0.5))))",
            "(assert (or (and (>= X_0 1.0) (>= Y_0 1.5))))",
        ]

        vnnlib_type, _data = fast_detect_and_parse(lines, n_inputs=1, n_outputs=1, verbose=False)

        # Should detect as TYPE5 or COMPLEX
        assert isinstance(vnnlib_type, VNNLIBType), "Should return a VNNLIBType"

    def test_large_variable_indices(self):
        """Test parsing with large variable indices."""
        lines = [
            "(assert (>= X_99 0.0))",
            "(assert (<= Y_199 1.0))",
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert result["simple_input_bounds"][0][2] == 99, "Should parse large X index"
        assert result["simple_output_bounds"][0][2] == 199, "Should parse large Y index"
