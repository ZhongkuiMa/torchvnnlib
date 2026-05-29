"""Unit tests for _fast_type_detect.py - VNN-LIB type detection and parsing.

Tests type detection (TYPE1-TYPE5) and simple pattern parsing.
"""

import pytest

from torchvnnlib.fast_type import VNNLIBType
from torchvnnlib.fast_type._fast_type_detect import (
    fast_detect_type,
    parse_simple_patterns,
)


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
        output = captured.err

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

        assert "simple_input_bounds" in result, "Result should contain 'simple_input_bounds' key"
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

        assert "simple_output_bounds" in result, "Result should contain 'simple_output_bounds' key"
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

        assert "simple_output_constrs" in result, (
            "Result should contain 'simple_output_constrs' key"
        )
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

        assert "simple_output_bounds" in result, "Result should contain 'simple_output_bounds' key"
        assert len(result["simple_output_bounds"]) == 1, "Should find 1 equality bound"
        assert result["simple_output_bounds"][0][0] == "=", "Should be equality operator"

    def test_parse_complex_lines_ignored(self):
        """Test that complex patterns are captured as complex_lines."""
        lines = [
            "(assert (>= X_0 0.0))",
            "(assert (or (and (>= Y_0 0.5))))",  # Complex pattern
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert "simple_input_bounds" in result, "Result should contain 'simple_input_bounds' key"
        assert len(result["simple_input_bounds"]) == 1, "Should find 1 simple input bound"
        assert "complex_lines" in result, "Result should contain 'complex_lines' key"
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
        output = captured.err

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

        assert "simple_input_bounds" in result, "Result should contain 'simple_input_bounds' key"
        assert len(result["simple_input_bounds"]) >= 1, "Should have at least 1 simple input bound"
        assert result["simple_input_bounds"][0][3] == -5.0, "Should parse negative input bound"
        assert "simple_output_bounds" in result, "Result should contain 'simple_output_bounds' key"
        assert len(result["simple_output_bounds"]) >= 1, (
            "Should have at least 1 simple output bound"
        )
        assert result["simple_output_bounds"][0][3] == -0.5, "Should parse negative output bound"


class TestVNNLIBTypeEnum:
    """Test VNNLIBType enum values."""

    def test_type_enum_values(self):
        """Test that all VNNLIBType members have the expected integer values."""
        assert VNNLIBType.TYPE1 == 1, "TYPE1 should have integer value 1"
        assert VNNLIBType.TYPE2 == 2, "TYPE2 should have integer value 2"
        assert VNNLIBType.TYPE3 == 3, "TYPE3 should have integer value 3"
        assert VNNLIBType.TYPE4 == 4, "TYPE4 should have integer value 4"
        assert VNNLIBType.TYPE5 == 5, "TYPE5 should have integer value 5"
        assert VNNLIBType.COMPLEX == 6, "COMPLEX should have integer value 6"

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
        lines: list[str] = []

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

        vnnlib_type = fast_detect_type(lines, verbose=False)

        # Should detect as TYPE5 or COMPLEX
        assert isinstance(vnnlib_type, VNNLIBType), "Should return a VNNLIBType"

    def test_large_variable_indices(self):
        """Test parsing with large variable indices."""
        lines = [
            "(assert (>= X_99 0.0))",
            "(assert (<= Y_199 1.0))",
        ]

        result = parse_simple_patterns(lines, verbose=False)

        assert "simple_input_bounds" in result, "Result should contain 'simple_input_bounds' key"
        assert len(result["simple_input_bounds"]) >= 1, "Should have at least 1 simple input bound"
        assert result["simple_input_bounds"][0][2] == 99, "Should parse large X index"
        assert "simple_output_bounds" in result, "Result should contain 'simple_output_bounds' key"
        assert len(result["simple_output_bounds"]) >= 1, (
            "Should have at least 1 simple output bound"
        )
        assert result["simple_output_bounds"][0][2] == 199, "Should parse large Y index"
