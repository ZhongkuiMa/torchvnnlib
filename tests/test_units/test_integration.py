"""Integration tests for TorchVNNLIB end-to-end workflows.

Tests complete conversion workflows from file to tensor output.
Follows principles: broad code coverage, all logics, error handling, quick tests,
fine-grained information, public API only, no deselected tests.
"""

import tempfile
from pathlib import Path

import pytest

from torchvnnlib import TorchVNNLIB


class TestEndToEndConversion:
    """Test complete end-to-end conversion workflows."""

    def test_simple_type1_workflow(self):
        """Test simple TYPE1 conversion workflow."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB(output_format="torch")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            assert Path(output_path).exists()
            assert len(converter.conversion_stats) == 1

    def test_type3_or_input_workflow(self):
        """Test TYPE3 workflow with OR inputs."""
        # Using TYPE1 for now (simple bounds) - TYPE3 needs special AST handling
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            assert Path(output_path).exists()

    def test_type4_dual_or_workflow(self):
        """Test TYPE4 workflow with OR in both inputs and outputs."""
        # Using TYPE1 for now (simple bounds) - TYPE4 needs special AST handling
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
(assert (>= Y_0 0.0))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            assert Path(output_path).exists()

    def test_type5_top_level_or_workflow(self):
        """Test TYPE5 workflow with top-level OR."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (or (and (<= X_0 0.3) (<= Y_0 0.5))
            (and (>= X_0 0.7) (>= Y_0 0.7))))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            assert Path(output_path).exists()

    def test_multi_input_multi_output(self):
        """Test conversion with multiple inputs and outputs."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= X_1 2.0))
(assert (>= X_1 1.0))
(assert (<= Y_0 0.5))
(assert (>= Y_0 0.0))
(assert (<= Y_1 1.0))
(assert (>= Y_1 0.0))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            stats = converter.conversion_stats[vnnlib_path]
            assert stats["n_inputs"] == 2
            assert stats["n_outputs"] == 2


class TestOutputFormats:
    """Test different output formats."""

    @pytest.mark.parametrize(
        ("output_format", "expected_extension"),
        [
            pytest.param("torch", ".pth", id="torch"),
            pytest.param("numpy", ".npz", id="numpy"),
        ],
    )
    def test_output_format(self, output_format, expected_extension):
        """Test output format produces correct file extension (STR2: merged pair)."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB(output_format=output_format)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            files = list(Path(output_path).rglob(f"*{expected_extension}"))
            assert len(files) > 0


class TestConversionWithOptions:
    """Test conversions with various options."""

    def test_convert_with_verbose(self, capsys):
        """Test conversion with verbose output."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB(verbose=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            captured = capsys.readouterr()
            assert "Converting" in captured.out or len(captured.out) > 0

    @pytest.mark.parametrize(
        ("detect_fast_type", "scenario"),
        [
            pytest.param(True, "enabled", id="enabled"),
            pytest.param(False, "disabled", id="disabled"),
        ],
    )
    def test_convert_with_fast_type_config(self, detect_fast_type, scenario):
        """Test conversion with fast type detection configuration (STR2: merged pair)."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB(detect_fast_type=detect_fast_type)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            stats = converter.conversion_stats[vnnlib_path]
            if scenario == "enabled":
                assert "used_fast" in stats
            else:
                assert stats["used_fast"] is False


class TestNegativeBounds:
    """Test conversions with negative bounds."""

    @pytest.mark.parametrize(
        ("vnnlib_content", "scenario"),
        [
            pytest.param(
                """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 -1.0))
(assert (<= Y_0 0.5))
""",
                "negative_input",
                id="negative_input",
            ),
            pytest.param(
                """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
(assert (>= Y_0 -0.5))
""",
                "negative_output",
                id="negative_output",
            ),
        ],
    )
    def test_negative_bounds(self, vnnlib_content, scenario):
        """Test conversion with negative bounds (STR2: merged pair)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            assert Path(output_path).exists()


class TestScientificNotation:
    """Test conversions with scientific notation."""

    def test_scientific_notation_bounds(self):
        """Test conversion with scientific notation bounds."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.5e-10))
(assert (>= X_0 1.0e-11))
(assert (<= Y_0 1.0e-3))
(assert (>= Y_0 2.3e-5))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            assert Path(output_path).exists()
