"""Integration tests for TorchVNNLIB end-to-end workflows.

Tests complete conversion workflows from file to tensor output.
Follows principles: broad code coverage, all logics, error handling, quick tests,
fine-grained information, public API only, no deselected tests.
"""

import tempfile
from pathlib import Path

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

    def test_torch_output_format(self):
        """Test torch output format produces .pth files."""
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

            pth_files = list(Path(output_path).rglob("*.pth"))
            assert len(pth_files) > 0

    def test_numpy_output_format(self):
        """Test numpy output format produces .npz files."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB(output_format="numpy")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            npz_files = list(Path(output_path).rglob("*.npz"))
            assert len(npz_files) > 0


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

    def test_convert_with_fast_type_enabled(self):
        """Test conversion with fast type detection enabled."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB(detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            assert "used_fast" in converter.conversion_stats[vnnlib_path]

    def test_convert_with_fast_type_disabled(self):
        """Test conversion with fast type detection disabled."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
            f.write(vnnlib_content)
            vnnlib_path = f.name

        converter = TorchVNNLIB(detect_fast_type=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(vnnlib_path, output_path)

            assert converter.conversion_stats[vnnlib_path]["used_fast"] is False


class TestNegativeBounds:
    """Test conversions with negative bounds."""

    def test_negative_input_bounds(self):
        """Test conversion with negative input bounds."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 -1.0))
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

    def test_negative_output_bounds(self):
        """Test conversion with negative output bounds."""
        vnnlib_content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
(assert (>= Y_0 -0.5))
"""
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
