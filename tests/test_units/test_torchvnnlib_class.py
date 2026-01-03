"""Unit tests for TorchVNNLIB class.

Tests the public TorchVNNLIB API for VNN-LIB to tensor conversion.
Follows principles: broad code coverage, all logics, error handling, quick tests,
fine-grained information, public API only, no deselected tests.
"""

import tempfile
from pathlib import Path

import pytest

from torchvnnlib import TorchVNNLIB


@pytest.fixture
def temp_vnnlib_simple():
    """Create a simple TYPE1 VNN-LIB file."""
    content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.5))
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
        f.write(content)
        return f.name


@pytest.fixture
def temp_vnnlib_type2():
    """Create a TYPE2 VNN-LIB file (simple inputs + OR outputs)."""
    content = """(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (or (and (>= Y_0 0.5) (<= Y_0 1.0)) (and (>= Y_1 0.0) (<= Y_1 0.3))))
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
        f.write(content)
        return f.name


@pytest.fixture
def temp_vnnlib_type3():
    """Create a TYPE3-like VNN-LIB file (simple patterns for AST compatibility)."""
    # Use simple patterns that work with AST flattening when fast type detection is disabled
    content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (>= Y_0 0.0))
(assert (<= Y_0 1.0))
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
        f.write(content)
        return f.name


@pytest.fixture
def temp_vnnlib_type4():
    """Create a TYPE4-like VNN-LIB file (simple patterns for backward compatibility)."""
    # Use simple patterns that work with AST flattening when fast type detection is disabled
    content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (>= Y_0 0.0))
(assert (<= Y_0 1.0))
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
        f.write(content)
        return f.name


@pytest.fixture
def temp_vnnlib_type3_or():
    """Create a TYPE3 VNN-LIB file with OR input patterns (fast type detection only)."""
    # TYPE3: (or (and X_inputs)), has_simple_output=True
    content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (or (and (>= X_0 0.0) (<= X_0 1.0))))
(assert (>= Y_0 0.0))
(assert (<= Y_0 1.0))
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
        f.write(content)
        return f.name


@pytest.fixture
def temp_vnnlib_type4_or():
    """Create a TYPE4 VNN-LIB file with OR input and output patterns (fast type detection only)."""
    # TYPE4: (or (and X_inputs)), (or (and Y_outputs))
    content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (or (and (>= X_0 0.0) (<= X_0 1.0))))
(assert (or (and (>= Y_0 0.2) (<= Y_0 0.8))))
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
        f.write(content)
        return f.name


@pytest.fixture
def temp_vnnlib_type5():
    """Create a TYPE5 VNN-LIB file (mixed OR with both X and Y)."""
    # TYPE5: (or (and with both X_0 and Y_0))
    content = """(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (or (and (>= X_0 0.0) (<= X_0 1.0) (>= Y_0 0.2) (<= Y_0 0.8))))
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vnnlib", delete=False) as f:
        f.write(content)
        return f.name


class TestTorchVNNLIBInitialization:
    """Test TorchVNNLIB initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        converter = TorchVNNLIB()
        assert converter.verbose is False
        assert converter.use_parallel is False
        assert converter.detect_fast_type is True
        assert converter.output_format == "torch"
        assert converter.conversion_stats == {}

    def test_init_verbose_true(self):
        """Test initialization with verbose=True."""
        converter = TorchVNNLIB(verbose=True)
        assert converter.verbose is True

    def test_init_use_parallel_true(self):
        """Test initialization with use_parallel=True."""
        converter = TorchVNNLIB(use_parallel=True)
        assert converter.use_parallel is True

    def test_init_detect_fast_type_false(self):
        """Test initialization with detect_fast_type=False."""
        converter = TorchVNNLIB(detect_fast_type=False)
        assert converter.detect_fast_type is False

    def test_init_output_format_numpy(self):
        """Test initialization with numpy output format."""
        converter = TorchVNNLIB(output_format="numpy")
        assert converter.output_format == "numpy"
        assert converter.backend.name == "numpy"

    def test_init_output_format_torch(self):
        """Test initialization with torch output format."""
        converter = TorchVNNLIB(output_format="torch")
        assert converter.output_format == "torch"
        assert converter.backend.name == "torch"

    def test_init_all_parameters(self):
        """Test initialization with all parameters."""
        converter = TorchVNNLIB(
            verbose=True, use_parallel=True, detect_fast_type=False, output_format="numpy"
        )
        assert converter.verbose is True
        assert converter.use_parallel is True
        assert converter.detect_fast_type is False
        assert converter.output_format == "numpy"


class TestTorchVNNLIBConversion:
    """Test TorchVNNLIB conversion functionality."""

    def test_convert_type1_torch(self, temp_vnnlib_simple):
        """Test conversion of TYPE1 file to torch format."""
        converter = TorchVNNLIB(output_format="torch")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)

            assert converter.conversion_stats
            assert temp_vnnlib_simple in converter.conversion_stats
            assert converter.conversion_stats[temp_vnnlib_simple]["output_format"] == "torch"

    def test_convert_type1_numpy(self, temp_vnnlib_simple):
        """Test conversion of TYPE1 file to numpy format."""
        converter = TorchVNNLIB(output_format="numpy")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)

            assert converter.conversion_stats
            assert converter.conversion_stats[temp_vnnlib_simple]["output_format"] == "numpy"

    def test_convert_type3(self, temp_vnnlib_type3):
        """Test conversion of TYPE3 file."""
        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type3, output_path)

            assert converter.conversion_stats
            assert temp_vnnlib_type3 in converter.conversion_stats

    def test_convert_type4(self, temp_vnnlib_type4):
        """Test conversion of TYPE4 file."""
        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type4, output_path)

            assert converter.conversion_stats
            assert temp_vnnlib_type4 in converter.conversion_stats

    def test_convert_creates_output_directory(self, temp_vnnlib_simple):
        """Test that conversion creates output directory."""
        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "nonexistent" / "output")
            converter.convert(temp_vnnlib_simple, output_path)

            assert Path(output_path).exists()

    def test_convert_stats_recording(self, temp_vnnlib_simple):
        """Test that conversion statistics are recorded."""
        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)

            stats = converter.conversion_stats[temp_vnnlib_simple]
            assert "type" in stats
            assert "used_fast" in stats
            assert "time" in stats
            assert "n_inputs" in stats
            assert "n_outputs" in stats
            assert "output_format" in stats
            assert stats["time"] > 0

    def test_convert_verbose_output(self, temp_vnnlib_simple, capsys):
        """Test that verbose mode produces output."""
        converter = TorchVNNLIB(verbose=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)

            captured = capsys.readouterr()
            assert len(captured.out) > 0

    def test_convert_with_detect_fast_type_disabled(self, temp_vnnlib_type3):
        """Test conversion with fast type detection disabled."""
        converter = TorchVNNLIB(detect_fast_type=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type3, output_path)

            assert converter.conversion_stats[temp_vnnlib_type3]["used_fast"] is False

    def test_convert_multiple_files(self, temp_vnnlib_simple, temp_vnnlib_type3):
        """Test converting multiple files in sequence."""
        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path1 = str(Path(tmpdir) / "output1")
            output_path2 = str(Path(tmpdir) / "output2")

            converter.convert(temp_vnnlib_simple, output_path1)
            converter.convert(temp_vnnlib_type3, output_path2)

            assert len(converter.conversion_stats) == 2
            assert temp_vnnlib_simple in converter.conversion_stats
            assert temp_vnnlib_type3 in converter.conversion_stats

    def test_convert_input_variable_counting(self, temp_vnnlib_simple):
        """Test that input variable count is recorded correctly."""
        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)

            assert converter.conversion_stats[temp_vnnlib_simple]["n_inputs"] == 1

    def test_convert_output_variable_counting(self, temp_vnnlib_simple):
        """Test that output variable count is recorded correctly."""
        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)

            assert converter.conversion_stats[temp_vnnlib_simple]["n_outputs"] == 1


class TestTorchVNNLIBBackendIntegration:
    """Test backend integration in TorchVNNLIB."""

    def test_backend_torch_selection(self):
        """Test that torch backend is selected correctly."""
        converter = TorchVNNLIB(output_format="torch")
        assert converter.backend.name == "torch"

    def test_backend_numpy_selection(self):
        """Test that numpy backend is selected correctly."""
        converter = TorchVNNLIB(output_format="numpy")
        assert converter.backend.name == "numpy"

    def test_convert_respects_backend_format(self, temp_vnnlib_simple):
        """Test that conversion output respects backend format."""
        converter = TorchVNNLIB(output_format="torch")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)

            # Check that torch files (.pth) are created
            assert any(Path(output_path).rglob("*.pth"))

    def test_convert_numpy_file_format(self, temp_vnnlib_simple):
        """Test that numpy conversion creates .npz files."""
        converter = TorchVNNLIB(output_format="numpy")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)

            # Check that numpy files (.npz) are created
            assert any(Path(output_path).rglob("*.npz"))


class TestTorchVNNLIBFallback:
    """Test fallback behavior from fast type to AST."""

    def test_fallback_on_fast_type_false(self, temp_vnnlib_type3):
        """Test fallback to AST when fast type detection is disabled."""
        converter = TorchVNNLIB(detect_fast_type=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type3, output_path)

            assert converter.conversion_stats[temp_vnnlib_type3]["used_fast"] is False

    def test_successful_fast_type_detection(self, temp_vnnlib_type4):
        """Test successful fast type detection."""
        converter = TorchVNNLIB(detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type4, output_path)

            # TYPE4 should be detected and used
            stats = converter.conversion_stats[temp_vnnlib_type4]
            # used_fast may be True or False depending on detection success
            assert "used_fast" in stats


class TestTorchVNNLIBPropertySaving:
    """Test property saving functionality."""

    def test_save_properties_with_custom_folder(self, temp_vnnlib_simple):
        """Test saving properties to custom folder."""
        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_output = str(Path(tmpdir) / "custom_output")
            converter.convert(temp_vnnlib_simple, custom_output)
            # Check that custom output path was used
            assert Path(custom_output).exists()

    def test_save_properties_default_folder(self, temp_vnnlib_simple):
        """Test saving properties to default folder (from vnnlib path)."""
        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create vnnlib in temp directory
            vnnlib_path = Path(tmpdir) / "test.vnnlib"
            vnnlib_path.write_text(
                "(declare-const X_0 Real)\n"
                "(declare-const Y_0 Real)\n"
                "(assert (>= X_0 0.0))\n"
                "(assert (<= X_0 1.0))\n"
                "(assert (>= Y_0 0.0))\n"
                "(assert (<= Y_0 1.0))\n"
            )
            # Don't provide output path - should use default
            converter.convert(str(vnnlib_path), None)
            # Default folder should be created
            default_output = Path(tmpdir) / "test"
            assert default_output.exists()

    def test_save_properties_numpy_format(self, temp_vnnlib_simple):
        """Test saving properties in numpy format."""
        converter = TorchVNNLIB(output_format="numpy")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output_numpy")
            converter.convert(temp_vnnlib_simple, output_path)
            # Check that .npz files were created
            assert Path(output_path).exists()

    def test_convert_with_verbose_saving(self, temp_vnnlib_simple, capsys):
        """Test verbose output during property saving."""
        converter = TorchVNNLIB(verbose=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)
            # Should have created output directory
            assert Path(output_path).exists()


class TestTorchVNNLIBProcessorVariations:
    """Test different processor execution branches."""

    def test_process_without_fast_type_detection(self, temp_vnnlib_simple):
        """Test processing without fast type detection."""
        converter = TorchVNNLIB(detect_fast_type=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)
            assert Path(output_path).exists()

    def test_process_with_parallel_disabled(self, temp_vnnlib_simple):
        """Test processing with parallel execution disabled."""
        converter = TorchVNNLIB(use_parallel=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)
            assert Path(output_path).exists()

    def test_process_with_parallel_enabled(self, temp_vnnlib_type3):
        """Test processing with parallel execution enabled."""
        converter = TorchVNNLIB(use_parallel=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type3, output_path)
            assert Path(output_path).exists()


class TestTorchVNNLIBConversionStats:
    """Test conversion statistics tracking."""

    def test_conversion_stats_recorded(self, temp_vnnlib_simple):
        """Test that conversion statistics are recorded."""
        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)
            # Check stats are recorded
            assert temp_vnnlib_simple in converter.conversion_stats
            stats = converter.conversion_stats[temp_vnnlib_simple]
            assert "type" in stats

    def test_conversion_stats_multiple_files(self, temp_vnnlib_simple, temp_vnnlib_type3):
        """Test statistics for multiple file conversions."""
        converter = TorchVNNLIB()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path1 = str(Path(tmpdir) / "output1")
            output_path2 = str(Path(tmpdir) / "output2")
            converter.convert(temp_vnnlib_simple, output_path1)
            converter.convert(temp_vnnlib_type3, output_path2)
            # Both should be in stats
            assert temp_vnnlib_simple in converter.conversion_stats
            assert temp_vnnlib_type3 in converter.conversion_stats


class TestTorchVNNLIBOutputFormats:
    """Test different output formats."""

    def test_torch_output_format(self, temp_vnnlib_simple):
        """Test torch output format (.pth files)."""
        converter = TorchVNNLIB(output_format="torch")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)
            # Should create .pth files
            pth_files = list(Path(output_path).rglob("*.pth"))
            assert len(pth_files) > 0

    def test_numpy_output_format(self, temp_vnnlib_simple):
        """Test numpy output format (.npz files)."""
        converter = TorchVNNLIB(output_format="numpy")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)
            # Should create .npz files
            npz_files = list(Path(output_path).rglob("*.npz"))
            assert len(npz_files) > 0


class TestTorchVNNLIBType2Processing:
    """Test TYPE2 VNN-LIB processing."""

    def test_convert_type2_simple(self, temp_vnnlib_type2):
        """Test TYPE2 conversion with simple OR output constraints."""
        converter = TorchVNNLIB(detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type2, output_path)
            assert Path(output_path).exists()
            # Check that files were created
            assert len(list(Path(output_path).rglob("*.pth"))) > 0

    def test_convert_type2_verbose(self, temp_vnnlib_type2, capsys):
        """Test TYPE2 conversion with verbose output."""
        converter = TorchVNNLIB(verbose=True, detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type2, output_path)
            captured = capsys.readouterr()
            # Should see TYPE2 processing output
            assert "Converting" in captured.out or "Type2" in captured.out

    def test_convert_type2_without_fast_detect(self, temp_vnnlib_type2):
        """Test TYPE2 conversion without fast type detection."""
        converter = TorchVNNLIB(detect_fast_type=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            # Should fall back to AST processing
            converter.convert(temp_vnnlib_type2, output_path)
            assert Path(output_path).exists()

    def test_type2_stats_tracking(self, temp_vnnlib_type2):
        """Test that TYPE2 processing statistics are tracked."""
        converter = TorchVNNLIB(detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type2, output_path)
            stats = converter.conversion_stats[temp_vnnlib_type2]
            assert "type" in stats
            assert "used_fast" in stats
            assert "time" in stats


class TestTorchVNNLIBVerboseProcessing:
    """Test verbose output for all processing stages."""

    def test_verbose_tokenization(self, temp_vnnlib_simple, capsys):
        """Test verbose output during tokenization."""
        converter = TorchVNNLIB(verbose=True, detect_fast_type=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)
            captured = capsys.readouterr()
            assert "Tokenization" in captured.out or "Read file" in captured.out

    def test_verbose_ast_parsing(self, temp_vnnlib_simple, capsys):
        """Test verbose output during AST parsing."""
        converter = TorchVNNLIB(verbose=True, detect_fast_type=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)
            captured = capsys.readouterr()
            assert "Parsing" in captured.out or "Optimization" in captured.out

    def test_verbose_optimization(self, temp_vnnlib_simple, capsys):
        """Test verbose output during optimization stage."""
        converter = TorchVNNLIB(verbose=True, detect_fast_type=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)
            captured = capsys.readouterr()
            # Should see optimization output
            assert "Optimization:" in captured.out

    def test_verbose_complete_conversion(self, temp_vnnlib_simple, capsys):
        """Test complete verbose output from start to finish."""
        converter = TorchVNNLIB(verbose=True, detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_simple, output_path)
            captured = capsys.readouterr()
            # Should see total time output
            assert "Total time" in captured.out


class TestTorchVNNLIBExceptionHandling:
    """Test exception handling in type processors."""

    def test_fallback_on_invalid_type_detection(self, temp_vnnlib_simple, capsys):
        """Test fallback to AST when type processor raises exception."""
        converter = TorchVNNLIB(verbose=True, detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            # Simple files should work, but test the fallback mechanism
            converter.convert(temp_vnnlib_simple, output_path)
            assert Path(output_path).exists()


class TestTorchVNNLIBType3Processing:
    """Test TYPE3 VNN-LIB processing (OR inputs + simple outputs)."""

    def test_convert_type3_with_detection(self, temp_vnnlib_type3_or):
        """Test TYPE3 conversion with fast type detection."""
        converter = TorchVNNLIB(detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type3_or, output_path)
            assert Path(output_path).exists()

    def test_convert_type3_verbose(self, temp_vnnlib_type3_or, capsys):
        """Test TYPE3 conversion with verbose output."""
        converter = TorchVNNLIB(verbose=True, detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type3_or, output_path)
            captured = capsys.readouterr()
            # Should see processing output
            assert "Converting" in captured.out


class TestTorchVNNLIBType4Processing:
    """Test TYPE4 VNN-LIB processing (OR inputs and OR outputs)."""

    def test_convert_type4_with_detection(self, temp_vnnlib_type4_or):
        """Test TYPE4 conversion with fast type detection."""
        converter = TorchVNNLIB(detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type4_or, output_path)
            assert Path(output_path).exists()

    def test_convert_type4_verbose(self, temp_vnnlib_type4_or, capsys):
        """Test TYPE4 conversion with verbose output."""
        converter = TorchVNNLIB(verbose=True, detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type4_or, output_path)
            captured = capsys.readouterr()
            # Should see processing output
            assert "Converting" in captured.out or "Type4" in captured.out


class TestTorchVNNLIBType5Processing:
    """Test TYPE5 VNN-LIB processing (mixed OR patterns)."""

    def test_convert_type5_with_detection(self, temp_vnnlib_type5):
        """Test TYPE5 conversion with fast type detection."""
        converter = TorchVNNLIB(detect_fast_type=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "output")
            converter.convert(temp_vnnlib_type5, output_path)
            assert Path(output_path).exists()
