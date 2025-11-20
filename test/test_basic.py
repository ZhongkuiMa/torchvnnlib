"""Basic functionality test for TorchVNNLib.

Tests that the conversion works correctly on a synthetic vnnlib property.
Supports testing both PyTorch and NumPy backends.
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from torchvnnlib import TorchVNNLIB


def create_simple_vnnlib():
    """Create a simple vnnlib property for testing.

    Creates a vnnlib file with:
    - 3 input variables (X_0, X_1, X_2)
    - 2 output variables (Y_0, Y_1)
    - 2 OR groups with different input constraints
    - Output constraint: Y_1 <= Y_0

    :return: Path to temporary vnnlib file and temp directory
    """
    vnnlib_content = """; Simple test property

(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)

(declare-const Y_0 Real)
(declare-const Y_1 Real)

; Input constraints: Two OR groups
(assert (or
    ; OR group 0: X_0 in [0, 1], X_1 in [-1, 1], X_2 in [0.5, 1.5]
    (and (<= X_0 1) (>= X_0 0) (<= X_1 1) (>= X_1 -1) (<= X_2 1.5) (>= X_2 0.5))
    ; OR group 1: X_0 in [-1, 0], X_1 in [0, 2], X_2 in [-0.5, 0.5]
    (and (<= X_0 0) (>= X_0 -1) (<= X_1 2) (>= X_1 0) (<= X_2 0.5) (>= X_2 -0.5))
))

; Output constraint: Y_1 <= Y_0 (equivalent to Y_0 - Y_1 >= 0)
(assert (or
    (and (<= Y_1 Y_0))
))
"""

    temp_dir = tempfile.mkdtemp()
    vnnlib_path = os.path.join(temp_dir, "test_property.vnnlib")
    with open(vnnlib_path, "w") as f:
        f.write(vnnlib_content)

    return vnnlib_path, temp_dir


def load_data(file_path: str, backend: str):
    """Load data from either .pth or .npz file.

    :param file_path: Path to data file
    :param backend: Backend type ('torch' or 'numpy')
    :return: Loaded data dictionary
    """
    if backend == "torch":
        return torch.load(file_path, weights_only=True)
    else:
        npz_data = np.load(file_path, allow_pickle=True)
        return {
            "input": npz_data["input"],
            "output": npz_data["output"],
        }


def arrays_close(arr1, arr2, rtol=1e-5):
    """Check if two arrays are close (works for both torch and numpy).

    :param arr1: First array
    :param arr2: Second array
    :param rtol: Relative tolerance
    :return: True if arrays are close
    """
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        return np.allclose(arr1, arr2, rtol=rtol)
    elif TORCH_AVAILABLE and isinstance(arr1, torch.Tensor):
        if isinstance(arr2, np.ndarray):
            arr2 = torch.from_numpy(arr2)
        return torch.allclose(arr1, arr2, rtol=rtol)
    else:
        return np.allclose(np.array(arr1), np.array(arr2), rtol=rtol)


def test_basic_conversion(backend="torch"):
    """Test basic vnnlib conversion.

    :param backend: Backend to use ('torch' or 'numpy')
    :return: True if test passes
    """
    if backend == "torch" and not TORCH_AVAILABLE:
        print("WARNING: PyTorch not available, skipping torch backend test")
        return True

    print(f"\n{'='*60}")
    print(f"Testing with backend: {backend.upper()}")
    print(f"{'='*60}")

    print("Creating test vnnlib property...")
    vnnlib_path, temp_dir = create_simple_vnnlib()

    try:
        print(f"Created: {vnnlib_path}")

        print(f"Running conversion with {backend} backend...")
        converter = TorchVNNLIB(output_format=backend)
        output_dir = os.path.join(temp_dir, f"output_{backend}")
        converter.convert(vnnlib_path, target_folder_path=output_dir)

        print(f"Conversion complete. Output: {output_dir}")

        # Determine file extension
        file_ext = ".pth" if backend == "torch" else ".npz"

        print(f"\n=== Verifying Output Structure ({backend}) ===")

        or_group_0 = os.path.join(output_dir, "or_group_0")

        assert os.path.exists(or_group_0), f"Missing or_group_0 directory"
        print(f"OK: or_group_0 exists")

        sub_prop_0_0 = os.path.join(or_group_0, f"sub_prop_0{file_ext}")
        sub_prop_0_1 = os.path.join(or_group_0, f"sub_prop_1{file_ext}")

        assert os.path.exists(
            sub_prop_0_0
        ), f"Missing sub_prop_0{file_ext} in or_group_0"
        assert os.path.exists(
            sub_prop_0_1
        ), f"Missing sub_prop_1{file_ext} in or_group_0"
        print(f"OK: Both sub-properties exist with {file_ext} extension")

        print(f"\n=== Verifying Data ({backend}) ===")

        data_0 = load_data(sub_prop_0_0, backend)
        assert "input" in data_0, "Missing 'input' key in sub_prop_0"
        assert "output" in data_0, "Missing 'output' key in sub_prop_0"
        print(f"OK: or_group_0/sub_prop_0{file_ext} has correct keys")

        input_tensor_0 = data_0["input"]
        expected_shape = (3, 2)
        actual_shape = (
            tuple(input_tensor_0.shape)
            if hasattr(input_tensor_0, "shape")
            else input_tensor_0.shape
        )
        assert (
            actual_shape == expected_shape
        ), f"Input shape mismatch: expected {expected_shape}, got {actual_shape}"
        print(f"OK: Input array shape: {actual_shape}")

        expected_bounds_0 = np.array(
            [[0.0, 1.0], [-1.0, 1.0], [0.5, 1.5]], dtype=np.float64
        )
        assert arrays_close(
            input_tensor_0, expected_bounds_0
        ), f"Input bounds mismatch for sub_prop_0"
        print(f"OK: Input bounds correct for sub_prop_0")

        output_constraints_0 = data_0["output"]
        if backend == "numpy":
            output_constraints_0 = output_constraints_0.tolist()
        assert isinstance(
            output_constraints_0, (list, np.ndarray)
        ), "Output should be a list or array"
        assert len(output_constraints_0) > 0, "Output constraints should not be empty"
        print(f"OK: Output has {len(output_constraints_0)} constraint(s)")

        data_1 = load_data(sub_prop_0_1, backend)
        input_tensor_1 = data_1["input"]

        expected_bounds_1 = np.array(
            [[-1.0, 0.0], [0.0, 2.0], [-0.5, 0.5]], dtype=np.float64
        )
        assert arrays_close(
            input_tensor_1, expected_bounds_1
        ), f"Input bounds mismatch for sub_prop_1"
        print(f"OK: Input bounds correct for sub_prop_1")

        print(f"\nAll assertions passed for {backend} backend!")
        return True

    finally:
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print(f"OK: Cleaned up {temp_dir}")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test TorchVNNLib with different backends"
    )
    parser.add_argument(
        "--backend",
        choices=["torch", "numpy", "both"],
        default="both",
        help="Backend to test (torch, numpy, or both)",
    )
    args = parser.parse_args()

    try:
        if args.backend == "both":
            print("Testing both PyTorch and NumPy backends")
            success_torch = test_basic_conversion("torch")
            success_numpy = test_basic_conversion("numpy")
            success = success_torch and success_numpy
        else:
            success = test_basic_conversion(args.backend)

        if success:
            print("\n" + "=" * 60)
            print("SUCCESS: All tests passed")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("FAILURE: Some tests failed")
            print("=" * 60)
            sys.exit(1)

    except Exception as e:
        print(f"\nERROR: Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
