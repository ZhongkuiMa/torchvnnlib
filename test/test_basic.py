"""Basic functionality test for TorchVNNLib.

Tests that the conversion works correctly on a synthetic vnnlib property.
"""

import os
import sys
import tempfile
import shutil

# Add parent directory to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
from torchvnnlib import TorchVNNLIB


def create_simple_vnnlib():
    """Create a simple vnnlib property for testing.

    Creates a vnnlib file with:
    - 3 input variables (X_0, X_1, X_2)
    - 2 output variables (Y_0, Y_1)
    - 2 OR groups with different input constraints
    - Output constraint: Y_1 <= Y_0

    :return: Path to temporary vnnlib file
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

    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    vnnlib_path = os.path.join(temp_dir, "test_property.vnnlib")
    with open(vnnlib_path, "w") as f:
        f.write(vnnlib_content)

    return vnnlib_path, temp_dir


def test_basic_conversion():
    """Test basic vnnlib conversion.

    :return: True if test passes
    :raises AssertionError: If conversion produces unexpected results
    """
    print("Creating test vnnlib property...")
    vnnlib_path, temp_dir = create_simple_vnnlib()

    try:
        print(f"Created: {vnnlib_path}")

        print("Running conversion...")
        converter = TorchVNNLIB()
        output_dir = os.path.join(temp_dir, "output")
        converter.convert(vnnlib_path, target_folder_path=output_dir)

        print(f"Conversion complete. Output: {output_dir}")

        # Verify output structure
        print("\n=== Verifying Output Structure ===")

        # Should have 1 OR group (from output assertion)
        or_group_0 = os.path.join(output_dir, "or_group_0")

        assert os.path.exists(or_group_0), f"Missing or_group_0 directory"
        print("✓ or_group_0 exists")

        # Should have 2 sub-properties (from 2 input OR clauses)
        sub_prop_0_0 = os.path.join(or_group_0, "sub_prop_0.pth")
        sub_prop_0_1 = os.path.join(or_group_0, "sub_prop_1.pth")

        assert os.path.exists(sub_prop_0_0), f"Missing sub_prop_0.pth in or_group_0"
        assert os.path.exists(sub_prop_0_1), f"Missing sub_prop_1.pth in or_group_0"
        print("✓ Both sub-properties exist")

        # Load and verify tensor data
        print("\n=== Verifying Tensor Data ===")

        # Check or_group_0/sub_prop_0.pth
        data_0 = torch.load(sub_prop_0_0, weights_only=True)
        assert "input" in data_0, "Missing 'input' key in sub_prop_0"
        assert "output" in data_0, "Missing 'output' key in sub_prop_0"
        print(f"✓ or_group_0/sub_prop_0.pth has correct keys")

        # Verify input shape: [3, 2] (3 inputs, 2 bounds per input)
        input_tensor_0 = data_0["input"]
        assert input_tensor_0.shape == torch.Size([3, 2]), \
            f"Input shape mismatch: expected [3, 2], got {input_tensor_0.shape}"
        print(f"✓ Input tensor shape: {input_tensor_0.shape}")

        # Verify input bounds for sub_prop_0 (first input OR clause)
        # Expected: X_0 in [0, 1], X_1 in [-1, 1], X_2 in [0.5, 1.5]
        expected_bounds_0 = torch.tensor([
            [0.0, 1.0],    # X_0
            [-1.0, 1.0],   # X_1
            [0.5, 1.5]     # X_2
        ], dtype=input_tensor_0.dtype)
        assert torch.allclose(input_tensor_0, expected_bounds_0, rtol=1e-5), \
            f"Input bounds mismatch for sub_prop_0:\nExpected:\n{expected_bounds_0}\nGot:\n{input_tensor_0}"
        print(f"✓ Input bounds correct for sub_prop_0")

        # Verify output constraints
        output_constraints_0 = data_0["output"]
        assert isinstance(output_constraints_0, list), "Output should be a list"
        assert len(output_constraints_0) > 0, "Output constraints list should not be empty"
        print(f"✓ Output has {len(output_constraints_0)} constraint(s)")

        # Check or_group_0/sub_prop_1.pth
        data_1 = torch.load(sub_prop_0_1, weights_only=True)
        input_tensor_1 = data_1["input"]

        # Verify input bounds for sub_prop_1 (second input OR clause)
        # Expected: X_0 in [-1, 0], X_1 in [0, 2], X_2 in [-0.5, 0.5]
        expected_bounds_1 = torch.tensor([
            [-1.0, 0.0],   # X_0
            [0.0, 2.0],    # X_1
            [-0.5, 0.5]    # X_2
        ], dtype=input_tensor_1.dtype)
        assert torch.allclose(input_tensor_1, expected_bounds_1, rtol=1e-5), \
            f"Input bounds mismatch for sub_prop_1:\nExpected:\n{expected_bounds_1}\nGot:\n{input_tensor_1}"
        print(f"✓ Input bounds correct for sub_prop_1")

        print("\nAll assertions passed!")
        print("TorchVNNLib conversion is working correctly!")
        return True

    finally:
        # Clean up temporary files
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print(f"✓ Cleaned up {temp_dir}")


if __name__ == "__main__":
    try:
        test_basic_conversion()
        print("\n" + "=" * 50)
        print("SUCCESS: Basic functionality test passed")
        print("=" * 50)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
