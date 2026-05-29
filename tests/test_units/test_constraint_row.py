"""Unit tests for ``_constraint_row.py`` row-emission primitives.

Covers the shared helpers used by both AST and fast-type pipelines:
``apply_input_bound``, ``write_value_bound_row``,
``write_value_bound_rows_eq``, ``write_compare_row``, and
``normalize_neg_zero``.
"""

import numpy as np
import pytest

from torchvnnlib._constraint_row import (
    apply_input_bound,
    normalize_neg_zero,
    write_compare_row,
    write_value_bound_row,
    write_value_bound_rows_eq,
)


class TestApplyInputBound:
    """Test in-place input-bound updates."""

    def test_leq_sets_upper(self):
        bounds = np.full((3, 2), float("nan"))
        apply_input_bound(bounds, idx=1, op="<=", value=2.5)
        assert bounds[1, 1] == 2.5
        assert np.isnan(bounds[1, 0])

    def test_geq_sets_lower(self):
        bounds = np.full((3, 2), float("nan"))
        apply_input_bound(bounds, idx=2, op=">=", value=-1.0)
        assert bounds[2, 0] == -1.0
        assert np.isnan(bounds[2, 1])

    def test_eq_sets_both(self):
        bounds = np.full((3, 2), float("nan"))
        apply_input_bound(bounds, idx=0, op="=", value=0.5)
        assert bounds[0, 0] == 0.5
        assert bounds[0, 1] == 0.5


class TestWriteValueBoundRow:
    """Test single-bound row emission."""

    def test_leq_layout(self):
        row = np.zeros((4,))
        write_value_bound_row(row, idx=1, op="<=", value=3.0)
        assert row[0] == 3.0
        assert row[2] == -1.0
        assert row[1] == 0.0
        assert row[3] == 0.0

    def test_geq_layout(self):
        row = np.zeros((4,))
        write_value_bound_row(row, idx=2, op=">=", value=1.5)
        assert row[0] == -1.5
        assert row[3] == 1.0

    def test_geq_zero_value_no_neg_zero(self):
        row = np.zeros((4,))
        write_value_bound_row(row, idx=0, op=">=", value=0.0)
        assert row[0] == 0.0
        assert repr(float(row[0])) == "0.0"

    def test_rejects_2d_row(self):
        """``row`` must be 1-D; passing a (1,n) slice would broadcast silently."""
        bad = np.zeros((1, 4))
        with pytest.raises(ValueError, match="1-D row view"):
            write_value_bound_row(bad, idx=0, op="<=", value=1.0)


class TestWriteValueBoundRowsEq:
    """Test the two-row ``=`` decomposition."""

    def test_eq_layout(self):
        rg = np.zeros((4,))
        rl = np.zeros((4,))
        write_value_bound_rows_eq(rg, rl, idx=1, value=2.0)
        # >= half
        assert rg[0] == -2.0
        assert rg[2] == 1.0
        # <= half
        assert rl[0] == 2.0
        assert rl[2] == -1.0

    def test_eq_zero_value_normalizes(self):
        rg = np.zeros((4,))
        rl = np.zeros((4,))
        write_value_bound_rows_eq(rg, rl, idx=0, value=0.0)
        assert repr(float(rg[0])) == "0.0"

    def test_rejects_2d(self):
        bad = np.zeros((1, 4))
        good = np.zeros((4,))
        with pytest.raises(ValueError, match="1-D row view"):
            write_value_bound_rows_eq(bad, good, idx=0, value=1.0)


class TestWriteCompareRow:
    """Test ``Y_i <op> Y_j`` row emission."""

    def test_leq(self):
        row = np.zeros((5,))
        write_compare_row(row, idx1=0, idx2=2, op="<=")
        assert row[1] == -1.0
        assert row[3] == 1.0
        assert row[0] == 0.0  # bias untouched

    def test_geq(self):
        row = np.zeros((5,))
        write_compare_row(row, idx1=0, idx2=2, op=">=")
        assert row[1] == 1.0
        assert row[3] == -1.0

    def test_rejects_2d(self):
        bad = np.zeros((1, 5))
        with pytest.raises(ValueError, match="1-D row view"):
            write_compare_row(bad, idx1=0, idx2=1, op="<=")


class TestNormalizeNegZero:
    """Test ``-0.0 -> +0.0`` normalisation."""

    def test_replaces_neg_zero(self):
        arr = np.array([-0.0, -1.0, 0.0, 2.0])
        normalize_neg_zero(arr)
        assert repr(float(arr[0])) == "0.0"
        assert arr[1] == -1.0
        assert arr[2] == 0.0
        assert arr[3] == 2.0

    def test_returns_same_object(self):
        arr = np.array([-0.0, 0.0])
        result = normalize_neg_zero(arr)
        assert result is arr


class TestNegativeIndexGuard:
    """Regression: row writers reject negative idx to avoid bias overwrite."""

    def test_apply_input_bound_rejects_negative(self):
        bounds = np.full((3, 2), float("nan"))
        with pytest.raises(ValueError, match="non-negative"):
            apply_input_bound(bounds, idx=-1, op="<=", value=1.0)

    def test_write_value_bound_row_rejects_negative(self):
        row = np.zeros((4,))
        with pytest.raises(ValueError, match="non-negative"):
            write_value_bound_row(row, idx=-1, op="<=", value=2.0)

    def test_write_value_bound_rows_eq_rejects_negative(self):
        rg = np.zeros((4,))
        rl = np.zeros((4,))
        with pytest.raises(ValueError, match="non-negative"):
            write_value_bound_rows_eq(rg, rl, idx=-1, value=0.5)

    def test_write_compare_row_rejects_negative(self):
        row = np.zeros((5,))
        with pytest.raises(ValueError, match="non-negative"):
            write_compare_row(row, idx1=-1, idx2=2, op="<=")
        with pytest.raises(ValueError, match="non-negative"):
            write_compare_row(row, idx1=0, idx2=-2, op="<=")
