"""Tests for the VNNLIBType enum in fast_type._enums.

Verifies enum member values, repr, uniqueness, and membership properties.
"""

import pytest

from torchvnnlib.fast_type._enums import VNNLIBType


class TestVNNLIBTypeValues:
    """Test VNNLIBType enum member values."""

    @pytest.mark.parametrize(
        ("enum_member", "expected_value"),
        [
            (VNNLIBType.TYPE1, 1),
            (VNNLIBType.TYPE2, 2),
            (VNNLIBType.TYPE3, 3),
            (VNNLIBType.TYPE4, 4),
            (VNNLIBType.TYPE5, 5),
            (VNNLIBType.COMPLEX, 6),
        ],
    )
    def test_enum_value_matches_int(self, enum_member, expected_value):
        """Each type enum member has the expected integer value (STR5: merged 5 value tests, STR11: 1 pair)."""
        assert enum_member == expected_value


class TestVNNLIBTypeRepr:
    """Test VNNLIBType string representation."""

    @pytest.mark.parametrize(
        ("enum_member", "expected_repr"),
        [
            (VNNLIBType.TYPE1, "TYPE1"),
            (VNNLIBType.TYPE2, "TYPE2"),
            (VNNLIBType.COMPLEX, "COMPLEX"),
        ],
    )
    def test_repr_returns_type_name(self, enum_member, expected_repr):
        """Repr of each type returns its name (STR5: merged 3 repr tests)."""
        assert repr(enum_member) == expected_repr


class TestVNNLIBTypeMembership:
    """Test VNNLIBType can be used as a regular IntEnum."""

    def test_isinstance_int(self):
        """VNNLIBType members are instances of int."""
        assert isinstance(VNNLIBType.TYPE1, int)

    # [REVIEW] Deleted: test_comparison_with_int. STR11: merged into parametrized test_enum_value_matches_int.

    def test_all_members_unique(self):
        """All VNNLIBType members have distinct values (enforced by @unique)."""
        values = [m.value for m in VNNLIBType]
        assert len(values) == len(set(values))

    def test_ordering_types_first(self):
        """TYPE1 through TYPE5 precede COMPLEX in declaration order."""
        members = list(VNNLIBType)
        assert members[:5] == [
            VNNLIBType.TYPE1,
            VNNLIBType.TYPE2,
            VNNLIBType.TYPE3,
            VNNLIBType.TYPE4,
            VNNLIBType.TYPE5,
        ]
        assert members[5] == VNNLIBType.COMPLEX
