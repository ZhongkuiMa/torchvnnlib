# === INFERRED IMPORT CONTRACTS (review before committing) ===
#
# Layers (highest rank -> lowest):
#   torchvnnlib (root) -> ast, fast_type (subpackages)
#
# ALLOWED (not tested):
#   torchvnnlib -> ast, torchvnnlib -> fast_type
#
# FORBIDDEN (inferred -- no lower->upper violations detected in current code):
#   ast -x-> fast_type  (ast must not import fast_type)
#   fast_type -x-> ast  (fast_type must not import ast -- verify actuals)
#
# [REVIEW] Approve these contracts before treating ARC3 as resolved.
# Run /python-optimize-tests apply @torchvnnlib/ to fill full ARC3 boundaries.
# ================================================================
"""Import architecture tests for torchvnnlib."""

__docformat__ = "restructuredtext"

import ast
import importlib
from pathlib import Path

import pytest

import torchvnnlib

_SRC = Path(__file__).parent.parent.parent / "src" / "torchvnnlib"


def _get_imports(path: Path) -> set[str]:
    """Return top-level module names imported by path."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


class TestImportSmoke:
    """Package imports without circular dependencies or broken __init__ chains."""

    def test_top_level_import(self):
        """Import package top-level without error."""
        assert isinstance(torchvnnlib.TorchVNNLIB, type)

    def test_submodule_imports_cleanly(self):
        """Every submodule imports without ImportError."""
        errors: list[str] = []
        for f in _SRC.rglob("*.py"):
            rel = f.relative_to(_SRC.parent).with_suffix("")
            mod = ".".join(rel.parts)
            try:
                importlib.import_module(mod)
            except ImportError as e:
                errors.append(f"{mod}: {e}")
        assert not errors, "Import errors:\n" + "\n".join(errors)


class TestLayerBoundaries:
    """Inferred layer boundary tests -- see contract header above.

    [REVIEW] These are stubs. Run /python-optimize-tests apply to generate
    full ARC3 boundary tests from the actual dependency graph.
    """

    # [REVIEW] STR11 merged: test_ast_does_not_import_fast_type +
    # test_fast_type_does_not_import_ast into test_subpackage_does_not_import_sibling.
    @pytest.mark.parametrize(
        ("scan_subpkg", "forbidden_subpkg"),
        [
            pytest.param("ast", "fast_type", id="ast_no_fast_type"),
            pytest.param("fast_type", "ast", id="fast_type_no_ast"),
        ],
    )
    def test_subpackage_does_not_import_sibling(self, scan_subpkg, forbidden_subpkg):
        """Subpackage must not import from its sibling subpackage."""
        violations = [
            str(f.relative_to(_SRC))
            for f in (_SRC / scan_subpkg).rglob("*.py")
            if forbidden_subpkg in _get_imports(f)
        ]
        assert not violations, f"{scan_subpkg} imports {forbidden_subpkg} in: {violations}"
