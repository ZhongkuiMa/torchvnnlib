"""Shared fixtures for test_torchvnnlib tests.

Most setup in this folder is test-specific (different VNN-LIB content per test,
distinct temp_vnnlib_* fixtures in test_torchvnnlib_class.py). The parent
conftest at tests/test_units/conftest.py already provides:
- ``backend`` (parametrized over torch/numpy)
- ``backend_name``, ``sample_input_vars``, ``sample_output_vars``
- ``type1_*``, ``type2_*`` fixture families

Only forward-looking shared utilities are placed here. Existing test files
should not need to change to use them.
"""

__docformat__ = "restructuredtext"

import pytest


@pytest.fixture
def output_path(tmp_path):
    """Return an output path string under pytest tmp_path.

    New tests should prefer this over ``tempfile.TemporaryDirectory`` +
    ``Path(tmpdir) / "output"`` inline boilerplate.

    :return: Absolute output path string (directory does not exist yet).
    """
    return str(tmp_path / "output")
