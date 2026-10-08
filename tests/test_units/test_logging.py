"""Tests for package logging configuration."""

import io
import logging
from contextlib import redirect_stderr

from torchvnnlib._logging import _enable_verbose


def test_verbose_handler_follows_current_stderr():
    """Verbose logs follow stderr redirection after initial configuration."""
    logger = logging.getLogger("torchvnnlib.test.dynamic_stderr")
    first = io.StringIO()
    second = io.StringIO()

    _enable_verbose()
    with redirect_stderr(first):
        logger.info("first destination")
    with redirect_stderr(second):
        logger.info("second destination")

    assert first.getvalue() == "first destination\n"
    assert second.getvalue() == "second destination\n"
