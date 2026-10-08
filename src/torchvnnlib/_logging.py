"""Logging configuration for torchvnnlib."""

__docformat__ = "restructuredtext"
__all__ = ["_enable_verbose"]

import logging
import sys
import threading

_VERBOSE_LOCK = threading.Lock()
_VERBOSE_ENABLED = False


class _DynamicStderrHandler(logging.StreamHandler):
    """Resolve ``sys.stderr`` at emission time instead of construction time."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def _enable_verbose() -> None:
    """Attach console handler to package logger and set DEBUG level.

    Idempotent and thread-safe: concurrent callers attach at most one
    ``StreamHandler``. Safe to call from worker threads.
    """
    global _VERBOSE_ENABLED
    if _VERBOSE_ENABLED:
        return
    with _VERBOSE_LOCK:
        if _VERBOSE_ENABLED:
            return
        pkg_logger = logging.getLogger("torchvnnlib")
        if not any(isinstance(h, logging.StreamHandler) for h in pkg_logger.handlers):
            handler = _DynamicStderrHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            pkg_logger.addHandler(handler)
        pkg_logger.setLevel(logging.DEBUG)
        _VERBOSE_ENABLED = True
