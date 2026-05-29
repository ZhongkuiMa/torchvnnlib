"""Logging configuration for torchvnnlib."""

__docformat__ = "restructuredtext"
__all__ = ["_enable_verbose"]

import logging
import threading

_VERBOSE_LOCK = threading.Lock()
_VERBOSE_ENABLED = False


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
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            pkg_logger.addHandler(handler)
        pkg_logger.setLevel(logging.DEBUG)
        _VERBOSE_ENABLED = True
