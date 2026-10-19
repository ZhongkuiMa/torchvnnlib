"""Process-isolated VNN-LIB 2.0 parsing and cache management."""

from __future__ import annotations

__docformat__ = "restructuredtext"
__all__ = ["parse_query_isolated"]

import hashlib
import multiprocessing
import tempfile
import traceback
from importlib.metadata import version as distribution_version
from multiprocessing.connection import Connection
from pathlib import Path

from torchvnnlib._query import VNNLibQuery
from torchvnnlib._query_codec import (
    _SCHEMA_VERSION,
    VNNLibQueryCodecError,
    dump_query,
    load_query,
)
from torchvnnlib._v2 import parse_query
from torchvnnlib._v2_stream import STREAMING_ABI


def _source_digest(path: Path) -> str:
    """Hash the exact source bytes used to identify a cache entry."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cache_entry_name(source_sha256: str) -> str:
    """Include every compatibility authority in the cache filename."""
    compatibility = f"{_SCHEMA_VERSION}\0{distribution_version('vnnlib')}\0{STREAMING_ABI}".encode()
    suffix = hashlib.sha256(compatibility).hexdigest()[:16]
    return f"{source_sha256}-{suffix}.json"


def _worker(
    source: str,
    output: str,
    source_sha256: str,
    error_connection: Connection,
) -> None:
    """Parse and serialize inside a disposable child process."""
    try:
        dump_query(parse_query(source), output, source_sha256=source_sha256)
    except BaseException:
        error_connection.send(traceback.format_exc())
        raise
    finally:
        error_connection.close()


def parse_query_isolated(
    path: str | Path,
    *,
    cache_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> VNNLibQuery:
    """Parse a file in a spawned process and optionally maintain a safe cache.

    ``cache_dir`` derives the entry name from the source digest computed here,
    avoiding a second full source read in callers.  It is mutually exclusive
    with an explicit ``cache_path``.
    """
    if cache_path is not None and cache_dir is not None:
        raise ValueError("cache_path and cache_dir are mutually exclusive")
    source = Path(path)
    source_sha256 = _source_digest(source)
    cache = (
        Path(cache_path)
        if cache_path is not None
        else Path(cache_dir) / _cache_entry_name(source_sha256)
        if cache_dir is not None
        else None
    )
    if cache is not None and cache.exists():
        try:
            return load_query(cache, expected_source_sha256=source_sha256)
        except (OSError, VNNLibQueryCodecError):
            pass

    temporary_parent = cache.parent if cache is not None else None
    if temporary_parent is not None:
        temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_parent) as directory:
        output = Path(directory) / "query.json"
        context = multiprocessing.get_context("spawn")
        error_receiver, error_sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker,
            args=(str(source), str(output), source_sha256, error_sender),
        )
        process.start()
        error_sender.close()
        process.join()
        if process.exitcode != 0:
            detail = (
                error_receiver.recv()
                if error_receiver.poll()
                else f"worker exited with code {process.exitcode}"
            )
            error_receiver.close()
            raise RuntimeError(f"Isolated VNN-LIB parsing failed:\n{detail}")
        error_receiver.close()
        query = load_query(output, expected_source_sha256=source_sha256)
        if cache is not None:
            output.replace(cache)
        return query
