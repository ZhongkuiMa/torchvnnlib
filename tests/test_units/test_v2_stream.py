"""Differential tests for the bounded VNN-LIB 2.0 streaming grammar."""

import gzip
from dataclasses import replace

import pytest

import torchvnnlib._v2 as v2_module
from torchvnnlib import TorchVNNLIB, VNNLibComparator, VNNLibParseBackend
from torchvnnlib._v2_stream import try_parse_canonical_query

CANONICAL_QUERY = """; canonical NN4Sys-style query
(vnnlib-version <2.0>)
(declare-network f
  (declare-input X float32 [2])
  (declare-output Y float32 [1]))
(assert (or
  (and (<= 0.0 X[0]) (<= X[0] 1.0) (>= X[1] 0.0) (< Y[0] 0.5))
  (and (>= X[0] 1.0) (<= X[1] 1.0) (> -0.25 Y[0]))))
"""


def test_streamed_query_matches_official_owned_ir(tmp_path):
    source = tmp_path / "canonical.vnnlib"
    source.write_text(CANONICAL_QUERY, encoding="utf-8")

    streamed = TorchVNNLIB.parse_query(source)
    official = TorchVNNLIB.parse_query(lines=CANONICAL_QUERY.splitlines(keepends=True))

    assert streamed.parse_backend is VNNLibParseBackend.CANONICAL_STREAM
    assert official.parse_backend is VNNLibParseBackend.OFFICIAL_AST
    assert replace(streamed, parse_backend=VNNLibParseBackend.OFFICIAL_AST) == official
    assert streamed.branch_input_domains[0].bound_at(0).comparator is VNNLibComparator.GE


def test_streaming_path_does_not_call_official_parser(tmp_path, monkeypatch):
    source = tmp_path / "canonical.vnnlib"
    source.write_text(CANONICAL_QUERY, encoding="utf-8")

    def fail_official_parser(path):
        raise AssertionError(f"official parser unexpectedly called for {path}")

    monkeypatch.setattr(v2_module.vnnlib, "parse_query_file", fail_official_parser)

    assert TorchVNNLIB.parse_query(source).parse_backend is VNNLibParseBackend.CANONICAL_STREAM


def test_streaming_path_reads_gzip_without_materializing_source(tmp_path, monkeypatch):
    source = tmp_path / "canonical.vnnlib.gz"
    with gzip.open(source, mode="wt", encoding="utf-8") as stream:
        stream.write(CANONICAL_QUERY)

    monkeypatch.setattr(
        v2_module.vnnlib,
        "parse_query_string",
        lambda text: pytest.fail("official string parser unexpectedly called"),
    )

    assert TorchVNNLIB.parse_query(source).parse_backend is VNNLibParseBackend.CANONICAL_STREAM


def test_noncanonical_file_falls_back_to_official_parser(tmp_path, monkeypatch):
    source = tmp_path / "general.vnnlib"
    source.write_text(
        """(vnnlib-version <2.0>)
(declare-network f
  (declare-input X float32 [1])
  (declare-output Y float32 [1]))
(assert (>= X[0] 0.0))
(assert (<= Y[0] 1.0))
""",
        encoding="utf-8",
    )
    original = v2_module.vnnlib.parse_query_file
    calls = 0

    def counted_parser(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(v2_module.vnnlib, "parse_query_file", counted_parser)

    query = TorchVNNLIB.parse_query(source)

    assert calls == 1
    assert query.parse_backend is VNNLibParseBackend.OFFICIAL_AST


@pytest.mark.parametrize(
    "replacement",
    [
        ("X[0]", "UNKNOWN[0]"),
        ("X[0]", "X[2]"),
        ("0.0", "0"),
        ("0.0", "0.0e0"),
        ("(< Y[0] 0.5)", "(+ Y[0] 0.5)"),
        (")\n", ")\n(assert (<= Y[0] 1.0))\n"),
    ],
)
def test_streaming_recognizer_rejects_noncanonical_structure(tmp_path, replacement):
    source = tmp_path / "noncanonical.vnnlib"
    old, new = replacement
    source.write_text(CANONICAL_QUERY.replace(old, new, 1), encoding="utf-8")

    assert try_parse_canonical_query(source) is None
