import json

import pytest

import torchvnnlib._isolated as isolated_module
from torchvnnlib import (
    TorchVNNLIB,
    VNNLibParseBackend,
    VNNLibQueryCodecError,
    dump_query,
    load_query,
)

QUERY = """(vnnlib-version <2.0>)
(declare-network f
  (declare-input X float32 [1])
  (declare-output Y float32 [1]))
(assert (>= X[0] -0.2500))
(assert (< Y[0] 1.500))
"""

MIXED_QUERY = """(vnnlib-version <2.0>)
(declare-network f
  (declare-input X float32 [1])
  (declare-output Y float32 [1]))
(assert (or
  (and (>= X[0] -0.2500) (< Y[0] 1.500))
  (and (<= X[0] 0.7500) (> Y[0] 2.500))))
"""


def test_query_codec_round_trip_preserves_owned_ir(tmp_path):
    query = TorchVNNLIB.parse_query(lines=QUERY.splitlines(keepends=True))
    path = tmp_path / "query.json"

    dump_query(query, path, source_sha256="abc")

    assert load_query(path, expected_source_sha256="abc") == query
    assert load_query(path).parse_backend is VNNLibParseBackend.OFFICIAL_AST


def test_query_codec_round_trip_preserves_branch_input_domains(tmp_path):
    query = TorchVNNLIB.parse_query(lines=MIXED_QUERY.splitlines(keepends=True))
    path = tmp_path / "query.json"

    dump_query(query, path, source_sha256="abc")
    restored = load_query(path, expected_source_sha256="abc")

    assert restored == query
    assert restored.branch_input_domains[0].bound_at(0).lexeme == "-0.2500"
    assert restored.branch_input_domains[1].bound_at(0).lexeme == "0.7500"


def test_query_codec_rejects_source_and_payload_corruption(tmp_path):
    query = TorchVNNLIB.parse_query(lines=QUERY.splitlines(keepends=True))
    path = tmp_path / "query.json"
    dump_query(query, path, source_sha256="abc")

    with pytest.raises(VNNLibQueryCodecError, match="source digest"):
        load_query(path, expected_source_sha256="def")

    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["input_domain"]["lexemes"] = "YmFk"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(VNNLibQueryCodecError, match="checksum"):
        load_query(path, expected_source_sha256="abc")


def test_query_codec_rejects_streaming_abi_mismatch(tmp_path):
    query = TorchVNNLIB.parse_query(lines=QUERY.splitlines(keepends=True))
    path = tmp_path / "query.json"
    dump_query(query, path, source_sha256="abc")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["streaming_abi"] += 1
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(VNNLibQueryCodecError, match="streaming parser ABI"):
        load_query(path, expected_source_sha256="abc")


def test_isolated_parser_recovers_from_corrupt_cache(tmp_path):
    source = tmp_path / "property.vnnlib"
    cache = tmp_path / "cache" / "property.json"
    source.write_text(QUERY, encoding="utf-8")
    cache.parent.mkdir()
    cache.write_text("not-json", encoding="utf-8")

    query = TorchVNNLIB.parse_query_isolated(source, cache_path=cache)

    assert query.global_input_domain.bound_at(0).lexeme == "-0.2500"
    assert load_query(cache) == query


def test_isolated_parser_derives_cache_entry_from_source_digest(tmp_path):
    source = tmp_path / "property.vnnlib"
    cache_dir = tmp_path / "cache"
    source.write_text(QUERY, encoding="utf-8")

    query = TorchVNNLIB.parse_query_isolated(source, cache_dir=cache_dir)

    entries = tuple(cache_dir.glob("*.json"))
    assert len(entries) == 1
    assert load_query(entries[0]) == query


def test_isolated_parser_rejects_ambiguous_cache_location(tmp_path):
    source = tmp_path / "property.vnnlib"
    source.write_text(QUERY, encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        TorchVNNLIB.parse_query_isolated(
            source,
            cache_path=tmp_path / "query.json",
            cache_dir=tmp_path / "cache",
        )


def test_isolated_worker_reports_write_failures_without_error_file(monkeypatch):
    query = TorchVNNLIB.parse_query(lines=QUERY.splitlines(keepends=True))
    messages = []
    closed = False

    class Connection:
        def send(self, message):
            messages.append(message)

        def close(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(isolated_module, "parse_query", lambda source: query)

    def fail_dump(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(isolated_module, "dump_query", fail_dump)

    with pytest.raises(OSError, match="No space left"):
        isolated_module._worker(  # noqa: SLF001 - exercise the child error channel directly
            "source", "output", "digest", Connection()
        )

    assert "No space left on device" in messages[0]
    assert closed
