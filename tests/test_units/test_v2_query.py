import gzip
from dataclasses import FrozenInstanceError

import pytest

from torchvnnlib import (
    TorchVNNLIB,
    VNNLibComparator,
    VNNLibMultiply,
    VNNLibTensorKind,
    VNNLibVariable,
    VNNLibVersion,
)

SIMPLE_QUERY = """(vnnlib-version <2.0>)
(declare-network f
  (declare-input X float32 [1,2])
  (declare-output Y float32 [1]))
(assert (>= X[0,0] 0.0))
(assert (or (< Y[0] 1.0) (>= Y[0] 2.0)))
"""


def test_detect_version_defaults_headerless_input_to_v1():
    assert TorchVNNLIB.detect_version(lines=["(declare-const X_0 Real)\n"]) == VNNLibVersion(1, 0)


def test_detect_version_ignores_commented_header():
    assert TorchVNNLIB.detect_version(
        lines=["; (vnnlib-version <2.0>)\n", "(declare-const X_0 Real)\n"]
    ) == VNNLibVersion(1, 0)


def test_detect_version_accepts_comments_before_header():
    assert TorchVNNLIB.detect_version(
        lines=["; language declaration follows\n", "(vnnlib-version <2.0>)\n"]
    ) == VNNLibVersion(2, 0)


def test_detect_version_accepts_multiline_header():
    assert TorchVNNLIB.detect_version(lines=["(vnnlib-version\n", "  <2.0>)\n"]) == VNNLibVersion(
        2, 0
    )


@pytest.mark.parametrize("header", ["(vnnlib-version 2)", "(vnnlib-version 2.0)"])
def test_detect_version_rejects_malformed_header(header):
    with pytest.raises(ValueError, match="Malformed VNN-LIB version declaration"):
        TorchVNNLIB.detect_version(lines=[f"{header}\n"])


def test_detect_version_rejects_late_header():
    with pytest.raises(ValueError, match="must be the first statement"):
        TorchVNNLIB.detect_version(lines=["(declare-const X_0 Real)\n", "(vnnlib-version <2.0>)\n"])


def test_parse_rejects_v2_with_actionable_api_guidance():
    with pytest.raises(ValueError, match=r"use parse_query\(\)"):
        TorchVNNLIB().parse(lines=SIMPLE_QUERY.splitlines(keepends=True))


def test_parse_query_rejects_future_minor_version():
    query = SIMPLE_QUERY.replace("<2.0>", "<2.1>", 1)

    with pytest.raises(ValueError, match=r"supports VNN-LIB 2\.0, got 2\.1"):
        TorchVNNLIB.parse_query(lines=query.splitlines(keepends=True))


def test_parse_query_owns_immutable_dnf_and_metadata():
    query = TorchVNNLIB.parse_query(lines=SIMPLE_QUERY.splitlines(keepends=True))

    assert query.version == VNNLibVersion(2, 0)
    assert len(query.networks) == 1
    assert query.networks[0].inputs[0].kind is VNNLibTensorKind.INPUT
    assert query.networks[0].inputs[0].shape == (1, 2)
    assert len(query.branches) == 2
    assert query.branches[0][0].comparator is VNNLibComparator.GE
    assert query.capabilities.has_strict_comparisons
    variable = query.branches[0][0].lhs
    assert isinstance(variable, VNNLibVariable)
    assert variable.indices == (0, 0)
    with pytest.raises(FrozenInstanceError):
        query.version = VNNLibVersion(1, 0)  # type: ignore[misc]


def test_parse_query_reads_gzip(tmp_path):
    path = tmp_path / "query.vnnlib.gz"
    with gzip.open(path, mode="wt", encoding="utf-8") as stream:
        stream.write(SIMPLE_QUERY)

    query = TorchVNNLIB.parse_query(str(path))

    assert query.version == VNNLibVersion(2, 0)
    assert len(query.branches) == 2


def test_parse_query_detects_multi_network_and_nonlinear_expression():
    query = TorchVNNLIB.parse_query(
        lines="""(vnnlib-version <2.0>)
(declare-network f
  (declare-input X float32 [1])
  (declare-output Y float32 [1]))
(declare-network g
  (declare-input Z float32 [1])
  (declare-output W float32 [1]))
(assert (<= (* X[0] Z[0]) (+ Y[0] W[0])))
""".splitlines(keepends=True)
    )

    assert query.capabilities.has_multiple_networks
    assert query.capabilities.has_nonlinear
    assert isinstance(query.branches[0][0].lhs, VNNLibMultiply)


def test_parse_query_preserves_network_relations_and_hidden_names():
    query = TorchVNNLIB.parse_query(
        lines="""(vnnlib-version <2.0>)
(declare-network f
  (declare-input X float32 [1])
  (declare-hidden H float32 [1] \"hidden_tensor\")
  (declare-output Y float32 [1]))
(declare-network g
  (isomorphic-to f)
  (declare-input Z float32 [1])
  (declare-output W float32 [1]))
(assert (>= H[0] 0.0))
""".splitlines()
    )

    assert query.networks[0].hidden[0].onnx_name == "hidden_tensor"
    assert query.networks[1].isomorphic_to == "f"
    assert query.capabilities.has_hidden
