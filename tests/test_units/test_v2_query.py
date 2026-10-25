import gzip
from dataclasses import FrozenInstanceError

import pytest

from torchvnnlib import (
    TorchVNNLIB,
    VNNLibComparator,
    VNNLibFlatInputBound,
    VNNLibInputDomain,
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
    assert len(query.global_input_domain) == 1
    input_bound = query.global_input_domain.bound_at(0)
    assert input_bound.tensor_name == "X"
    assert input_bound.indices == (0, 0)
    assert input_bound.comparator is VNNLibComparator.GE
    assert input_bound.lexeme == "0.0"
    assert len(query.branches) == 2
    assert query.branches[0][0].comparator is VNNLibComparator.LT
    assert query.capabilities.has_strict_comparisons
    variable = query.branches[0][0].lhs
    assert isinstance(variable, VNNLibVariable)
    assert variable.indices == (0,)
    with pytest.raises(FrozenInstanceError):
        query.version = VNNLibVersion(1, 0)  # type: ignore[misc]


def test_parse_query_packs_global_and_branch_local_input_bounds():
    query = TorchVNNLIB.parse_query(
        lines="""(vnnlib-version <2.0>)
(declare-network f
  (declare-input X float32 [2])
  (declare-output Y float32 [1]))
(assert (and (>= X[0] -0.1250) (< 1.500 X[1])))
(assert (or (<= X[0] 0.0) (>= Y[0] 2.0)))
""".splitlines(keepends=True)
    )

    assert [
        (bound.tensor_name, bound.indices, bound.comparator, bound.lexeme)
        for bound in query.global_input_domain
    ] == [
        ("X", (0,), VNNLibComparator.GE, "-0.1250"),
        ("X", (1,), VNNLibComparator.GT, "1.500"),
    ]
    assert len(query.branches) == 2
    assert len(query.branch_input_domains) == 2
    assert [
        (bound.tensor_name, bound.indices, bound.comparator, bound.lexeme)
        for bound in query.branch_input_domains[0]
    ] == [("X", (0,), VNNLibComparator.LE, "0.0")]
    assert len(query.branch_input_domains[1]) == 0
    assert query.branches[0] == ()
    assert query.branches[1][0].comparator is VNNLibComparator.GE


def test_parse_query_keeps_input_only_disjunction_as_packed_branches():
    query = TorchVNNLIB.parse_query(
        lines="""(vnnlib-version <2.0>)
(declare-network f
  (declare-input X float32 [1])
  (declare-output Y float32 [1]))
(assert (or (< X[0] 0.0) (> X[0] 1.0)))
""".splitlines(keepends=True)
    )

    assert query.branches == ((), ())
    assert [domain.bound_at(0).comparator for domain in query.branch_input_domains] == [
        VNNLibComparator.LT,
        VNNLibComparator.GT,
    ]
    assert query.capabilities.has_strict_comparisons


def test_input_domain_rejects_corrupt_packed_records():
    with pytest.raises(ValueError, match="truncated"):
        VNNLibInputDomain(records=b"broken")


def test_parse_query_deduplicates_packed_numeric_lexemes():
    query = TorchVNNLIB.parse_query(
        lines="""(vnnlib-version <2.0>)
(declare-network f
  (declare-input X float32 [2])
  (declare-output Y float32 [1]))
(assert (and (>= X[0] 0.0) (>= X[1] 0.0)))
""".splitlines(keepends=True)
    )

    assert query.global_input_domain.lexemes == b"0.0"
    assert [bound.lexeme for bound in query.global_input_domain] == ["0.0", "0.0"]


def test_input_domain_iterates_validated_flat_bounds():
    query = TorchVNNLIB.parse_query(
        lines="""(vnnlib-version <2.0>)
(declare-network f
  (declare-input X float32 [2,2])
  (declare-output Y float32 [1]))
(assert (and (>= X[0,1] 0.0) (<= X[1,1] 0.0)))
""".splitlines(keepends=True)
    )

    bounds = list(query.global_input_domain.iter_flat_bounds())
    fields = list(query.global_input_domain.iter_flat_bound_fields())

    assert bounds == [
        VNNLibFlatInputBound("X", 1, VNNLibComparator.GE, "0.0"),
        VNNLibFlatInputBound("X", 3, VNNLibComparator.LE, "0.0"),
    ]
    assert bounds[0].lexeme is bounds[1].lexeme
    assert fields == [
        ("X", 1, VNNLibComparator.GE, "0.0"),
        ("X", 3, VNNLibComparator.LE, "0.0"),
    ]
    assert fields[0][3] is fields[1][3]


def test_official_float_nodes_expose_exact_lexemes():
    query = TorchVNNLIB.parse_query(
        lines=SIMPLE_QUERY.replace("0.0", "0.1000", 1).splitlines(keepends=True)
    )

    assert query.global_input_domain.bound_at(0).lexeme == "0.1000"


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
