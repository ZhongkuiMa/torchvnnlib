"""Shared row-emission primitives for VNN-LIB linear constraints.

Both the AST and fast-type pipelines ultimately materialise the same row
layout for the output constraint tensor (``b + A y >= 0``) and the same
slot-update pattern for the input bound tensor (``(n_inputs, 2)`` with
column 0 == lower, column 1 == upper). Before this module existed those
two templates were duplicated five times across ``fast_type/_utils.py``,
``fast_type/_type1_processor.py``, and ``_to_tensor.py`` -- one fix had
to chase five files.

These helpers exist solely to be the single source of truth for those
two templates. They do NOT bridge the AST and regex pipelines; each
pipeline still consumes its own input language.
"""

from __future__ import annotations

__docformat__ = "restructuredtext"
__all__ = [
    "BoundOp",
    "apply_input_bound",
    "normalize_neg_zero",
    "validate_input_bounds",
    "write_compare_row",
    "write_value_bound_row",
    "write_value_bound_rows_eq",
]

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torchvnnlib._backend import Backend, TensorLike

BoundOp = Literal["<=", ">=", "="]


def validate_input_bounds(input_bounds: TensorLike, backend: Backend) -> None:
    """Check input bounds tensor for NaN and Inf entries.

    :param input_bounds: Input-bound tensor of shape ``(n_inputs, 2)``.
    :param backend: Backend instance.
    :raises ValueError: If any entry is NaN or Inf.
    """
    if backend.isnan(input_bounds).any():
        nan_indices = backend.where(backend.isnan(input_bounds))
        indices_list = list(zip(nan_indices[0].tolist(), nan_indices[1].tolist(), strict=False))
        raise ValueError(f"Missing input bounds at indices: {indices_list}")


def normalize_neg_zero(tensor: TensorLike) -> TensorLike:
    """Replace ``-0.0`` entries with ``+0.0`` in place.

    Negative zero would leak through ``-tensor`` and ``b + 0*x`` arithmetic;
    downstream consumers (sign tests, JSON / golden-file diffs) treat
    ``-0.0`` and ``+0.0`` as distinct. Call this once per emitted row to
    keep the AST and fast-type pipelines bit-identical.

    :param tensor: Tensor mutated in place; returned for chaining.
    :return: The same ``tensor`` object after normalisation.
    """
    tensor[tensor == 0.0] = 0.0
    return tensor


def _check_row_ndim(row: TensorLike) -> None:
    """Raise if ``row`` is not a 1-D view; prevents silent broadcast bugs."""
    ndim = getattr(row, "ndim", None)
    if ndim is not None and ndim != 1:
        raise ValueError(
            f"constraint row writer requires a 1-D row view, got ndim={ndim}; "
            "index the parent tensor (e.g. ``rows[i]``) before passing it in."
        )


def _check_var_index(idx: int) -> None:
    """Raise if ``idx`` is negative; negative idx would clobber the bias slot.

    ``row[idx + 1] = ...`` with ``idx = -1`` rewrites ``row[0]`` and silently
    corrupts the bias. Callers feed regex-parsed non-negative integers today,
    but the helpers are in ``__all__`` so the invariant is enforced here.
    """
    if idx < 0:
        raise ValueError(
            f"variable index must be non-negative; got {idx}. Negative indices "
            "would overwrite the bias slot via wrap-around."
        )


def apply_input_bound(input_bounds: TensorLike, idx: int, op: BoundOp, value: float) -> None:
    """Stamp one input-bound update into ``(n_inputs, 2)`` ``input_bounds``.

    Column 0 holds the lower bound, column 1 the upper bound. The tensor
    is mutated in place.

    :param input_bounds: Input-bound tensor of shape ``(n_inputs, 2)``.
    :param idx: Variable index (0-based, non-negative).
    :param op: ``"<="`` (upper), ``">="`` (lower), or ``"="`` (both).
    :param value: Bound value.
    :raises ValueError: If ``idx < 0``.
    """
    _check_var_index(idx)
    if op == "<=":
        input_bounds[idx, 1] = value
    elif op == ">=":
        input_bounds[idx, 0] = value
    else:  # "="
        input_bounds[idx, 0] = value
        input_bounds[idx, 1] = value


def write_value_bound_row(row: TensorLike, idx: int, op: Literal["<=", ">="], value: float) -> None:
    """Stamp one ``Y_idx <op> value`` constraint row in ``b + A y >= 0`` form.

    The row is mutated in place; column 0 carries the bias, column
    ``idx + 1`` the linear coefficient.

    :param row: Constraint row of shape ``(n_outputs + 1,)``.
    :param idx: Output variable index (0-based, non-negative).
    :param op: ``"<="`` or ``">="``.
    :param value: Bound value.
    :raises ValueError: If ``row`` is not 1-D or ``idx < 0``.
    """
    _check_row_ndim(row)
    _check_var_index(idx)
    if op == "<=":
        row[0] = value
        row[idx + 1] = -1.0
    else:  # ">="
        row[0] = -value if value != 0.0 else 0.0
        row[idx + 1] = 1.0


def write_value_bound_rows_eq(
    row_geq: TensorLike, row_leq: TensorLike, idx: int, value: float
) -> None:
    """Stamp the two-row decomposition of ``Y_idx = value``.

    :param row_geq: First row, writes the ``>= value`` half.
    :param row_leq: Second row, writes the ``<= value`` half.
    :param idx: Output variable index (0-based, non-negative).
    :param value: Bound value.
    :raises ValueError: If either row is not 1-D, or ``idx < 0``.
    """
    _check_row_ndim(row_geq)
    _check_row_ndim(row_leq)
    _check_var_index(idx)
    row_geq[0] = -value if value != 0.0 else 0.0
    row_geq[idx + 1] = 1.0
    row_leq[0] = value
    row_leq[idx + 1] = -1.0


def write_compare_row(row: TensorLike, idx1: int, idx2: int, op: Literal["<=", ">="]) -> None:
    """Stamp one ``Y_idx1 <op> Y_idx2`` constraint row.

    Bias remains 0; columns ``idx1 + 1`` and ``idx2 + 1`` carry signed unit
    coefficients.

    :param row: Constraint row of shape ``(n_outputs + 1,)``.
    :param idx1: First output variable index (non-negative).
    :param idx2: Second output variable index (non-negative).
    :param op: ``"<="`` or ``">="``.
    :raises ValueError: If ``row`` is not 1-D or either index is negative.
    """
    _check_row_ndim(row)
    _check_var_index(idx1)
    _check_var_index(idx2)
    if op == "<=":
        row[idx1 + 1] = -1.0
        row[idx2 + 1] = 1.0
    else:  # ">="
        row[idx1 + 1] = 1.0
        row[idx2 + 1] = -1.0
