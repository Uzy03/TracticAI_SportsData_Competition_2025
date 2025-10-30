"""Utilities for handling D2 (four-view) transformations.

This module centralizes utilities that keep the four canonical views
identity (id), horizontal flip (H), vertical flip (V), and both flips (HV)
aligned. In particular, :func:`align_views` can be used to map logits or
features produced per-view back to the canonical (identity) orientation so
they can be compared or aggregated.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import torch

# Canonical ordering for D2 views.
D2_VIEWS: Tuple[str, ...] = ("id", "h", "v", "hv")

# Mapping from view index to the sign that should be applied to (x, y) style
# coordinates to undo the view-specific flip and return to identity space.
_VIEW_SIGNS: Tuple[Tuple[int, int], ...] = (
    (1, 1),    # id
    (-1, 1),   # h
    (1, -1),   # v
    (-1, -1),  # hv
)

IndexLike = Optional[Union[int, Sequence[int]]]


def _canon_indices(idxs: IndexLike) -> Optional[Tuple[int, ...]]:
    """Normalize index specification to a canonical tuple.

    Accepts integers, iterables of integers, or ``None`` and converts them into
    a tuple of positive indices (relative to the last dimension).
    """
    if idxs is None:
        return None
    if isinstance(idxs, int):
        return (idxs,)
    if isinstance(idxs, Iterable):
        return tuple(int(i) for i in idxs)
    raise TypeError(f"Unsupported index specification: {idxs!r}")


def _apply_signs(
    tensor: torch.Tensor,
    indices: Optional[Tuple[int, ...]],
    sign: int,
) -> None:
    """Apply sign flips to the specified indices in-place."""
    if indices is None:
        return
    last_dim = tensor.dim() - 1
    for idx in indices:
        canonical_idx = idx if idx >= 0 else tensor.size(last_dim) + idx
        tensor[..., canonical_idx] = tensor[..., canonical_idx] * sign


def apply_view_transform(
    tensor: torch.Tensor,
    view: Union[int, str],
    xy_indices: Tuple[IndexLike, IndexLike] = (0, 1),
) -> torch.Tensor:
    """Return a copy of ``tensor`` after applying the specified D2 view.

    Args:
        tensor: Data to transform. Assumes coordinate-style features live in the
            last dimension (e.g. [..., D]).
        view: View index (0-3) or name in ``D2_VIEWS``.
        xy_indices: Tuple of indices for x-like and y-like coordinates that
            should change sign under horizontal/vertical flips. Pass ``None`` if
            the tensor does not contain such coordinates.

    Returns:
        Transformed tensor that matches the requested view orientation.
    """
    view_idx = _resolve_view_index(view)
    x_indices = _canon_indices(xy_indices[0])
    y_indices = _canon_indices(xy_indices[1])

    out = tensor.clone()
    sign_x, sign_y = _VIEW_SIGNS[view_idx]
    _apply_signs(out, x_indices, sign_x)
    _apply_signs(out, y_indices, sign_y)
    return out


def align_views(
    tensor: torch.Tensor,
    view_axis: int = 1,
    xy_indices: Tuple[IndexLike, IndexLike] = (0, 1),
) -> torch.Tensor:
    """Align a per-view tensor back to the canonical (identity) orientation.

    This is typically used on logits or features produced by a four-view model.
    The function undoes the horizontal/vertical flips corresponding to each
    view so that the aligned tensor can be compared or aggregated directly.

    Args:
        tensor: Input tensor that includes a view dimension.
        view_axis: Axis along which the four D2 views are stored.
        xy_indices: Tuple of indices (or iterables of indices) for features that
            should change sign when undoing horizontal (x) and vertical (y)
            flips. Use ``None`` for entries that are flip-invariant.

    Returns:
        Tensor with the same shape as ``tensor`` but with every view expressed
        in the canonical orientation.
    """
    if tensor.dim() < 1:
        raise ValueError("Expected tensor with at least one dimension.")

    view_axis = view_axis if view_axis >= 0 else tensor.dim() + view_axis
    if tensor.size(view_axis) != len(D2_VIEWS):
        raise ValueError(
            f"Expected view axis of length {len(D2_VIEWS)}, "
            f"got {tensor.size(view_axis)}."
        )

    x_indices = _canon_indices(xy_indices[0])
    y_indices = _canon_indices(xy_indices[1])

    # Bring the view axis to position 1 for simpler iteration.
    permute_order = list(range(tensor.dim()))
    permute_order[1], permute_order[view_axis] = (
        permute_order[view_axis],
        permute_order[1],
    )
    aligned = tensor.permute(*permute_order).clone()

    for view_idx, (sign_x, sign_y) in enumerate(_VIEW_SIGNS):
        view_tensor = aligned.select(1, view_idx)
        _apply_signs(view_tensor, x_indices, sign_x)
        _apply_signs(view_tensor, y_indices, sign_y)

    # Undo permutation so the output matches the original layout.
    inverse_perm = [0] * tensor.dim()
    for i, p in enumerate(permute_order):
        inverse_perm[p] = i
    return aligned.permute(*inverse_perm)


def _resolve_view_index(view: Union[int, str]) -> int:
    """Resolve a view index from either integer or string form."""
    if isinstance(view, int):
        if 0 <= view < len(D2_VIEWS):
            return view
        raise ValueError(f"View index must be in [0, {len(D2_VIEWS) - 1}]")
    view_lower = view.lower()
    if view_lower in D2_VIEWS:
        return D2_VIEWS.index(view_lower)
    raise ValueError(f"Unknown D2 view: {view}")


__all__ = ["D2_VIEWS", "align_views", "apply_view_transform"]
