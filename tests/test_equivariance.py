"""Tests for D2 view alignment utilities."""

import torch

from tacticai.modules.view_ops import D2_VIEWS, align_views, apply_view_transform


def _make_four_view_tensor(
    base: torch.Tensor,
    xy_indices=(0, 1),
) -> torch.Tensor:
    """Expand base features into the four canonical D2 views."""
    views = [
        apply_view_transform(base, view, xy_indices=xy_indices) for view in D2_VIEWS
    ]
    return torch.stack(views, dim=1)


def test_align_views_recovers_identity():
    """align_views should undo the per-view flips and recover the base tensor."""
    batch, nodes, dims = 3, 6, 4
    base = torch.randn(batch, nodes, dims)
    four_view = _make_four_view_tensor(base, xy_indices=(0, 1))

    aligned = align_views(four_view, view_axis=1, xy_indices=(0, 1))

    for idx in range(len(D2_VIEWS)):
        assert torch.allclose(aligned[:, idx], base, atol=1e-6)


def test_horizontal_flip_is_involution():
    """Applying the horizontal flip twice should return the original tensor."""
    data = torch.randn(5, 7, 3)
    flipped_once = apply_view_transform(data, "h", xy_indices=(0, 1))
    flipped_twice = apply_view_transform(flipped_once, "h", xy_indices=(0, 1))
    assert torch.allclose(flipped_twice, data, atol=1e-6)
