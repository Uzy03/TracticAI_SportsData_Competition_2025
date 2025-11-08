"""Unit tests for minimal ReceiverHead."""

import torch
import torch.nn.functional as F

from tacticai.models.mlp_heads import ReceiverHead, mask_logits


def test_receiver_head_pointwise_linear():
    """H:[B,N,d] -> Linear -> squeeze, logits differ across nodes."""
    B, N, D = 2, 4, 3
    head = ReceiverHead(input_dim=D)

    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([[0.1, 0.2, 0.3]]))
        head.linear.bias.zero_()

    H = torch.arange(B * N * D, dtype=torch.float32).view(B, N, D)
    logits = head(H)  # [B, N]
    assert logits.shape == (B, N)
    assert not torch.allclose(logits[0, 0], logits[0, 1])
    assert not torch.allclose(logits[1, 2], logits[1, 3])


def test_receiver_head_mask_softmax_ce():
    """cand_mask -> -1e9 add -> softmax -> CE."""
    N, D = 5, 2
    head = ReceiverHead(input_dim=D)
    with torch.no_grad():
        head.linear.weight.fill_(1.0)
        head.linear.bias.zero_()

    H = torch.arange(N * D, dtype=torch.float32).view(1, N, D)
    logits = head(H)  # [1, N]

    cand_mask = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.bool)
    masked_logits = mask_logits(logits, cand_mask)
    probs = F.softmax(masked_logits, dim=-1)
    assert torch.allclose(probs[~cand_mask], torch.zeros_like(probs[~cand_mask]), atol=1e-6)

    target = torch.tensor([1])
    loss = F.nll_loss(torch.log(probs + 1e-12), target)
    assert torch.isfinite(loss)
