import torch
import torch.nn.functional as F

from tacticai.models.mlp_heads import ReceiverHead


def test_receiver_head_pointwise_logits():
    batch_size, num_nodes, feature_dim = 2, 4, 3
    head = ReceiverHead(input_dim=feature_dim)

    with torch.no_grad():
        head.linear.weight.copy_(torch.arange(feature_dim).float().unsqueeze(0))
        head.linear.bias.zero_()

    hidden = torch.arange(batch_size * num_nodes * feature_dim, dtype=torch.float32)
    hidden = hidden.view(batch_size, num_nodes, feature_dim)

    logits = head(hidden)
    assert logits.shape == (batch_size, num_nodes)
    assert not torch.allclose(logits[0, 0], logits[0, 1])
    assert not torch.allclose(logits[0, 0], logits[1, 0])


def test_receiver_head_mask_and_cross_entropy():
    num_nodes, feature_dim = 5, 3
    head = ReceiverHead(input_dim=feature_dim)

    with torch.no_grad():
        head.linear.weight.fill_(1.0)
        head.linear.bias.zero_()

    hidden = torch.arange(num_nodes * feature_dim, dtype=torch.float32).view(1, num_nodes, feature_dim)
    logits = head(hidden)  # [1, N]

    cand_mask = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.bool)
    masked_logits = logits + (~cand_mask) * (-1e9)
    probs = torch.softmax(masked_logits, dim=-1)

    assert torch.allclose(probs[cand_mask.logical_not()], torch.zeros_like(probs[cand_mask.logical_not()]), atol=1e-6)

    target = torch.tensor([1])
    nll = -torch.log(probs[torch.arange(1), target])
    assert torch.isfinite(nll).all()

