"""Model implementations for TacticAI tasks."""

from .gatv2 import (
    GATv2Layer,
    GATv2Layer4View,
    GATv2Network,
    GATv2Network4View,
)
from .mlp_heads import (
    ReceiverHead,
    ShotHead,
    ShotHeadConditional,
    ShotHeadNodeBased,
    CVAEHead,
    marginalize_shot,
)
from .heads.generator import CVAEGenerator, ConditionLayout
from .cvae import CVAEEncoder, CVAEDecoder, CVAEModel
from .multitask_model import MultiTaskModel

__all__ = [
    "GATv2Layer",
    "GATv2Layer4View",
    "GATv2Network",
    "GATv2Network4View",
    "ReceiverHead",
    "ShotHead",
    "ShotHeadConditional",
    "ShotHeadNodeBased",
    "CVAEHead",
    "marginalize_shot",
    "CVAEGenerator",
    "ConditionLayout",
    "CVAEEncoder",
    "CVAEDecoder",
    "CVAEModel",
    "MultiTaskModel",
]
