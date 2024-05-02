import typing as t

import torch
# noinspection PyProtectedMember
from torch.nn.modules.loss import (
    _Loss,
    _WeightedLoss
)

__all__ = [
    "Loss",
    "WeightedLoss"
]


class Loss(_Loss):
    reduction: t.Literal["none", "mean", "sum"]

    def __init__(self, reduction: t.Literal["none", "mean", "sum"] = "mean"):
        super(Loss, self).__init__(reduction=reduction)


class WeightedLoss(_WeightedLoss):
    weight: t.Optional[torch.Tensor]
    reduction: t.Literal["none", "mean", "sum"]

    def __init__(
            self,
            weight: t.Optional[torch.Tensor] = None,
            reduction: t.Literal["none", "mean", "sum"] = "mean"
    ):
        super(WeightedLoss, self).__init__(weight=weight, reduction=reduction)
