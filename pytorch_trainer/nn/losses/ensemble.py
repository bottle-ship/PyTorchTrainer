import typing as t

import torch
import torch.nn as nn

from .loss import Loss

__all__ = [
    "ensemble_loss",
    "EnsembleLoss"
]

T = t.TypeVar("T", torch.Tensor, t.Sequence[torch.Tensor])


def ensemble_loss(
        inputs: t.List[T],
        targets: t.List[T],
        loss_fn: t.Union[t.Callable[[...], ...], nn.Module],
        reduction: t.Literal["none", "mean", "sum"] = "mean"
) -> t.List[torch.Tensor]:
    loss = list()
    for inputs_i, targets_i in zip(inputs, targets):  # noqa
        loss_i = loss_fn(inputs_i, targets_i)
        if reduction == "mean":
            loss_i = torch.mean(loss_i)
        elif reduction == "sum":
            loss_i = torch.sum(loss_i)
        loss.append(loss_i)

    return loss


class EnsembleLoss(Loss):

    def __init__(
            self,
            loss_fn: t.Union[t.Callable[[...], ...], nn.Module],
            reduction: t.Literal["none", "mean", "sum"] = "mean"
    ):
        super(EnsembleLoss, self).__init__(reduction=reduction)
        self._loss_fn = loss_fn

    def forward(self, inputs: t.List[T], targets: t.List[T]) -> t.List[torch.Tensor]:
        return ensemble_loss(inputs, targets, loss_fn=self._loss_fn, reduction=self.reduction)
