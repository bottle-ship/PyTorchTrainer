import typing as t

from ignite.engine import Events, Engine
from ignite.handlers import EarlyStopping as IgniteEarlyStopping

from .callback import Callback

__all__ = ["EarlyStopping"]


class EarlyStopping(Callback):

    def __init__(self, monitor: str, patience: int, min_delta: float = 0.0, cumulative_delta: bool = False):
        super(EarlyStopping, self).__init__(
            engine="val_evaluator",
            event_name=Events.COMPLETED,
            every=None
        )

        self._monitor = monitor
        self._patience = patience
        self._min_delta = min_delta
        self._cumulative_delta = cumulative_delta

    def build_handler(self, trainer: t.Optional[Engine] = None) -> t.Callable:
        return IgniteEarlyStopping(
            patience=self._patience,
            score_function=lambda engine: engine.state.metrics[self._monitor],
            trainer=trainer,
            min_delta=self._min_delta,
            cumulative_delta=self._cumulative_delta
        )
