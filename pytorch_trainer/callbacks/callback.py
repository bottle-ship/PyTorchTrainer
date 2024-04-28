import typing as t
from abc import ABC, abstractmethod

from ignite.engine import Engine
from ignite.engine.events import CallableEventWithFilter

__all__ = ["Callback"]


class Callback(ABC):

    def __init__(
            self,
            engine: t.Literal["trainer", "train_evaluator", "val_evaluator"],
            event_name: CallableEventWithFilter,
            every: t.Optional[int] = None
    ):
        self.engine = engine
        self.event_name = event_name
        self.every = every

    def __call__(
            self,
            trainer: t.Optional[Engine] = None,
            train_evaluator: t.Optional[Engine] = None,
            val_evaluator: t.Optional[Engine] = None
    ):
        engine = None
        if self.engine == "trainer":
            engine = trainer
        elif self.engine == "train_evaluator":
            engine = train_evaluator
        elif self.engine == "val_evaluator":
            engine = val_evaluator

        engine.add_event_handler(event_name=self.event_name(every=self.every), handler=self.build_handler(trainer))

    @abstractmethod
    def build_handler(self, trainer: t.Optional[Engine] = None) -> t.Callable:
        raise NotImplemented
