from minerva.trainer import Trainer
from minerva.utils import config


def test_trainer_init() -> None:
    trainer = Trainer(**config)
    assert isinstance(trainer, Trainer)
