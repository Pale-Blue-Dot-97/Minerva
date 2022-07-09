from minerva.trainer import Trainer
from minerva.utils import config


def test_trainer() -> None:
    params = config.copy()
    trainer = Trainer(**params)
    assert isinstance(trainer, Trainer)

    trainer.fit()
