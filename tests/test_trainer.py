from minerva.trainer import Trainer
from minerva.utils.utils import config, set_seeds

set_seeds(42)


def test_trainer() -> None:
    params = config.copy()
    trainer = Trainer(**params)
    assert isinstance(trainer, Trainer)

    trainer.fit()

    trainer.test()
