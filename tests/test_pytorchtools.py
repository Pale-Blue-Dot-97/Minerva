# -*- coding: utf-8 -*-
import tempfile
from pathlib import Path

from torchvision.models import alexnet

from minerva.pytorchtools import EarlyStopping


def test_earlystopping() -> None:
    path = Path(tempfile.gettempdir(), "exp1.pt")

    path.unlink(missing_ok=True)

    stopper = EarlyStopping(patience=3, verbose=True, path=path)

    assert isinstance(stopper, EarlyStopping)

    model = alexnet()

    for loss in (2.2, 2.1, 3.4, 2.5, 2.0, 1.8, 1.9, 2.0):
        stopper(loss, model)
        assert stopper.early_stop is False

    stopper(2.1, model)
    assert stopper.early_stop

    path.unlink(missing_ok=True)
