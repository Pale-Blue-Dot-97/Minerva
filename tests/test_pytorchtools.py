from minerva.pytorchtools import EarlyStopping
import os
import tempfile
from torchvision.models import alexnet


def test_earlystopping() -> None:
    path = tempfile.gettempdir()

    exp_name = "exp1.pt"

    path = os.path.join(path, exp_name)

    if os.path.exists(path):
        os.remove(path)

    stopper = EarlyStopping(patience=3, verbose=True, path=path)

    assert isinstance(stopper, EarlyStopping)

    model = alexnet()

    for loss in (2.2, 2.1, 3.4, 2.5, 2.0, 1.8, 1.9, 2.0):
        stopper(loss, model)
        assert stopper.early_stop is False

    stopper(2.1, model)
    assert stopper.early_stop

    os.remove(path)
