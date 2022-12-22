import pytest
import numpy as np
import torch
from torch import Tensor

from minerva.models.__depreciated import MLP


def test_minervamodel(x_entropy_loss) -> None:
    x = torch.rand(16, (288))
    y = torch.LongTensor(np.random.randint(0, 8, size=16))

    with pytest.raises(NotImplementedError, match="Criterion has not been set!"):
        model_fail = MLP()
        optimiser = torch.optim.SGD(model_fail.parameters(), lr=1.0e-3)

        model_fail.set_optimiser(optimiser)
        _ = model_fail.step(x, y, train=True)

    model = MLP(x_entropy_loss)

    with pytest.raises(NotImplementedError, match="Optimiser has not been set!"):
        _ = model.step(x, y, train=True)

    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert model.output_shape is model.n_classes

    for mode in ("train", "val", "test"):
        if mode == "train":
            loss, z = model.step(x, y, train=True)
        else:
            loss, z = model.step(x, y, train=False)

        assert type(loss.item()) is float
        assert isinstance(z, Tensor)
        assert z.size() == (16, 8)
