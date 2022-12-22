import numpy as np
import torch
from torch import Tensor

from minerva.models.__depreciated import MLP, CNN


def test_mlp(x_entropy_loss) -> None:
    model_1 = MLP(x_entropy_loss)
    model_2 = MLP(x_entropy_loss, hidden_sizes=128)

    for model in (model_1, model_2):
        optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

        model.set_optimiser(optimiser)

        model.determine_output_dim()
        assert model.output_shape is model.n_classes

        x = torch.rand(16, (288))
        y = torch.LongTensor(np.random.randint(0, 8, size=16))

        for mode in ("train", "val", "test"):
            if mode == "train":
                loss, z = model.step(x, y, train=True)
            else:
                loss, z = model.step(x, y, train=False)

            assert type(loss.item()) is float
            assert isinstance(z, Tensor)
            assert z.size() == (16, 8)


def test_cnn(x_entropy_loss) -> None:
    input_size = (4, 64, 64)
    model = CNN(x_entropy_loss, input_size=input_size)

    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert model.output_shape is model.n_classes

    x = torch.rand(6, *input_size)
    y = torch.LongTensor(np.random.randint(0, 8, size=6))

    loss, z = model.step(x, y, train=True)

    assert type(loss.item()) is float
    assert isinstance(z, Tensor)
    assert z.size() == (6, 8)
