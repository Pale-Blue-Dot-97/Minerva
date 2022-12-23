import pytest
import numpy as np
import torch
from torch import Tensor
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum
from lightly.loss import NTXentLoss
import internet_sabotage

from minerva.models import SimCLR18, get_torch_weights, get_output_shape, bilinear_init
from minerva.models.__depreciated import MLP


def test_minerva_model(x_entropy_loss) -> None:
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


def test_minerva_backbone() -> None:
    loss_func = NTXentLoss(0.3)
    input_size = (4, 64, 64)

    model = SimCLR18(loss_func, input_size=input_size)

    assert isinstance(model.get_backbone(), Module)


def test_minerva_dataparallel() -> None:
    pass


def test_get_torch_weights() -> None:
    weights = get_torch_weights("ResNet18_Weights.IMAGENET1K_V1")

    assert isinstance(weights, WeightsEnum)

    with internet_sabotage.no_connection():
        weights = get_torch_weights("ResNet18_Weights.IMAGENET1K_V1")


def test_get_output_shape(exp_mlp) -> None:
    output_shape = get_output_shape(exp_mlp, 64)

    assert output_shape == 8


def test_bilinear_init() -> None:
    weights = bilinear_init(12, 12, 5)
    assert isinstance(weights, Tensor)
