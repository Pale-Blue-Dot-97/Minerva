import pytest
import torch
from lightly.loss import NTXentLoss, NegativeCosineSimilarity

from minerva.models import (
    SimCLR18,
    SimCLR34,
    SimCLR50,
    SimSiam18,
    SimSiam34,
    SimSiam50,
)


def test_simclr() -> None:
    loss_func = NTXentLoss(0.3)

    input_size = (4, 64, 64)

    x = torch.rand((6, *input_size))

    x = torch.stack([x, x])

    for model in (
        SimCLR18(loss_func, input_size=input_size),
        SimCLR34(loss_func, input_size=input_size),
        SimCLR50(loss_func, input_size=input_size),
    ):
        optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

        model.set_optimiser(optimiser)

        model.determine_output_dim(sample_pairs=True)
        assert model.output_shape == (128,)

        loss, z = model.step(x, train=True)

        assert type(loss.item()) is float
        assert z.size() == (12, 128)

    model = SimCLR18(loss_func)

    with pytest.raises(NotImplementedError, match="Optimiser has not been set!"):
        _ = model.step(x, train=True)


def test_simsiam() -> None:
    loss_func = NegativeCosineSimilarity()

    input_size = (4, 64, 64)

    x = torch.rand((6, *input_size))

    x = torch.stack([x, x])

    for model in (
        SimSiam18(loss_func, input_size=input_size),
        SimSiam34(loss_func, input_size=input_size),
        SimSiam50(loss_func, input_size=input_size),
    ):
        optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

        model.set_optimiser(optimiser)

        model.determine_output_dim(sample_pairs=True)
        assert model.output_shape == (128,)

        loss, z = model.step(x, train=True)

        assert type(loss.item()) is float
        assert z.size() == (12, 128)

    model = SimSiam18(loss_func)

    with pytest.raises(NotImplementedError, match="Optimiser has not been set!"):
        _ = model.step(x, train=True)
