import numpy as np
import pytest
import torch
from torch import Tensor, LongTensor
import torch.nn.modules as nn
from lightly.loss import NTXentLoss, NegativeCosineSimilarity

from minerva.models import (
    MinervaModel,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    SimCLR18,
    SimCLR34,
    SimCLR50,
    SimSiam18,
    SimSiam34,
    SimSiam50,
    FCN8ResNet18,
    FCN8ResNet34,
    FCN8ResNet50,
    FCN8ResNet101,
    FCN8ResNet152,
    FCN16ResNet18,
    FCN16ResNet34,
    FCN16ResNet50,
    FCN32ResNet18,
    FCN32ResNet34,
    FCN32ResNet50,
)
from minerva.models.__depreciated import MLP, CNN

criterion = nn.CrossEntropyLoss()


def test_mlp() -> None:
    model_1 = MLP(criterion)
    model_2 = MLP(criterion, hidden_sizes=128)

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


def test_cnn() -> None:
    input_size = (4, 64, 64)
    model = CNN(criterion, input_size=input_size)

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


def test_resnets() -> None:
    def resnet_test(test_model: MinervaModel, x: Tensor, y: Tensor) -> None:
        optimiser = torch.optim.SGD(test_model.parameters(), lr=1.0e-3)

        test_model.set_optimiser(optimiser)

        test_model.determine_output_dim()
        assert test_model.output_shape is test_model.n_classes

        loss, z = test_model.step(x, y, True)

        assert type(loss.item()) is float
        assert isinstance(z, Tensor)
        assert z.size() == (6, 8)

    with pytest.raises(ValueError):
        _ = ResNet18(replace_stride_with_dilation=(True, False))  # type: ignore[arg-type]

    input_size = (4, 64, 64)

    x = torch.rand(6, *input_size)
    y = LongTensor(np.random.randint(0, 8, size=6))

    for zero_init_residual in (True, False):

        resnet18 = ResNet18(
            criterion, input_size=input_size, zero_init_residual=zero_init_residual
        )

        resnet_test(resnet18, x, y)

    for model in (
        ResNet34(criterion, input_size=input_size),
        ResNet50(criterion, input_size=input_size),
        ResNet50(
            criterion,
            input_size=input_size,
            replace_stride_with_dilation=(True, True, False),
            zero_init_residual=True,
        ),
        ResNet101(criterion, input_size=input_size),
        ResNet152(criterion, input_size=input_size),
    ):
        resnet_test(model, x, y)

    encoder = ResNet18(criterion, input_size=input_size, encoder=True)
    optimiser = torch.optim.SGD(encoder.parameters(), lr=1.0e-3)

    encoder.set_optimiser(optimiser)

    encoder.determine_output_dim()
    print(encoder.output_shape)
    assert encoder.output_shape == (512, 2, 2)

    x = torch.rand(6, *input_size)
    assert len(encoder(x)) == 5


def test_fcnresnets() -> None:
    def resnet_test(test_model: MinervaModel, x: Tensor, y: Tensor) -> None:
        optimiser = torch.optim.SGD(test_model.parameters(), lr=1.0e-3)

        test_model.set_optimiser(optimiser)

        test_model.determine_output_dim()
        assert test_model.output_shape == (64, 64)

        loss, z = test_model.step(x, y, True)

        assert type(loss.item()) is float
        assert isinstance(z, Tensor)
        assert z.size() == (6, 8, 64, 64)

    input_size = (4, 64, 64)

    x = torch.rand((6, *input_size))
    y = torch.randint(0, 8, (6, 64, 64))  # type: ignore[attr-defined]

    for model in (
        FCN32ResNet18(criterion, input_size=input_size),
        FCN32ResNet34(criterion, input_size=input_size),
        FCN32ResNet50(criterion, input_size=input_size),
        FCN16ResNet18(criterion, input_size=input_size),
        FCN16ResNet34(criterion, input_size=input_size),
        FCN16ResNet50(criterion, input_size=input_size),
        FCN8ResNet18(criterion, input_size=input_size),
        FCN8ResNet18(criterion, input_size=input_size),
        FCN8ResNet34(criterion, input_size=input_size),
        FCN8ResNet50(criterion, input_size=input_size),
        FCN8ResNet101(criterion, input_size=input_size),
        FCN8ResNet152(criterion, input_size=input_size),
    ):
        resnet_test(model, x, y)


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
