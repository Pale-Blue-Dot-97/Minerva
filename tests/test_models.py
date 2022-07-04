import numpy as np
import pytest
import torch
from lightly.loss import NTXentLoss

import minerva.models as mm

criterion = torch.nn.CrossEntropyLoss()


def test_mlp() -> None:
    model_1 = mm.MLP(criterion)
    model_2 = mm.MLP(criterion, hidden_sizes=128)

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
            assert z.size() == (16, 8)


def test_cnn() -> None:
    input_size = (4, 64, 64)
    model = mm.CNN(criterion, input_size=input_size)

    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert model.output_shape is model.n_classes

    x = torch.rand(6, *input_size)
    y = torch.LongTensor(np.random.randint(0, 8, size=6))

    loss, z = model.step(x, y, train=True)

    assert type(loss.item()) is float
    assert z.size() == (6, 8)


def test_resnets() -> None:
    def resnet_test(
        test_model: mm.MinervaModel, x: torch.FloatTensor, y: torch.LongTensor
    ) -> None:
        optimiser = torch.optim.SGD(test_model.parameters(), lr=1.0e-3)

        test_model.set_optimiser(optimiser)

        test_model.determine_output_dim()
        assert test_model.output_shape is test_model.n_classes

        loss, z = test_model.step(x, y, True)

        assert type(loss.item()) is float
        assert z.size() == (6, 8)

    with pytest.raises(ValueError):
        _ = mm.ResNet18(replace_stride_with_dilation=(True, False))

    input_size = (4, 64, 64)

    x = torch.rand(6, *input_size)
    y = torch.LongTensor(np.random.randint(0, 8, size=6))

    for zero_init_residual in (True, False):

        model = mm.ResNet18(
            criterion, input_size=input_size, zero_init_residual=zero_init_residual
        )

        resnet_test(model, x, y)

    for model in (
        mm.ResNet34(criterion, input_size=input_size),
        mm.ResNet50(criterion, input_size=input_size),
        mm.ResNet50(
            criterion,
            input_size=input_size,
            replace_stride_with_dilation=(True, True, False),
            zero_init_residual=True,
        ),
        mm.ResNet101(criterion, input_size=input_size),
        mm.ResNet152(criterion, input_size=input_size),
    ):
        resnet_test(model, x, y)

    encoder = mm.ResNet18(criterion, input_size=input_size, encoder=True)
    optimiser = torch.optim.SGD(encoder.parameters(), lr=1.0e-3)

    encoder.set_optimiser(optimiser)

    encoder.determine_output_dim()
    print(encoder.output_shape)
    assert encoder.output_shape == (512, 2, 2)

    x = torch.rand(6, *input_size)
    assert len(encoder(x)) == 5


def test_fcnresnets() -> None:
    def resnet_test(
        test_model: mm.MinervaModel, x: torch.FloatTensor, y: torch.LongTensor
    ) -> None:
        optimiser = torch.optim.SGD(test_model.parameters(), lr=1.0e-3)

        test_model.set_optimiser(optimiser)

        test_model.determine_output_dim()
        assert test_model.output_shape == (64, 64)

        loss, z = test_model.step(x, y, True)

        assert type(loss.item()) is float
        assert z.size() == (6, 8, 64, 64)

    input_size = (4, 64, 64)

    x = torch.rand((6, *input_size))
    y = torch.randint(0, 8, (6, 64, 64))

    for model in (
        mm.FCN32ResNet18(criterion, input_size=input_size),
        mm.FCN32ResNet34(criterion, input_size=input_size),
        mm.FCN32ResNet50(criterion, input_size=input_size),
        mm.FCN16ResNet18(criterion, input_size=input_size),
        mm.FCN16ResNet34(criterion, input_size=input_size),
        mm.FCN16ResNet50(criterion, input_size=input_size),
        mm.FCN8ResNet18(criterion, input_size=input_size),
        mm.FCN8ResNet18(criterion, input_size=input_size),
        mm.FCN8ResNet34(criterion, input_size=input_size),
        mm.FCN8ResNet50(criterion, input_size=input_size),
        mm.FCN8ResNet101(criterion, input_size=input_size),
        mm.FCN8ResNet152(criterion, input_size=input_size),
    ):
        resnet_test(model, x, y)


def test_simclr() -> None:
    loss_func = NTXentLoss(0.3)

    input_size = (4, 64, 64)

    x = torch.rand((6, *input_size))

    x = torch.stack([x, x])

    for model in (
        mm.SimCLR18(loss_func, input_size=input_size),
        mm.SimCLR34(loss_func, input_size=input_size),
        mm.SimCLR50(loss_func, input_size=input_size),
    ):
        optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

        model.set_optimiser(optimiser)

        model.determine_output_dim(sample_pairs=True)
        assert model.output_shape == (128,)

        loss, z = model.step(x, train=True)

        assert type(loss.item()) is float
        assert z.size() == (12, 128)
