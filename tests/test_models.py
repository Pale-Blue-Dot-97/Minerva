import minerva.models as mm
import torch
import numpy as np
import pytest


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
                loss, z = model.training_step(x, y)
            if mode == "val":
                loss, z = model.validation_step(x, y)
            if mode == "test":
                loss, z = model.testing_step(x, y)

            assert type(loss.item()) is float
            assert z.size() == (16, 8)


def test_cnn() -> None:
    model = mm.CNN(criterion)

    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert model.output_shape is model.n_classes

    x = torch.rand(16, *(4, 256, 256))
    y = torch.LongTensor(np.random.randint(0, 8, size=16))

    loss, z = model.training_step(x, y)

    assert type(loss.item()) is float
    assert z.size() == (16, 8)


def test_resnets() -> None:
    def resnet_test(
        test_model: mm.MinervaModel, x: torch.FloatTensor, y: torch.LongTensor
    ) -> None:
        optimiser = torch.optim.SGD(test_model.parameters(), lr=1.0e-3)

        test_model.set_optimiser(optimiser)

        test_model.determine_output_dim()
        assert test_model.output_shape is test_model.n_classes

        loss, z = test_model.training_step(x, y)

        assert type(loss.item()) is float
        assert z.size() == (16, 8)

    with pytest.raises(ValueError):
        _ = mm.ResNet18(replace_stride_with_dilation=(True, False))

    x = torch.rand(16, *(4, 256, 256))
    y = torch.LongTensor(np.random.randint(0, 8, size=16))

    for zero_init_residual in (True, False):

        model = mm.ResNet18(criterion, zero_init_residual=zero_init_residual)

        resnet_test(model, x, y)

    for model in (
        mm.ResNet34(criterion),
        mm.ResNet50(criterion),
        mm.ResNet101(criterion),
        mm.ResNet152(criterion),
    ):
        resnet_test(model, x, y)

    encoder = mm.ResNet18(criterion, encoder=True)
    optimiser = torch.optim.SGD(encoder.parameters(), lr=1.0e-3)

    encoder.set_optimiser(optimiser)

    encoder.determine_output_dim()
    print(encoder.output_shape)
    assert encoder.output_shape == (512, 8, 8)

    x = torch.rand(16, *(4, 256, 256))
    assert len(encoder(x)) is 5


def test_fcnresnets() -> None:
    def resnet_test(
        test_model: mm.MinervaModel, x: torch.FloatTensor, y: torch.LongTensor
    ) -> None:
        optimiser = torch.optim.SGD(test_model.parameters(), lr=1.0e-3)

        test_model.set_optimiser(optimiser)

        test_model.determine_output_dim()
        assert test_model.output_shape == (256, 256)

        loss, z = test_model.training_step(x, y)

        assert type(loss.item()) is float
        assert z.size() == (16, 8, 256, 256)

    x = torch.rand((16, 4, 256, 256))
    y = torch.randint(0, 8, (16, 256, 256))

    for model in (
        mm.FCN32ResNet18(criterion),
        mm.FCN32ResNet34(criterion),
        mm.FCN32ResNet50(criterion),
        mm.FCN16ResNet18(criterion),
        mm.FCN16ResNet34(criterion),
        mm.FCN16ResNet50(criterion),
        mm.FCN8ResNet18(criterion),
        mm.FCN8ResNet18(criterion),
        mm.FCN8ResNet34(criterion),
        mm.FCN8ResNet50(criterion),
        mm.FCN8ResNet101(criterion),
        mm.FCN8ResNet152(criterion),
    ):
        resnet_test(model, x, y)


def test_simclr() -> None:
    pass
