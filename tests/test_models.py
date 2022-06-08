import minerva.models as mm
import torch
import numpy as np
import pytest


def test_mlp() -> None:
    criterion = torch.nn.CrossEntropyLoss()
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
    criterion = torch.nn.CrossEntropyLoss()
    model = mm.CNN(criterion)

    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    model.set_optimiser(optimiser)

    model.determine_output_dim()
    assert model.output_shape is model.n_classes

    x = torch.rand(16, *(4, 256, 256))
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


def test_resnet18() -> None:
    with pytest.raises(ValueError):
        _ = mm.ResNet18(replace_stride_with_dilation=(True, False))

    for zero_init_residual in (True, False):
        criterion = torch.nn.CrossEntropyLoss()
        model = mm.ResNet18(
            criterion, input_size=(4, 224, 224), zero_init_residual=zero_init_residual
        )

        optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)

        model.set_optimiser(optimiser)

        model.determine_output_dim()
        assert model.output_shape is model.n_classes

        x = torch.rand(16, *(4, 224, 224))
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
