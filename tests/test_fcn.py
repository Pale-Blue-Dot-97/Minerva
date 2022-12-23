import torch
from torch import Tensor

from minerva.models import (
    MinervaModel,
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


input_size = (4, 64, 64)

x = torch.rand((6, *input_size))
y = torch.randint(0, 8, (6, 64, 64))  # type: ignore[attr-defined]


def fcn_test(test_model: MinervaModel, x: Tensor, y: Tensor) -> None:
    optimiser = torch.optim.SGD(test_model.parameters(), lr=1.0e-3)

    test_model.set_optimiser(optimiser)

    test_model.determine_output_dim()
    assert test_model.output_shape == (64, 64)

    loss, z = test_model.step(x, y, True)

    assert type(loss.item()) is float
    assert isinstance(z, Tensor)
    assert z.size() == (6, 8, 64, 64)


def test_fcn32resnet18(x_entropy_loss) -> None:
    model = FCN32ResNet18(x_entropy_loss, input_size=input_size)
    fcn_test(model, x, y)


def test_fcn32resnet34(x_entropy_loss) -> None:
    model = FCN32ResNet34(x_entropy_loss, input_size=input_size)
    fcn_test(model, x, y)


def test_fcn32resnet50(x_entropy_loss) -> None:
    model = FCN32ResNet50(x_entropy_loss, input_size=input_size)
    fcn_test(model, x, y)


def test_fcn16resnet18(x_entropy_loss) -> None:
    model = FCN16ResNet18(x_entropy_loss, input_size=input_size)
    fcn_test(model, x, y)


def test_fcn16resnet34(x_entropy_loss) -> None:
    model = FCN16ResNet34(x_entropy_loss, input_size=input_size)
    fcn_test(model, x, y)


def test_fcn16resnet50(x_entropy_loss) -> None:
    model = FCN16ResNet50(x_entropy_loss, input_size=input_size)
    fcn_test(model, x, y)


def test_fcn8resnet18(x_entropy_loss) -> None:
    model = FCN8ResNet18(x_entropy_loss, input_size=input_size)
    fcn_test(model, x, y)


def test_fcn8resnet34(x_entropy_loss) -> None:
    model = FCN8ResNet34(x_entropy_loss, input_size=input_size)
    fcn_test(model, x, y)


def test_fcn8resnet50(x_entropy_loss) -> None:
    model = FCN8ResNet50(x_entropy_loss, input_size=input_size)
    fcn_test(model, x, y)


def test_fcn8resnet101(x_entropy_loss) -> None:
    model = FCN8ResNet101(x_entropy_loss, input_size=input_size)
    fcn_test(model, x, y)


def test_fcn8resnet152(x_entropy_loss) -> None:
    model = FCN8ResNet152(x_entropy_loss, input_size=input_size)
    fcn_test(model, x, y)


def test_fcnresnet_torch_weights(x_entropy_loss) -> None:
    model = FCN8ResNet18(x_entropy_loss, input_size=input_size, torch_weights=True)
    fcn_test(model, x, y)
