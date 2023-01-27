import pytest
import torch
from torch import Tensor

from minerva.models import (
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
    MinervaModel,
    ResNet18,
)
from minerva.models.fcn import DCN

input_size = (4, 64, 64)
batch_size = 2
n_classes = 8

x = torch.rand((batch_size, *input_size))
y = torch.randint(0, n_classes, (batch_size, *input_size[1:]))  # type: ignore[attr-defined]


def fcn_test(test_model: MinervaModel, x: Tensor, y: Tensor) -> None:
    optimiser = torch.optim.SGD(test_model.parameters(), lr=1.0e-3)

    test_model.set_optimiser(optimiser)

    test_model.determine_output_dim()
    assert test_model.output_shape == input_size[1:]

    loss, z = test_model.step(x, y, True)

    assert type(loss.item()) is float
    assert isinstance(z, Tensor)
    assert z.size() == (batch_size, n_classes, *input_size[1:])


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
    for _model in (
        FCN8ResNet18,
        FCN16ResNet34,
        FCN32ResNet50,
        FCN8ResNet101,
        FCN8ResNet152,
    ):
        try:
            model = _model(x_entropy_loss, input_size=input_size, torch_weights=True)
            fcn_test(model, x, y)
        except ImportError as err:
            print(err)


def test_dcn() -> None:
    with pytest.raises(
        NotImplementedError, match="Variant 42 does not match known types"
    ):
        _ = DCN(variant="42")  # type: ignore[arg-type]

    dcn = DCN(variant="32")
    resnet = ResNet18()
    with pytest.raises(
        NotImplementedError, match="Variant 42 does not match known types"
    ):
        dcn.variant = "42"  # type: ignore[assignment]
        _ = dcn.forward(resnet(torch.rand((batch_size, *input_size))))
