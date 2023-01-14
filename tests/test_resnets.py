import numpy as np
import pytest
import torch
from torch import Tensor, LongTensor
from torchvision.models.resnet import BasicBlock

from minerva.models import (
    MinervaModel,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)
from minerva.models.resnet import ResNet, _preload_weights

input_size = (4, 64, 64)

x = torch.rand(6, *input_size)
y = LongTensor(np.random.randint(0, 8, size=6))


def resnet_test(test_model: MinervaModel, x: Tensor, y: Tensor) -> None:
    optimiser = torch.optim.SGD(test_model.parameters(), lr=1.0e-3)

    test_model.set_optimiser(optimiser)

    test_model.determine_output_dim()
    assert test_model.output_shape is test_model.n_classes

    loss, z = test_model.step(x, y, True)

    assert type(loss.item()) is float
    assert isinstance(z, Tensor)
    assert z.size() == (6, 8)


def test_resnet():
    assert isinstance(ResNet(BasicBlock, [2, 2, 2, 2], groups=2), ResNet)


def test_resnet18(x_entropy_loss) -> None:
    with pytest.raises(ValueError):
        _ = ResNet18(replace_stride_with_dilation=(True, False))  # type: ignore[arg-type]

    for zero_init_residual in (True, False):

        resnet18 = ResNet18(
            x_entropy_loss, input_size=input_size, zero_init_residual=zero_init_residual
        )

        resnet_test(resnet18, x, y)


def test_resnet34(x_entropy_loss) -> None:
    model = ResNet34(x_entropy_loss, input_size=input_size)
    resnet_test(model, x, y)


def test_resnet50(x_entropy_loss) -> None:
    for model in (
        ResNet50(x_entropy_loss, input_size=input_size),
        ResNet50(
            x_entropy_loss,
            input_size=input_size,
            replace_stride_with_dilation=(True, True, False),
            zero_init_residual=True,
        ),
    ):
        resnet_test(model, x, y)


def test_resnet101(x_entropy_loss) -> None:
    model = ResNet101(x_entropy_loss, input_size=input_size)
    resnet_test(model, x, y)


def test_resnet152(x_entropy_loss) -> None:
    model = ResNet152(x_entropy_loss, input_size=input_size)
    resnet_test(model, x, y)


def test_resnet_encoder(x_entropy_loss) -> None:
    encoder = ResNet18(x_entropy_loss, input_size=input_size, encoder=True)
    optimiser = torch.optim.SGD(encoder.parameters(), lr=1.0e-3)

    encoder.set_optimiser(optimiser)

    encoder.determine_output_dim()
    print(encoder.output_shape)
    assert encoder.output_shape == (512, 2, 2)

    x = torch.rand(6, *input_size)
    assert len(encoder(x)) == 5


def test_preload_weights():
    resnet = ResNet(BasicBlock, [2, 2, 2, 2])
    new_resnet = _preload_weights(resnet, None, (4, 32, 32), encoder_on=False)

    assert resnet == new_resnet
