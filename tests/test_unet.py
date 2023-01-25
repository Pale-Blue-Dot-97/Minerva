import torch
from torch import Tensor

from minerva.models import MinervaModel, UNet, UNetR18, UNetR34

input_size = (4, 64, 64)
batch_size = 2
n_classes = 8

x = torch.rand((batch_size, *input_size))
y = torch.randint(0, n_classes, (batch_size, *input_size[1:]))  # type: ignore[attr-defined]


def unet_test(test_model: MinervaModel, x: Tensor, y: Tensor) -> None:
    optimiser = torch.optim.SGD(test_model.parameters(), lr=1.0e-3)

    test_model.set_optimiser(optimiser)

    test_model.determine_output_dim()
    assert test_model.output_shape == input_size[1:]

    loss, z = test_model.step(x, y, True)

    assert type(loss.item()) is float
    assert isinstance(z, Tensor)
    assert z.size() == (batch_size, n_classes, *input_size[1:])


def test_unet(x_entropy_loss) -> None:
    model = UNet(x_entropy_loss, input_size=input_size)
    unet_test(model, x, y)

    bilinear_model = UNet(x_entropy_loss, input_size=input_size, bilinear=True)
    unet_test(bilinear_model, x, y)


def test_unetr18(x_entropy_loss) -> None:
    model = UNetR18(x_entropy_loss, input_size=input_size)
    unet_test(model, x, y)


def test_unetr34(x_entropy_loss) -> None:
    model = UNetR34(x_entropy_loss, input_size=input_size)
    unet_test(model, x, y)
