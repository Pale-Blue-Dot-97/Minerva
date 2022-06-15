import pytest
import torch

from minerva.models import CNN
from minerva.optimisers import LARS


def test_lars() -> None:
    model = CNN(torch.nn.CrossEntropyLoss(), input_size=(3, 224, 224))

    with pytest.raises(ValueError, match="Invalid learning rate: -0.1"):
        _ = LARS(model.parameters(), lr=-0.1)

    with pytest.raises(ValueError, match="Invalid momentum value: -0.1"):
        _ = LARS(model.parameters(), lr=1.0, momentum=-0.1)

    with pytest.raises(ValueError, match="Invalid weight_decay value: -0.01"):
        _ = LARS(model.parameters(), lr=0.1, weight_decay=-0.01)

    with pytest.raises(ValueError, match="Invalid LARS coefficient value: -0.02"):
        _ = LARS(model.parameters(), lr=1.0, eta=-0.02)

    model.set_optimiser(LARS(model.parameters(), lr=1.0e-3))

    x = torch.rand(60, 6, 3, 224, 224)
    y = torch.randint(0, 8, size=(60, 6))

    for mode in (True, False):
        for i in range(60):
            loss, z = model.step(x[i], y[i], train=mode)
