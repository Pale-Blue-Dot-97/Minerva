import minerva.models as mm
import torch
import numpy as np


def test_mlp() -> None:
    criterion = torch.nn.CrossEntropyLoss()
    model = mm.MLP(criterion)

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
