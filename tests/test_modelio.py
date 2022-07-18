import torch
from numpy.testing import assert_array_equal
from lightly.loss import NTXentLoss
from torchgeo.datasets.utils import BoundingBox

from minerva.modelio import ssl_pair_tg, sup_tg
from minerva.models import FCN32ResNet18, Siam34

input_size = (4, 224, 224)
device = torch.device("cpu")


def test_sup_tg() -> None:
    criterion = torch.nn.CrossEntropyLoss()
    model = FCN32ResNet18(criterion, input_size=input_size)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val", "test"):
        images = torch.rand(size=(6, *input_size))
        masks = torch.randint(0, 8, (6, *input_size[1:]))
        bboxes = [BoundingBox(0, 1, 0, 1, 0, 1)] * 6
        batch = {
            "image": images,
            "mask": masks,
            "bbox": bboxes,
        }

        results = sup_tg(batch, model, device, mode)

        assert type(results[0]) is torch.Tensor
        assert results[1].size() == (6, 8, *input_size[1:])
        assert assert_array_equal(results[2], batch["mask"]) is None
        assert results[3] == batch["bbox"]


def test_ssl_pair_tg() -> None:
    criterion = NTXentLoss(0.5)
    model = Siam34(criterion, input_size=input_size)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val"):
        images_1 = torch.rand(size=(6, *input_size))
        bboxes_1 = [BoundingBox(0, 1, 0, 1, 0, 1)] * 6

        batch_1 = {
            "image": images_1,
            "bbox": bboxes_1,
        }

        images_2 = torch.rand(size=(6, *input_size))
        bboxes_2 = [BoundingBox(0, 1, 0, 1, 0, 1)] * 6

        batch_2 = {
            "image": images_2,
            "bbox": bboxes_2,
        }

        results = ssl_pair_tg((batch_1, batch_2), model, device, mode)

        assert type(results[0]) is torch.Tensor
        assert results[1].size() == (12, 128)
        assert results[2] is None
        assert results[3] == batch_1["bbox"] + batch_2["bbox"]
