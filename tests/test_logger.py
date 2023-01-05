from typing import Any, Dict, List, Union
import shutil
from pathlib import Path
import tempfile
from nptyping import NDArray, Shape

import numpy as np
import torch
from torch import Tensor
import torch.nn.modules as nn
from numpy.testing import assert_array_equal
from lightly.loss import NTXentLoss
from torch.utils.tensorboard.writer import SummaryWriter
from torchgeo.datasets.utils import BoundingBox

from minerva.logger import SSL_Logger, STG_Logger
from minerva.modelio import ssl_pair_tg, sup_tg
from minerva.models import FCN16ResNet18, SimCLR18

device = torch.device("cpu")  # type: ignore[attr-defined]


def test_STG_Logger(simple_bbox):
    criterion = nn.CrossEntropyLoss()

    path = Path(tempfile.gettempdir(), "exp1")

    if not path.exists():
        path.mkdir()

    writer = SummaryWriter(log_dir=path)

    model = FCN16ResNet18(criterion)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)
    model.determine_output_dim()

    n_batches = 8

    output_shape = model.output_shape
    assert isinstance(output_shape, tuple)

    for mode in ("train", "val", "test"):
        logger = STG_Logger(
            n_batches=n_batches,
            batch_size=6,
            n_samples=8 * 6 * 256 * 256,
            out_shape=output_shape,
            n_classes=8,
            record_int=True,
            record_float=True,
        )
        data: List[Dict[str, Union[Tensor, List[Any]]]] = []
        for i in range(n_batches):
            images = torch.rand(size=(6, 4, 256, 256))
            masks = torch.randint(0, 8, (6, 256, 256))  # type: ignore[attr-defined]
            bboxes = [simple_bbox] * 6
            batch: Dict[str, Union[Tensor, List[Any]]] = {
                "image": images,
                "mask": masks,
                "bbox": bboxes,
            }
            data.append(batch)

            logger(mode, i, writer, *sup_tg(batch, model, device=device, mode=mode))

        logs = logger.get_logs
        assert logs["batch_num"] == 8
        assert type(logs["total_loss"]) is float
        assert type(logs["total_correct"]) is float

        results = logger.get_results
        assert results["z"].shape == (8, 6, 256, 256)
        assert results["y"].shape == (8, 6, 256, 256)
        assert np.array(results["ids"]).shape == (8, 6)

        y: NDArray[Shape["8, 6, 256, 256"], Any] = np.empty(
            (n_batches, 6, *output_shape), dtype=np.uint8
        )
        for i in range(n_batches):
            mask: Union[Tensor, List[Any]] = data[i]["mask"]
            assert isinstance(mask, Tensor)
            y[i] = mask.cpu().numpy()

        assert_array_equal(results["y"], y)

    shutil.rmtree(path, ignore_errors=True)


def test_SSL_Logger(simple_bbox):
    criterion = NTXentLoss(0.5)

    path = Path(tempfile.gettempdir(), "exp2")

    if not path.exists():
        path.mkdir()

    writer = SummaryWriter(log_dir=path)

    model = SimCLR18(criterion)
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    n_batches = 8

    for mode in ("train", "val", "test"):
        logger = SSL_Logger(
            n_batches=n_batches,
            batch_size=6,
            n_samples=8 * 6,
            record_int=True,
            record_float=True,
        )
        data = []
        for i in range(n_batches):
            images = torch.rand(size=(6, 4, 256, 256))
            bboxes = [simple_bbox] * 6
            batch = {
                "image": images,
                "bbox": bboxes,
            }
            data.append((batch, batch))

            logger(
                mode,
                i,
                writer,
                *ssl_pair_tg((batch, batch), model, device=device, mode=mode)
            )

        logs = logger.get_logs
        assert logs["batch_num"] == 8
        assert type(logs["total_loss"]) is float
        assert type(logs["total_correct"]) is float
        assert type(logs["total_top5"]) is float

        results = logger.get_results
        assert results == {}

    shutil.rmtree(path, ignore_errors=True)
