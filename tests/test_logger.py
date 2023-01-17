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

from minerva.logger import SSL_Logger, STG_Logger
from minerva.modelio import ssl_pair_tg, sup_tg
from minerva.models import FCN16ResNet18, SimCLR18

device = torch.device("cpu")  # type: ignore[attr-defined]
n_batches = 2
batch_size = 3
patch_size = (32, 32)
n_classes = 8


def test_STG_Logger(simple_bbox):
    criterion = nn.CrossEntropyLoss()

    path = Path(tempfile.gettempdir(), "exp1")

    if not path.exists():
        path.mkdir()

    writer = SummaryWriter(log_dir=path)

    model = FCN16ResNet18(criterion, input_size=(4, *patch_size))
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)
    model.determine_output_dim()

    output_shape = model.output_shape
    assert isinstance(output_shape, tuple)

    for mode in ("train", "val", "test"):
        for model_type in ("scene_classifier", "segmentation"):
            logger = STG_Logger(
                n_batches=n_batches,
                batch_size=batch_size,
                n_samples=n_batches * batch_size * patch_size[0] * patch_size[1],
                out_shape=output_shape,
                n_classes=n_classes,
                record_int=True,
                record_float=True,
                model_type=model_type,
            )
            data: List[Dict[str, Union[Tensor, List[Any]]]] = []
            for i in range(n_batches):
                images = torch.rand(size=(batch_size, 4, *patch_size))
                masks = torch.randint(0, n_classes, (batch_size, *patch_size))  # type: ignore[attr-defined]
                bboxes = [simple_bbox] * batch_size
                batch: Dict[str, Union[Tensor, List[Any]]] = {
                    "image": images,
                    "mask": masks,
                    "bbox": bboxes,
                }
                data.append(batch)

                logger(mode, i, writer, *sup_tg(batch, model, device=device, mode=mode))

            logs = logger.get_logs
            assert logs["batch_num"] == n_batches
            assert type(logs["total_loss"]) is float
            assert type(logs["total_correct"]) is float

            if model_type == "segmentation":
                assert type(logs["total_miou"]) is float

            results = logger.get_results
            assert results["z"].shape == (n_batches, batch_size, *patch_size)
            assert results["y"].shape == (n_batches, batch_size, *patch_size)
            assert np.array(results["ids"]).shape == (n_batches, batch_size)

            shape = f"{n_batches}, {batch_size}, {patch_size[0]}, {patch_size[1]}"
            y: NDArray[Shape[shape], Any] = np.empty(
                (n_batches, batch_size, *output_shape), dtype=np.uint8
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

    model = SimCLR18(criterion, input_size=(4, *patch_size))
    optimiser = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model.set_optimiser(optimiser)

    for mode in ("train", "val", "test"):
        for extra_metrics in (True, False):
            logger = SSL_Logger(
                n_batches=n_batches,
                batch_size=batch_size,
                n_samples=n_batches * batch_size,
                record_int=True,
                record_float=True,
                collapse_level=extra_metrics,
                euclidean=extra_metrics,
            )
            data = []
            for i in range(n_batches):
                images = torch.rand(size=(batch_size, 4, *patch_size))
                bboxes = [simple_bbox] * batch_size
                batch = {
                    "image": images,
                    "bbox": bboxes,
                }
                data.append((batch, batch))

                logger(
                    mode,
                    i,
                    writer,
                    *ssl_pair_tg((batch, batch), model, device=device, mode=mode),
                )

            logs = logger.get_logs
            assert logs["batch_num"] == n_batches
            assert type(logs["total_loss"]) is float
            assert type(logs["total_correct"]) is float
            assert type(logs["total_top5"]) is float

            if extra_metrics:
                assert type(logs["collapse_level"]) is float
                assert type(logs["euc_dist"]) is float

            results = logger.get_results
            assert results == {}

    shutil.rmtree(path, ignore_errors=True)
