# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torch.nn.modules as nn
from nptyping import Float, Int, NDArray, Shape
from torch import LongTensor, Tensor
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.utils import BoundingBox

from minerva.datasets import make_dataset
from minerva.models import CNN, MLP, MinervaModel
from minerva.utils import CONFIG, utils


@pytest.fixture(scope="session", autouse=True)
def set_seeds():
    utils.set_seeds(42)


@pytest.fixture(scope="session", autouse=True)
def results_dir():
    path = Path(__file__).parent / "tmp" / "results"
    yield path
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def data_root():
    return Path(__file__).parent / "tmp" / "results"


@pytest.fixture
def img_root(data_root: Path):
    return data_root.parent / "data" / "test_images"


@pytest.fixture
def config_root(data_root: Path):
    config_path = data_root.parent / "config"

    # Make a temporary copy of a config manifest example
    os.makedirs(config_path, exist_ok=True)
    shutil.copy(
        Path(__file__).parent.parent / "inbuilt_cfgs" / "exp_mf_config.yml", config_path
    )
    yield config_path

    # Delete it afterwards
    shutil.rmtree(config_path)


@pytest.fixture
def config_here():
    here = Path(__file__).parent.parent

    # Make a temporary copy where we're running from
    shutil.copy(here / "inbuilt_cfgs" / "exp_mf_config.yml", here)

    yield here

    os.unlink(here / "exp_mf_config.yml")


@pytest.fixture
def x_entropy_loss():
    return nn.CrossEntropyLoss()


@pytest.fixture
def rgbi_input_size() -> Tuple[int, int, int]:
    return (4, 64, 64)


@pytest.fixture
def exp_mlp(x_entropy_loss) -> MinervaModel:
    return MLP(x_entropy_loss, 64)


@pytest.fixture
def exp_cnn(x_entropy_loss, rgbi_input_size) -> MinervaModel:
    return CNN(x_entropy_loss, rgbi_input_size)


@pytest.fixture
def random_mask() -> NDArray[Shape["32, 32"], Int]:
    return np.random.randint(0, 7, size=(32, 32))


@pytest.fixture
def random_image() -> NDArray[Shape["32, 32, 3"], Float]:
    return np.random.rand(32, 32, 3)


@pytest.fixture
def random_rgbi_image() -> NDArray[Shape["32, 32, 4"], Float]:
    return np.random.rand(32, 32, 4)


@pytest.fixture
def random_rgbi_tensor(rgbi_input_size) -> Tensor:
    return torch.rand(rgbi_input_size)


@pytest.fixture
def random_tensor_mask() -> LongTensor:
    mask = torch.randint(0, 7, size=(32, 32), dtype=torch.long)
    assert isinstance(mask, LongTensor)
    return mask


@pytest.fixture
def bounds_for_test_img() -> BoundingBox:
    return BoundingBox(
        -1.4153283567520825,
        -1.3964510733477618,
        50.91896360773007,
        50.93781998522083,
        1.0,
        2.0,
    )


@pytest.fixture
def exp_classes() -> Dict[int, str]:
    return {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}


@pytest.fixture
def simple_mask() -> LongTensor:
    mask: LongTensor = torch.tensor(  # type: ignore[attr-defined, assignment]
        [[1, 3, 5], [4, 5, 1], [1, 1, 1]], dtype=torch.long  # type: ignore[attr-defined]
    )
    return mask


@pytest.fixture
def example_matrix() -> Dict[int, int]:
    return {1: 1, 3: 3, 4: 2, 5: 0}


@pytest.fixture
def simple_bbox():
    return BoundingBox(0, 1, 0, 1, 0, 1)


@pytest.fixture
def default_dataset() -> GeoDataset:
    dataset, _ = make_dataset(CONFIG["dir"]["data"], CONFIG["dataset_params"]["test"])
    assert isinstance(dataset, GeoDataset)
    return dataset


@pytest.fixture(scope="session", autouse=True)
def wandb_offline():
    yield os.system("wandb offline")  # nosec B605, B607
    os.system("wandb online")  # nosec B605, B607
