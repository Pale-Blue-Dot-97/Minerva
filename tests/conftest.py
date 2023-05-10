# -*- coding: utf-8 -*-
# Copyright (C) 2023 Harry Baker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
r""":mod:`pytest` fixtures for :mod:`minerva` CI/CD.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
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


# =====================================================================================================================
#                                                     FIXTURES
# =====================================================================================================================
@pytest.fixture(scope="session", autouse=True)
def set_seeds():
    utils.set_seeds(42)


@pytest.fixture(scope="session", autouse=True)
def results_dir():
    path = Path(__file__).parent / "tmp" / "results"
    if not path.exists():
        path.mkdir(parents=True)

    yield path
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def data_root() -> Path:
    return Path(__file__).parent / "tmp" / "results"


@pytest.fixture
def img_root(data_root: Path) -> Path:
    return data_root.parent / "data" / "test_images"


@pytest.fixture
def lc_root(data_root: Path) -> Path:
    return data_root.parent / "data" / "test_lc"


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
def default_device() -> torch.device:
    return utils.get_cuda_device()


@pytest.fixture
def std_batch_size() -> int:
    return 3


@pytest.fixture
def std_n_classes() -> int:
    return 8


@pytest.fixture
def std_n_batches() -> int:
    return 2


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
def random_rgbi_tensor(rgbi_input_size: Tuple[int, int, int]) -> Tensor:
    return torch.rand(rgbi_input_size)


@pytest.fixture
def random_rgbi_batch(
    rgbi_input_size: Tuple[int, int, int], std_batch_size: int
) -> Tensor:
    return torch.rand((std_batch_size, *rgbi_input_size))


@pytest.fixture
def random_tensor_mask(std_n_classes: int) -> LongTensor:
    mask = torch.randint(0, std_n_classes - 1, size=(32, 32), dtype=torch.long)
    assert isinstance(mask, LongTensor)
    return mask


@pytest.fixture
def random_mask_batch(
    std_batch_size: int, std_n_classes: int, rgbi_input_size: Tuple[int, int, int]
) -> LongTensor:
    mask = torch.randint(
        0,
        std_n_classes - 1,
        size=(std_batch_size, *rgbi_input_size[1:]),
        dtype=torch.long,
    )
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
    print(os.getcwd())
    dataset, _ = make_dataset(CONFIG["dir"]["data"], CONFIG["dataset_params"]["test"])
    assert isinstance(dataset, GeoDataset)
    return dataset


@pytest.fixture(scope="session", autouse=True)
def wandb_offline():
    yield os.system("wandb offline")  # nosec B605, B607
    os.system("wandb online")  # nosec B605, B607
