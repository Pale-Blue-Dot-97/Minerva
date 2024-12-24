# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2024 Harry Baker

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
r""":mod:`pytest` fixtures for :mod:`minerva` CI/CD."""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Any, Generator

import hydra
import numpy as np
import pytest
import torch
import torch.nn.modules as nn
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rasterio.crs import CRS
from torch import LongTensor, Tensor
from torchgeo.datasets import IntersectionDataset, RasterDataset
from torchgeo.datasets.utils import BoundingBox

from minerva.datasets import GeoSSL4EOS12Sentinel2, make_dataset
from minerva.loss import SegBarlowTwinsLoss
from minerva.models import CNN, MLP, FCN32ResNet18, SimConvPSP
from minerva.utils import DEFAULT_CONFIG_NAME, utils
from minerva.utils.runner import _config_load_resolver, _construct_patch_size


# =====================================================================================================================
#                                                      METHODS
# =====================================================================================================================
def pytest_addoption(parser):
    """Adds a custom CLI option to activate :func:`torch.autograd.set_detect_anomaly`"""
    parser.addoption("--detect-anomaly", action="store_const", const=True)


# =====================================================================================================================
#                                                     FIXTURES
# =====================================================================================================================
@pytest.fixture(scope="session", autouse=True)
def set_seeds() -> None:
    utils.set_seeds(42)


@pytest.fixture(scope="session", autouse=True)
def set_multiprocessing_to_fork():
    # Workaround for pickling issues from multiprocessing on Mac OS.
    try:
        multiprocessing.set_start_method("fork")
    except ValueError:
        # Raises ValueError on Windows so just bypass this.
        pass


@pytest.fixture(scope="session", autouse=True)
def use_detect_anomaly(request):
    if request.config.getoption("--detect-anomaly"):
        # Activates PyTorch's anomaly detection.
        yield torch.autograd.set_detect_anomaly(True, True)  # type: ignore[attr-defined]

        # Deactivate anomaly detection after tests.
        torch.autograd.set_detect_anomaly(False)  # type: ignore[attr-defined]
    else:
        yield torch.autograd.set_detect_anomaly(False)  # type: ignore[attr-defined]


@pytest.fixture(scope="session", autouse=True)
def results_dir() -> Generator[Path, None, None]:
    path = Path(__file__).parent / "tmp" / "results"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    yield path
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def cache_dir() -> Generator[Path, None, None]:
    path = Path(__file__).parent / "tmp" / "cache"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    yield path
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def results_root() -> Path:
    return Path(__file__).parent / "tmp" / "results"


@pytest.fixture
def data_root() -> Path:
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture
def img_root(data_root: Path) -> Path:
    return data_root / "NAIP"


@pytest.fixture
def lc_root(data_root: Path) -> Path:
    return data_root / "Chesapeake7"


@pytest.fixture
def config_root(
    inbuilt_cfg_root: Path, results_root: Path
) -> Generator[Path, None, None]:
    config_path = results_root.parent / "config"

    # Make a temporary copy of a config manifest example
    os.makedirs(config_path, exist_ok=True)
    shutil.copy(inbuilt_cfg_root / "exp_mf_config.yml", config_path)
    yield config_path

    # Delete it afterwards
    shutil.rmtree(config_path)


@pytest.fixture
def config_here(inbuilt_cfg_root: Path) -> Generator[Path, None, None]:
    here = Path(__file__).parent.parent

    # Make a temporary copy where we're running from
    shutil.copy(inbuilt_cfg_root / "exp_mf_config.yml", here)

    yield here

    os.unlink(here / "exp_mf_config.yml")


@pytest.fixture
def default_config(inbuilt_cfg_root: Path) -> DictConfig:
    OmegaConf.register_new_resolver("cfg_load", _config_load_resolver, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "to_patch_size", _construct_patch_size, replace=True
    )

    with hydra.initialize(version_base="1.3", config_path=str(inbuilt_cfg_root)):
        # config is relative to a module
        cfg = hydra.compose(config_name=DEFAULT_CONFIG_NAME)
        return cfg


@pytest.fixture
def inbuilt_cfg_root() -> Path:
    return Path("..", "minerva", "inbuilt_cfgs")


@pytest.fixture
def default_device() -> torch.device:
    return utils.get_cuda_device()


@pytest.fixture
def std_batch_size() -> int:
    return 3


@pytest.fixture
def std_n_classes(exp_classes: dict[int, str]) -> int:
    return len(exp_classes)


@pytest.fixture
def std_n_batches() -> int:
    return 2


@pytest.fixture
def x_entropy_loss() -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss()


@pytest.fixture
def small_patch_size() -> tuple[int, int]:
    return (32, 32)


@pytest.fixture
def rgbi_input_size() -> tuple[int, int, int]:
    return (4, 32, 32)


@pytest.fixture
def exp_mlp(x_entropy_loss: nn.CrossEntropyLoss) -> MLP:
    return MLP(x_entropy_loss, 64)


@pytest.fixture
def exp_cnn(
    x_entropy_loss: nn.CrossEntropyLoss, rgbi_input_size: tuple[int, int, int]
) -> CNN:
    return CNN(x_entropy_loss, rgbi_input_size)


@pytest.fixture
def exp_fcn(
    x_entropy_loss: nn.CrossEntropyLoss,
    rgbi_input_size: tuple[int, int, int],
    std_n_classes: int,
) -> FCN32ResNet18:
    return FCN32ResNet18(x_entropy_loss, rgbi_input_size, std_n_classes)


@pytest.fixture
def exp_simconv(
    rgbi_input_size: tuple[int, int, int],
) -> SimConvPSP:
    return SimConvPSP(SegBarlowTwinsLoss(), rgbi_input_size, feature_dim=128)


@pytest.fixture
def random_mask(
    small_patch_size: tuple[int, int], std_n_classes: int
) -> NDArray[np.int_]:
    mask = np.random.randint(0, std_n_classes - 1, size=small_patch_size)
    assert isinstance(mask, np.ndarray)
    return mask


@pytest.fixture
def random_image(
    small_patch_size: tuple[int, int],
) -> NDArray[np.float64]:
    return np.random.rand(*small_patch_size, 3)


@pytest.fixture
def random_rgbi_image(
    small_patch_size: tuple[int, int],
) -> NDArray[np.float64]:
    return np.random.rand(*small_patch_size, 4)


@pytest.fixture
def random_rgbi_tensor(rgbi_input_size: tuple[int, int, int]) -> Tensor:
    return torch.rand(rgbi_input_size)


@pytest.fixture
def simple_rgb_img() -> Tensor:
    img: Tensor = torch.tensor(  # type: ignore[attr-defined]
        [[255.0, 0.0, 127.5], [102.0, 127.5, 76.5], [178.5, 255.0, 204.0]]
    )
    assert isinstance(img, Tensor)
    return img


@pytest.fixture
def norm_simple_rgb_img(simple_rgb_img: Tensor) -> Tensor:
    norm_img = simple_rgb_img / 255
    assert isinstance(norm_img, Tensor)
    return norm_img


@pytest.fixture
def flipped_rgb_img() -> Tensor:
    img: Tensor = torch.tensor(  # type: ignore[attr-defined]
        [[127.5, 0.0, 255.0], [76.5, 127.5, 102.0], [204.0, 255.0, 178.5]]
    )
    assert isinstance(img, Tensor)
    return img


@pytest.fixture
def simple_sample(simple_rgb_img: Tensor, simple_mask: LongTensor) -> dict[str, Tensor]:
    return {"image": simple_rgb_img, "mask": simple_mask}


@pytest.fixture
def flipped_simple_sample(
    flipped_rgb_img: Tensor, flipped_simple_mask: LongTensor
) -> dict[str, Tensor]:
    return {"image": flipped_rgb_img, "mask": flipped_simple_mask}


@pytest.fixture
def random_rgbi_batch(
    rgbi_input_size: tuple[int, int, int], std_batch_size: int
) -> Tensor:
    return torch.rand((std_batch_size, *rgbi_input_size))


@pytest.fixture
def random_tensor_mask(
    std_n_classes: int, small_patch_size: tuple[int, int]
) -> LongTensor:
    mask = torch.randint(0, std_n_classes - 1, size=small_patch_size, dtype=torch.long)
    assert isinstance(mask, LongTensor)
    return mask


@pytest.fixture
def random_mask_batch(
    std_batch_size: int, std_n_classes: int, rgbi_input_size: tuple[int, int, int]
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
def random_scene_classification_batch(
    std_batch_size: int, std_n_classes: int
) -> LongTensor:
    batch = torch.randint(0, std_n_classes - 1, size=(std_batch_size,))
    assert isinstance(batch, LongTensor)
    return batch


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
def exp_classes() -> dict[int, str]:
    return {
        0: "No Data",
        1: "Water",
        2: "Trees/Shrub",
        3: "Low Vegetation",
        4: "Barren",
        5: "Surfaces",
        6: "Roads",
        7: "Military\nBase",
    }


@pytest.fixture
def exp_cmap_dict() -> dict[int, str]:
    return {
        0: "#000000",  # Transparent
        1: "#00c5ff",  # Light Blue
        2: "#267300",  # Dark green
        3: "#a3ff73",  # v. Light green
        4: "#ffaa00",  # Orange
        5: "#9c9c9c",  # Light grey
        6: "#000000",  # Black
        7: "#c500ff",  # Purple
    }


@pytest.fixture
def simple_mask() -> LongTensor:
    mask: LongTensor = torch.tensor(  # type: ignore[attr-defined, assignment]
        [[1, 3, 5], [4, 5, 1], [1, 1, 1]],
        dtype=torch.long,  # type: ignore[attr-defined]
    )
    return mask


@pytest.fixture
def flipped_simple_mask() -> LongTensor:
    mask: LongTensor = torch.tensor([[5, 3, 1], [1, 5, 4], [1, 1, 1]], dtype=torch.long)  # type: ignore[assignment]
    assert isinstance(mask, LongTensor)
    return mask


@pytest.fixture
def example_matrix() -> dict[int, int]:
    return {1: 1, 3: 3, 4: 2, 5: 0}


@pytest.fixture
def simple_bbox() -> BoundingBox:
    return BoundingBox(0, 1, 0, 1, 0, 1)


@pytest.fixture
def exp_dataset_params() -> dict[str, Any]:
    return {
        "image": {
            "transforms": {"AutoNorm": {"length": 12}},
            "_target_": "minerva.datasets.__testing.TstImgDataset",
            "paths": "NAIP",
            "res": 1.0,
            "crs": 26918,
        },
        "mask": {
            "transforms": False,
            "_target_": "minerva.datasets.__testing.TstMaskDataset",
            "paths": "Chesapeake7",
            "res": 1.0,
        },
    }


@pytest.fixture
def exp_sampler_params(small_patch_size: tuple[int, int]):
    return {
        "_target_": "torchgeo.samplers.RandomGeoSampler",
        "roi": False,
        "size": small_patch_size,
        "length": 120,
    }


@pytest.fixture
def exp_loader_params(default_config):
    return OmegaConf.to_object(default_config)["loader_params"]  # type: ignore[call-overload, index]


@pytest.fixture
def exp_collator_params(default_config):
    return OmegaConf.to_object(default_config)["collator"]  # type: ignore[call-overload, index]


@pytest.fixture
def default_dataset(
    default_config: DictConfig, data_root: Path, cache_dir: Path
) -> IntersectionDataset:
    dataset, _ = make_dataset(
        data_root,
        OmegaConf.to_object(default_config["tasks"]["test-test"]["dataset_params"]),  # type: ignore[arg-type]
        cache_dir=cache_dir,
    )
    assert isinstance(dataset, IntersectionDataset)
    return dataset


@pytest.fixture
def default_image_dataset(
    default_config: DictConfig, exp_dataset_params: dict[str, Any]
) -> RasterDataset:
    del exp_dataset_params["mask"]
    dataset, _ = make_dataset(
        default_config["data_root"], exp_dataset_params, cache=False
    )
    assert isinstance(dataset, RasterDataset)
    return dataset


@pytest.fixture
def ssl4eo_s12_dataset(data_root: Path, epsg3857: CRS) -> RasterDataset:
    return GeoSSL4EOS12Sentinel2(
        str(data_root / "SSL4EO-S12"), epsg3857, 10.0, bands=["B2", "B3", "B4", "B8"]
    )


@pytest.fixture(scope="session", autouse=True)
def wandb_offline() -> Generator[int, None, None]:
    yield os.system("wandb offline")  # nosec B605, B607
    os.system("wandb online")  # nosec B605, B607


@pytest.fixture
def epsg3857() -> CRS:
    return CRS.from_epsg("3857")
