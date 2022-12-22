from typing import Tuple
import pytest
from pathlib import Path
import os
import shutil
import torch.nn.modules as nn

from minerva.models import MinervaModel, MLP, CNN


@pytest.fixture
def data_root():
    return Path(__file__).parent / "tests" / "tmp"


@pytest.fixture
def img_root(data_root):
    return data_root / "data" / "test_images"


@pytest.fixture
def config_root(data_root):
    config_path = data_root / "config"

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
