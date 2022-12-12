import pytest
from pathlib import Path
import os
import shutil


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
