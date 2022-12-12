import os
from minerva.utils import load_configs


def test_config_path(config_root, config_here):
    assert "tmp/config" in str(config_root)

    # Still works because we are relative to inbuilt_cfgs here
    base, aux = load_configs(config_root / "exp_mf_config.yml")
    assert base
    assert aux

    base, aux = load_configs(config_here / "exp_mf_config.yml")
    assert base
    assert aux


