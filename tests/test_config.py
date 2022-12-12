from minerva.utils import runner, AUX_CONFIGS, CONFIG, utils, visutils


def test_config_path(config_root):
    with open(config_root / "exp_mf_config.yml", "r") as config:
        print(config.read())
