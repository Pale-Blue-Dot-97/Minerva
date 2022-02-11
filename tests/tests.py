from Minerva.utils import utils
import os
import shutil

config_dir_path = 'config'


def test_config_loading():
    config_exist = False
    if os.path.exists(f'{config_dir_path}/config.yml'):
        config_exist = True

    if not config_exist:
        shutil.copy(f'{config_dir_path}/exp_config.yml', f'{config_dir_path}/config.yml')

    master_config, aux_configs = utils.load_configs(f'{config_dir_path}/config.yml')

    if not config_exist:
        os.remove(f'{config_dir_path}/config.yml')

    assert master_config is dict
    assert aux_configs is dict


def test_cuda_device():
    assert utils.get_cuda_device() in ('cuda:0', 'cpu')
