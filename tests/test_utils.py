from pyparsing import dictOf
from Minerva.utils import utils, config, aux_configs


def test_config_loading():
    assert type(config) is dict
    assert type(aux_configs) is dict


def test_datetime_reformat():
    dt = '2018-12-15'
    assert utils.datetime_reformat(dt, '%Y-%m-%d', '%d.%m.%Y') == '15.12.2018'
