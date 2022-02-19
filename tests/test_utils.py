from cProfile import label
from Minerva.utils import utils, config, aux_configs
import numpy as np


def test_config_loading():
    assert type(config) is dict
    assert type(aux_configs) is dict


def test_datetime_reformat():
    dt = '2018-12-15'
    assert utils.datetime_reformat(dt, '%Y-%m-%d', '%d.%m.%Y') == '15.12.2018'


def test_ohe_labels():
    labels = [3, 2, 4, 5, 6]
    n_classes = 10
    target = np.array([np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
                       np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
                       np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                       np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                       np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])])
    assert utils.labels_to_ohe(labels=labels, n_classes=n_classes) == target


def test_empty_classes():
    labels = [(3, 321), (4, 112), (1, 671), (5, 456)]
    classes = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
    assert utils.find_empty_classes(labels, classes) == [0, 2]


def test_eliminate_classes():
    empty = [0, 2]
    old_classes = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
    old_cmap = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
    new_classes = {0: '5', 1: '1', 2: '4', 3: '3'}
    new_cmap = {0: '5', 1: '1', 2: '4', 3: '3'}
    conversion = {0: 5, 1: 1, 2: 4, 3: 3}

    results = utils.eliminate_classes(empty_classes=empty, old_classes=old_classes, old_cmap=old_cmap)
    
    assert new_classes == results[0]
    assert conversion == results[1]
    assert new_cmap == results[2]
