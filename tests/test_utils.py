import os
from Minerva.utils import utils, config, aux_configs
import numpy as np
from numpy.testing import assert_array_equal


def test_config_loading():
    assert type(config) is dict
    assert type(aux_configs) is dict


def test_datetime_reformat():
    dt = '2018-12-15'
    assert utils.datetime_reformat(dt, '%Y-%m-%d', '%d.%m.%Y') == '15.12.2018'


def test_ohe_labels():
    labels = [3, 2, 4, 1, 0]
    correct_targets = np.array([[0., 0., 0., 1., 0., 0.],
                                [0., 0., 1., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0.],
                                [0., 1., 0., 0., 0., 0.],
                                [1., 0., 0., 0., 0., 0.]])
    targets = utils.labels_to_ohe(labels=labels, n_classes=6)
    assert assert_array_equal(correct_targets, targets) is None


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
    conversion = {1: 1, 3: 3, 4: 2, 5: 0}

    results = utils.eliminate_classes(empty_classes=empty, old_classes=old_classes, old_cmap=old_cmap)
    
    assert new_classes == results[0]
    assert conversion == results[1]
    assert new_cmap == results[2]


def test_file_check():
    fn = "tests/test.txt"
    with open(fn, "x") as f:
        f.write("")
    
    utils.exist_delete_check(fn)

    assert os.path.exists(fn) is False


def test_class_transform():
    matrix = {1: 1, 3: 3, 4: 2, 5: 0}

    assert utils.class_transform(1, matrix) == 1
    assert utils.class_transform(5, matrix) == 0
    assert utils.class_transform(3, matrix) == 3
    assert utils.class_transform(4, matrix) == 2
