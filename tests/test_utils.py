from typing import Union
import os
from minerva.utils import utils, config, aux_configs
import numpy as np
from numpy.testing import assert_array_equal
from datetime import datetime
from torch.utils.data import DataLoader


def test_config_loading():
    assert type(config) is dict
    assert type(aux_configs) is dict


def test_datetime_reformat():
    dt = "2018-12-15"
    assert utils.datetime_reformat(dt, "%Y-%m-%d", "%d.%m.%Y") == "15.12.2018"


def test_ohe_labels():
    labels = [3, 2, 4, 1, 0]
    correct_targets = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    targets = utils.labels_to_ohe(labels=labels, n_classes=6)
    assert assert_array_equal(correct_targets, targets) is None


def test_empty_classes():
    labels = [(3, 321), (4, 112), (1, 671), (5, 456)]
    classes = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    assert utils.find_empty_classes(labels, classes) == [0, 2]


def test_eliminate_classes():
    empty = [0, 2]
    old_classes = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    old_cmap = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    new_classes = {0: "5", 1: "1", 2: "4", 3: "3"}
    new_cmap = {0: "5", 1: "1", 2: "4", 3: "3"}
    conversion = {1: 1, 3: 3, 4: 2, 5: 0}

    results = utils.eliminate_classes(
        empty_classes=empty, old_classes=old_classes, old_cmap=old_cmap
    )

    assert new_classes == results[0]
    assert conversion == results[1]
    assert new_cmap == results[2]


def test_check_test_empty():
    old_classes = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    new_classes = {0: "5", 1: "1", 2: "4", 3: "3"}
    old_labels_1 = [1, 1, 3, 5, 1, 4, 1, 5, 3]
    new_labels = [1, 1, 3, 0, 1, 2, 1, 0, 3]
    old_pred_1 = [1, 4, 1, 5, 1, 4, 1, 5, 1]
    new_pred = [1, 2, 1, 0, 1, 2, 1, 0, 1]

    results_1 = utils.check_test_empty(old_pred_1, old_labels_1, old_classes)

    assert assert_array_equal(results_1[0], new_pred) is None
    assert assert_array_equal(results_1[1], new_labels) is None
    assert results_1[2] == new_classes

    old_labels_2 = [2, 4, 5, 1, 1, 3, 0, 2, 1, 5, 1]
    old_pred_2 = [2, 1, 5, 1, 3, 3, 0, 1, 1, 5, 1]

    results_2 = utils.check_test_empty(old_pred_2, old_labels_2, old_classes)

    assert assert_array_equal(results_2[0], old_pred_2) is None
    assert assert_array_equal(results_2[1], old_labels_2) is None
    assert results_2[2] == old_classes

    old_labels_3 = np.array(old_labels_2)
    old_pred_3 = np.array(old_pred_2)

    results_3 = utils.check_test_empty(old_pred_3, old_labels_3, old_classes)

    assert assert_array_equal(results_3[0], old_pred_3) is None
    assert assert_array_equal(results_3[1], old_labels_3) is None
    assert results_3[2] == old_classes


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


def test_check_len():
    assert utils.check_len([0, 0, 0], [0, 0, 0]) == [0, 0, 0]
    assert utils.check_len([0, 0], [0, 0, 0]) == [0, 0, 0]
    assert utils.check_len(0, [0, 1, 0]) == [0, 0, 0]
    assert utils.check_len(1, [3, 4, 2]) == [1, 1, 1]


def test_func_by_str():
    assert utils.func_by_str("typing", "Union") is Union
    assert utils.func_by_str("datetime", "datetime") is datetime
    assert utils.func_by_str("torch.utils.data", "DataLoader") is DataLoader


def test_timestamp_now():
    assert utils.timestamp_now("%d-%m-%Y_%H%M") == datetime.now().strftime(
        "%d-%m-%Y_%H%M"
    )
    assert utils.timestamp_now("%d-%m-%Y") == datetime.now().strftime("%d-%m-%Y")
    assert utils.timestamp_now("%Y-%m-%d") == datetime.now().strftime("%Y-%m-%d")
    assert utils.timestamp_now("%H%M") == datetime.now().strftime("%H%M")
    assert utils.timestamp_now("%H:%M") == datetime.now().strftime("%H:%M")
