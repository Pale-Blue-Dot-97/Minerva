from typing import Union, Tuple, Dict, Any
from collections import defaultdict
import os
import random
import math
import cmath
from minerva.utils import utils, config, aux_configs
import numpy as np
import torch
from torchgeo.datasets.utils import BoundingBox, stack_samples
from numpy.testing import assert_array_equal
from datetime import datetime
from torch.utils.data import DataLoader


def test_return_updated_kwargs() -> None:
    @utils.return_updated_kwargs
    def example_func(*args, **kwargs) -> Tuple[Any, Dict[str, Any]]:

        _ = (
            kwargs["update_1"] * kwargs["update_3"]
            - kwargs["static_2"] / args[1] * args[0]
        )

        updates = {"update_1": 100, "update_2": "Minerva", "update_3": 24}

        return 37, updates

    old_kwargs = {
        "update_1": 0,
        "static_1": True,
        "update_2": "minerva",
        "update_3": 12,
        "static_2": 42,
    }

    new_kwargs = {
        "update_1": 100,
        "static_1": True,
        "update_2": "Minerva",
        "update_3": 24,
        "static_2": 42,
    }

    arg1, arg2 = 23, 45

    results = example_func(arg1, arg2, **old_kwargs)

    assert results[0] == 37
    assert type(results[1]) is dict
    assert results[1] == new_kwargs


def test_pair_collate() -> None:
    collator = utils.pair_collate(stack_samples)

    sample = {"image": torch.rand(4, 224, 224), "mask": torch.rand(224, 224)}

    batch = [(sample, sample)] * 16

    output = collator(batch)

    assert type(output) is tuple
    assert type(output[0]) is defaultdict
    assert type(output[1]) is defaultdict
    assert len(output[0]["image"]) == len(output[1]["image"])
    assert len(output[1]["mask"]) == len(output[0]["image"])


def test_cuda_device() -> None:
    assert type(utils.get_cuda_device()) is torch.device


def test_config_loading() -> None:
    assert type(config) is dict
    assert type(aux_configs) is dict


def test_datetime_reformat() -> None:
    dt = "2018-12-15"
    assert utils.datetime_reformat(dt, "%Y-%m-%d", "%d.%m.%Y") == "15.12.2018"


def test_ohe_labels() -> None:
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


def test_empty_classes() -> None:
    labels = [(3, 321), (4, 112), (1, 671), (5, 456)]
    classes = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    assert utils.find_empty_classes(labels, classes) == [0, 2]


def test_eliminate_classes() -> None:
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


def test_check_test_empty() -> None:
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


def test_find_subpopulations() -> None:
    classes = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    labels = [1, 1, 3, 5, 1, 4, 1, 5, 3, 3]

    class_dist = utils.find_modes(labels)

    assert class_dist == [(1, 4), (3, 3), (5, 2), (4, 1)]

    assert utils.print_class_dist(class_dist, classes) is None


def test_file_check() -> None:
    fn = "tests/test.txt"
    with open(fn, "x") as f:
        f.write("")

    utils.exist_delete_check(fn)

    assert os.path.exists(fn) is False


def test_class_transform() -> None:
    matrix = {1: 1, 3: 3, 4: 2, 5: 0}

    assert utils.class_transform(1, matrix) == 1
    assert utils.class_transform(5, matrix) == 0
    assert utils.class_transform(3, matrix) == 3
    assert utils.class_transform(4, matrix) == 2


def test_check_len() -> None:
    assert utils.check_len([0, 0, 0], [0, 0, 0]) == [0, 0, 0]
    assert utils.check_len([0, 0], [0, 0, 0]) == [0, 0, 0]
    assert utils.check_len(0, [0, 1, 0]) == [0, 0, 0]
    assert utils.check_len(1, [3, 4, 2]) == [1, 1, 1]


def test_func_by_str() -> None:
    assert utils.func_by_str("typing", "Union") is Union
    assert utils.func_by_str("datetime", "datetime") is datetime
    assert utils.func_by_str("torch.utils.data", "DataLoader") is DataLoader


def test_timestamp_now() -> None:
    assert utils.timestamp_now("%d-%m-%Y_%H%M") == datetime.now().strftime(
        "%d-%m-%Y_%H%M"
    )
    assert utils.timestamp_now("%d-%m-%Y") == datetime.now().strftime("%d-%m-%Y")
    assert utils.timestamp_now("%Y-%m-%d") == datetime.now().strftime("%Y-%m-%d")
    assert utils.timestamp_now("%H%M") == datetime.now().strftime("%H%M")
    assert utils.timestamp_now("%H:%M") == datetime.now().strftime("%H:%M")


def test_find_geo_similar() -> None:
    max_r = 120

    r = random.randint(0, max_r)
    assert r <= 120

    phi = random.random() * math.pow(-1, random.randint(1, 2)) * math.pi
    assert phi <= math.pi
    assert phi >= -math.pi

    z = cmath.rect(r, phi)
    x, y = z.real, z.imag

    assert np.sqrt((math.pow(x, 2) + math.pow(y, 2))) <= max_r

    bbox = BoundingBox(10, 20, 20, 30, 1, 2)

    assert type(utils.find_geo_similar(bbox, max_r)) is BoundingBox


def test_batch_flatten() -> None:
    a = np.random.rand(256, 256)
    b = np.random.rand(16, 128, 128)
    c = list(a)

    assert len(utils.batch_flatten(a)) == 256 * 256
    assert len(utils.batch_flatten(b)) == 16 * 128 * 128
    assert len(utils.batch_flatten(c)) == 256 * 256
