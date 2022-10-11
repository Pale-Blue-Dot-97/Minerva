import cmath
import math
import ntpath
import os
import random
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import torch
from internet_sabotage import no_connection
from numpy.testing import assert_array_equal
from rasterio.crs import CRS
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import BoundingBox, stack_samples
from torchvision.datasets import FakeData

from minerva.utils import AUX_CONFIGS, CONFIG, utils, visutils


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


def test_pair_return() -> None:
    _dataset = utils.pair_return(FakeData)
    dataset = _dataset(size=64)

    returns = dataset[(1, 42)]
    assert len(returns) == 2
    assert len(returns[0]) == 2
    assert len(returns[1]) == 2

    assert hasattr(dataset, "size") is True
    assert hasattr(dataset, "wrap") is True

    with pytest.raises(AttributeError):
        getattr(dataset, "__len__")

    assert repr(dataset) == repr(FakeData(size=64))


def test_cuda_device() -> None:
    assert type(utils.get_cuda_device()) is torch.device  # type: ignore[attr-defined]


def test_config_loading() -> None:
    assert type(CONFIG) is dict
    assert type(AUX_CONFIGS) is dict


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
    assert_array_equal(correct_targets, targets)


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

    assert_array_equal(results_1[0], new_pred)
    assert_array_equal(results_1[1], new_labels)
    assert results_1[2] == new_classes

    old_labels_2 = [2, 4, 5, 1, 1, 3, 0, 2, 1, 5, 1]
    old_pred_2 = [2, 1, 5, 1, 3, 3, 0, 1, 1, 5, 1]

    results_2 = utils.check_test_empty(old_pred_2, old_labels_2, old_classes)

    assert_array_equal(results_2[0], old_pred_2)
    assert_array_equal(results_2[1], old_labels_2)
    assert results_2[2] == old_classes

    old_labels_3 = np.array(old_labels_2)
    old_pred_3 = np.array(old_pred_2)

    results_3 = utils.check_test_empty(old_pred_3, old_labels_3, old_classes)

    assert_array_equal(results_3[0], old_pred_3)
    assert_array_equal(results_3[1], old_labels_3)
    assert results_3[2] == old_classes


def test_find_modes() -> None:
    classes = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    labels = [1, 1, 3, 5, 1, 4, 1, 5, 3, 3]

    class_dist = utils.find_modes(labels, plot=True)

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


def test_transform_coordinates() -> None:
    x_1 = [-1.3958972757520531]
    y_1 = 50.936371897509154

    x_2 = [-1.3958972757520531, 0.0]
    y_2 = [50.936371897509154, 51.47687968807581]

    x_3 = 0.0
    y_3 = 6706085.70

    src_crs = utils.WGS84
    new_crs = CRS.from_epsg(3857)

    new_x_1 = [-155390.57]
    new_y_1 = [6610046.36]

    new_x_2 = [-155390.57, 0.0]
    new_y_2 = [6610046.36, 6706085.70]

    new_y_3 = 51.47687968807581

    results_1 = utils.transform_coordinates(x_1, y_1, src_crs, new_crs)
    results_2 = utils.transform_coordinates(x_2, y_2, src_crs, new_crs)
    results_3 = utils.transform_coordinates(x_3, y_3, new_crs, src_crs)

    assert results_1[0] == pytest.approx(new_x_1)
    assert results_1[1] == pytest.approx(new_y_1)
    assert results_2[0] == pytest.approx(new_x_2)
    assert results_2[1] == pytest.approx(new_y_2)
    assert results_3[0] == pytest.approx(0.0)
    assert results_3[1] == pytest.approx(new_y_3)


def test_check_within_bounds() -> None:
    bounds = BoundingBox(0.0, 3.0, 0.0, 3.0, 0.0, 3.0)

    bbox_1 = BoundingBox(1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
    bbox_2 = BoundingBox(1.0, 4.0, 1.0, 2.0, 1.0, 2.0)
    bbox_3 = BoundingBox(-1.0, 1.0, -1.0, 4.0, 0.0, 4.0)

    correct_2 = BoundingBox(1.0, 3.0, 1.0, 2.0, 1.0, 2.0)
    correct_3 = BoundingBox(0.0, 1.0, 0.0, 3.0, 0.0, 4.0)

    new_bbox_2 = utils.check_within_bounds(bbox_2, bounds)
    new_bbox_3 = utils.check_within_bounds(bbox_3, bounds)

    assert utils.check_within_bounds(bbox_1, bounds) == bbox_1
    assert new_bbox_2 != bbox_2
    assert new_bbox_2 == correct_2
    assert new_bbox_3 != bbox_3
    assert new_bbox_3 == correct_3


def test_dec2deg() -> None:
    lon_1 = -1.3958972757520531
    lat_1 = 50.936371897509154

    lon_2 = 172.63809028021004
    lat_2 = -43.525396528993525

    assert utils.deg_to_dms(lon_1, "lon") == "1º23'45\"W"
    assert utils.deg_to_dms(lat_1, "lat") == "50º56'11\"N"

    assert utils.dec2deg([lon_1, lon_2], "lon") == ["1º23'45\"W", "172º38'17\"E"]
    assert utils.dec2deg([lat_1, lat_2], "lat") == ["50º56'11\"N", "43º31'31\"S"]


def test_get_centre_loc() -> None:
    bbox = BoundingBox(1.0, 3.0, 1.0, 5.0, 0.0, 2.0)

    centre = utils.get_centre_loc(bbox)
    assert pytest.approx(centre[0]) == 2.0
    assert pytest.approx(centre[1]) == 3.0


def test_lat_lon_to_loc() -> None:
    # Belper, UK.
    lat_1 = 53.02324371916741
    lon_1 = -1.482418942412615

    # City of London.
    lat_2 = 51.51331165954196
    lon_2 = -0.08889921085815589

    # Random point in Ohio.
    lat_3 = 36.53849331792166
    lon_3 = -102.65475905788739

    # Bermuda Triangle.
    lat_4 = 30.45028570174185
    lon_4 = -76.49581035362436

    # Vatican City.
    lat_5 = 41.90204312927206
    lon_5 = 12.45644780021287

    assert utils.lat_lon_to_loc(lat_1, lon_1) == "Amber Valley, England"
    assert utils.lat_lon_to_loc(str(lat_1), str(lon_1)) == "Amber Valley, England"
    assert utils.lat_lon_to_loc(lat_2, lon_2) == "City of London, England"
    assert utils.lat_lon_to_loc(lat_3, lon_3) == "Cimarron County, Oklahoma"

    assert utils.lat_lon_to_loc(lat_4, lon_4) == ""
    assert utils.lat_lon_to_loc(lat_5, lon_5) in (
        "Civitas Vaticana - Città del Vaticano",
        "Civitas Vaticana",
    )
    with no_connection():
        assert utils.lat_lon_to_loc(lat_1, lon_1) == ""


def test_class_weighting() -> None:
    class_dist = [(5, 750), (3, 150), (2, 60), (4, 20), (1, 12), (0, 8)]
    cls_weights = {5: 1 / 750, 3: 1 / 150, 2: 1 / 60, 4: 1 / 20, 1: 1 / 12, 0: 1 / 8}
    norm_cls_weights = {
        5: 1000 / 750,
        3: 1000 / 150,
        2: 1000 / 60,
        4: 1000 / 20,
        1: 1000 / 12,
        0: 1000 / 8,
    }

    results_1 = utils.class_weighting(class_dist)
    results_2 = utils.class_weighting(class_dist, normalise=True)

    for key in results_1.keys():
        assert cls_weights[key] == pytest.approx(results_1[key])

    for key in results_2.keys():
        assert norm_cls_weights[key] == pytest.approx(results_2[key])


def test_class_dist_transform() -> None:
    old_class_dist = [(5, 750), (7, 150), (2, 60), (4, 20), (6, 12), (0, 8)]
    new_class_dist = [(5, 750), (3, 150), (2, 60), (4, 20), (1, 12), (0, 8)]

    transform = {5: 5, 7: 3, 2: 2, 4: 4, 6: 1, 0: 0}

    assert utils.class_dist_transform(old_class_dist, transform) == new_class_dist


def test_class_frac() -> None:
    class_dist = [(5, 750), (3, 150), (2, 60), (4, 20), (1, 12), (0, 8)]
    row_dict = {"MODES": class_dist, "AnotherColumn": "stuff"}
    row = pd.Series(row_dict)

    new_row = {
        "MODES": class_dist,
        "AnotherColumn": "stuff",
        5: 750 / 1000,
        3: 150 / 1000,
        2: 60 / 1000,
        4: 20 / 1000,
        1: 12 / 1000,
        0: 8 / 1000,
    }
    new_row = utils.class_frac(row)

    for mode in class_dist:
        assert mode[1] / 1000 == pytest.approx(new_row[mode[0]])

    assert new_row["MODES"] == class_dist
    assert new_row["AnotherColumn"] == "stuff"


def test_cloud_cover() -> None:
    assert utils.cloud_cover(np.zeros((5, 5))) == pytest.approx(0.0)

    cloud_mask = [[0, 5, 0], [3, 4, 6], [1, 3, 10]]

    assert utils.cloud_cover(np.array(cloud_mask)) == pytest.approx(32.0 / 9.0)


def test_threshold_scene_select() -> None:
    df = pd.DataFrame()

    df["DATE"] = ["2018-12-12", "2017-04-24", "2018-06-21"]
    df["COVER"] = [0.7, 0.8, 0.1]

    assert utils.threshold_scene_select(df, thres=0.3) == ["2018-06-21"]


def test_find_best_of() -> None:
    df = pd.DataFrame()

    df["PATCH"] = ["34MP", "34MP", "34MP", "56TG", "56TG"]
    df["DATE"] = ["2018_12_12", "2017_04_24", "2018_06_21", "2018_07_18", "2018_08_23"]
    df["COVER"] = [0.7, 0.8, 0.1, 0.4, 0.5]

    scene = utils.find_best_of(
        patch_id="34MP", manifest=df, selector=utils.threshold_scene_select, thres=0.3
    )
    assert scene == ["2018_06_21"]


def test_modes_from_manifest() -> None:
    df = pd.DataFrame()

    class_dist = [
        (5, 750 / 7),
        (7, 150 / 7),
        (2, 60 / 7),
        (4, 20 / 7),
        (6, 12 / 7),
        (0, 8 / 7),
    ]

    counts = [
        (5, [250, 100, 50, 40, 60, 210, 40]),
        (7, [10, 20, 20, 50, 40, 5, 5]),
        (2, [2, 4, 6, 30, 10, 8, 0]),
        (4, [2, 3, 0, 1, 4, 8, 2]),
        (6, [3, 4, 0, 1, 3, 1, 0]),
        (0, [1, 0, 0, 3, 4, 0, 0]),
        (1, [0, 0, 0, 0, 0, 0, 0]),
    ]

    for mode in counts:
        df[mode[0]] = mode[1]

    assert utils.modes_from_manifest(df, plot=True) == class_dist


def test_make_classification_report() -> None:
    pred = [0, 3, 3, 1, 3, 2, 0, 0]
    labels = [0, 3, 2, 1, 3, 2, 1, 0]
    class_labels = {0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3"}

    df = pd.DataFrame()
    df["LABEL"] = class_labels.values()
    df["precision"] = [2.0 / 3.0, 1.0, 1.0, 2.0 / 3.0]
    df["recall"] = [1.0, 0.5, 0.5, 1.0]
    df["f1-score"] = [0.8, 2.0 / 3.0, 2.0 / 3.0, 0.8]
    df["support"] = [2.0, 2.0, 2.0, 2.0]

    df.index = df.index.astype(str)

    cr_df = utils.make_classification_report(pred, labels, class_labels, p_dist=True)

    assert df.equals(cr_df)


def test_compute_roc_curves() -> None:
    probs = [
        [0.7, 0.1, 0.05, 0.15],
        [0.01, 0.01, 0.03, 0.95],
        [0.005, 0.015, 0.08, 0.9],
        [0.01, 0.29, 0.45, 0.25],
        [0.02, 0.11, 0.06, 0.81],
        [0.04, 0.08, 0.72, 0.16],
        [0.35, 0.20, 0.15, 0.30],
        [0.40, 0.25, 0.15, 0.20],
    ]

    labels = [0, 3, 2, 1, 3, 2, 1, 0]
    class_labels = [0, 1, 2, 3]

    fpr = {
        0: np.array([0.0, 0.0, 0.0, 0.5, 5.0 / 6.0, 1.0]),
        1: np.array([0.0, 0.0, 1.0 / 6.0, 1.0 / 6.0, 1.0]),
        2: np.array([0.0, 0.0, 1.0 / 6.0, 0.5, 0.5, 1.0]),
        3: np.array([0.0, 0.0, 1.0 / 6.0, 1.0 / 6.0, 1.0]),
        "micro": np.array(
            [
                0.0,
                0.0,
                1.0 / 24.0,
                1.0 / 24.0,
                5.0 / 60.0,
                5.0 / 60.0,
                1.0 / 6.0,
                1.0 / 6.0,
                0.25,
                7.0 / 24.0,
                1.0 / 3.0,
                11.0 / 24.0,
                13.0 / 24.0,
                14.0 / 24.0,
                5.0 / 6.0,
                23.0 / 24.0,
                1.0,
            ]
        ),
        "macro": np.array([0.0, 1.0 / 6.0, 0.5, 5.0 / 6.0, 1.0]),
    }

    tpr = {
        0: np.array([0.0, 0.5, 1.0, 1.0, 1.0, 1.0]),
        1: np.array([0.0, 0.5, 0.5, 1.0, 1.0]),
        2: np.array([0.0, 0.5, 0.5, 0.5, 1.0, 1.0]),
        3: np.array([0.0, 0.5, 0.5, 1.0, 1.0]),
        "micro": np.array(
            [
                0.0,
                0.125,
                0.125,
                0.5,
                0.5,
                0.625,
                0.625,
                0.75,
                0.75,
                0.875,
                0.875,
                0.875,
                0.875,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        ),
        "macro": np.array([0.625, 0.875, 1.0, 1.0, 1.0]),
    }

    auc = {
        0: 1.0,
        1: 11.0 / 12.0,
        2: 0.75,
        3: 11.0 / 12.0,
        "micro": 0.8489583333333333,
        "macro": 0.9375000000000001,
    }

    results = utils.compute_roc_curves(
        np.array(probs), labels, class_labels, micro=True, macro=True
    )

    for key in fpr:
        assert_array_equal(results[0][key], fpr[key])
        assert_array_equal(results[1][key], tpr[key])

    assert results[2] == pytest.approx(auc)

    class_names = {0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3"}
    cmap_dict = {0: "#000000", 1: "#00c5ff", 2: "#267300", 3: "#a3ff73"}

    path = os.getcwd()
    if isinstance(path, str):
        path = os.path.join(path, "tmp")
    else:
        path = os.path.join(*path, "tmp")

    os.makedirs(path, exist_ok=True)
    fn = f"{path}/roc_curve.png"

    assert (
        visutils.make_roc_curves(
            probs, labels, class_names, cmap_dict, filename=fn, show=True
        )
        is None
    )

    # CLean up.
    shutil.rmtree(path)


def test_check_dict_key() -> None:
    dictionary = {
        "exist_none": None,
        "exist_false": False,
        "exist_true": True,
        "exist_value": 42,
    }

    assert utils.check_dict_key(dictionary, "does_not_exist") is False
    assert utils.check_dict_key(dictionary, "exist_none") is False
    assert utils.check_dict_key(dictionary, "exist_false") is False
    assert utils.check_dict_key(dictionary, "exist_true")
    assert utils.check_dict_key(dictionary, "exist_value")


def test_load_data_specs() -> None:
    class_dist = [(3, 321), (4, 112), (1, 671), (5, 456)]

    new_classes = {
        0: "Surfaces",
        1: "Water",
        2: "Barren",
        3: "Low Vegetation",
    }
    new_cmap = {0: "#9c9c9c", 1: "#00c5ff", 2: "#ffaa00", 3: "#a3ff73"}
    conversion = {1: 1, 3: 3, 4: 2, 5: 0}

    results_1 = utils.load_data_specs(class_dist, elim=False)
    results_2 = utils.load_data_specs(class_dist, elim=True)

    assert results_1[0] == utils.CLASSES
    assert results_1[1] == {}
    assert results_1[2] == utils.CMAP_DICT
    assert new_classes == results_2[0]
    assert conversion == results_2[1]
    assert new_cmap == results_2[2]


def test_mkexpdir() -> None:
    name = "exp1"

    try:
        os.makedirs(utils.RESULTS_DIR)
    except FileExistsError:
        pass

    utils.mkexpdir(name)

    assert os.path.isdir(os.path.join(utils.RESULTS_DIR, name))

    assert utils.mkexpdir(name) is None

    os.rmdir(os.path.join(utils.RESULTS_DIR, name))


def test_get_dataset_name() -> None:
    assert utils.get_dataset_name() == "Chesapeake7"


def test_run_tensorboard() -> None:
    try:
        env_name = ntpath.basename(os.environ["CONDA_DEFAULT_ENV"])
    except KeyError:
        env_name = "base"

    assert utils.run_tensorboard("non_exp", env_name=env_name) is None

    exp_name = "exp1"

    path = tempfile.gettempdir()
    if not isinstance(path, str):
        path = os.path.join(*path)

    if not os.path.exists(os.path.join(path, exp_name)):
        os.mkdir(os.path.join(path, exp_name))

    assert (
        utils.run_tensorboard(
            exp_name, env_name=env_name, path=tempfile.gettempdir(), _testing=True
        )
        == 0
    )

    results_dir = CONFIG["dir"]["results"]
    del CONFIG["dir"]["results"]

    print(CONFIG["dir"])

    assert utils.run_tensorboard(exp_name, env_name=env_name) is None

    utils.CONFIG["dir"]["results"] = results_dir

    os.rmdir(os.path.join(path, exp_name))


def test_calc_constrastive_acc() -> None:
    pred = torch.Tensor(
        [
            [0.1642, 0.6131, 0.1704, 0.7223, 0.8332, 0.2259, 0.0694, 0.5820],
            [0.5060, 0.3063, 0.0573, 0.2115, 0.5379, 0.3470, 0.7090, 0.8790],
            [0.7996, 0.9784, 0.6017, 0.6036, 0.5077, 0.0555, 0.8548, 0.3769],
            [0.9803, 0.1221, 0.9618, 0.8202, 0.8939, 0.9609, 0.6091, 0.8273],
        ]
    )

    correct = torch.Tensor([1, 0, 2, 0]).tolist()

    results = utils.calc_contrastive_acc(pred).tolist()

    assert results == correct


def test_print_config() -> None:
    assert utils.print_config() is None
    assert utils.print_config(utils.CLASSES) is None


def test_calc_grad() -> None:
    pass
