# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2023 Harry Baker

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
r"""Tests for :mod:`minerva.utils.utils`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
import cmath
import math
import os
import random
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import requests
import torch
from geopy.exc import GeocoderUnavailable
from internet_sabotage import no_connection
from nptyping import Float, NDArray, Shape
from numpy.testing import assert_array_equal
from pytest_lazyfixture import lazy_fixture
from rasterio.crs import CRS
from torchgeo.datasets.utils import BoundingBox, stack_samples
from torchvision.datasets import FakeData

from minerva.models import MinervaModel
from minerva.utils import AUX_CONFIGS, CONFIG, utils, visutils


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_print_banner() -> None:
    utils._print_banner()


@pytest.mark.parametrize(
    ["input", "expected"], [(1, int), ("we want a shrubery...", str), (str, str)]
)
def test_extract_class_type(input: Any, expected: type) -> None:
    assert utils.extract_class_type(input) == expected


def test_is_notebook() -> None:
    assert utils.is_notebook() is False


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


@pytest.mark.parametrize(
    ("string", "substrs", "all_true", "expected"),
    (
        ("siamese-segmentation", "siamese", False, True),
        ("siamese-segmentation", ("ssl", "siamese"), False, True),
        ("siamese-segmentation", ("ssl", "siamese"), True, False),
        ("siamese-segmentation", ("segmentation", "siamese"), True, True),
    ),
)
def test_check_substrings_in_string(
    string: str, substrs, all_true: bool, expected: bool
) -> None:
    assert (
        utils.check_substrings_in_string(string, *substrs, all_true=all_true)
        is expected
    )


def test_datetime_reformat() -> None:
    dt = "2018-12-15"
    assert utils.datetime_reformat(dt, "%Y-%m-%d", "%d.%m.%Y") == "15.12.2018"


def test_ohe_labels() -> None:
    labels = [3, 2, 4, 1, 0]
    correct_targets: NDArray[Shape["5, 6"], Float] = np.array(
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


def test_empty_classes(exp_classes: Dict[int, str]) -> None:
    labels = [(3, 321), (4, 112), (1, 671), (5, 456)]
    assert utils.find_empty_classes(labels, exp_classes) == [0, 2]


def test_eliminate_classes(exp_classes: Dict[int, str]) -> None:
    empty = [0, 2]
    old_cmap = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    new_classes = {0: "5", 1: "1", 2: "4", 3: "3"}
    new_cmap = {0: "5", 1: "1", 2: "4", 3: "3"}
    conversion = {1: 1, 3: 3, 4: 2, 5: 0}

    results = utils.eliminate_classes(
        empty_classes=empty, old_classes=exp_classes, old_cmap=old_cmap
    )

    assert new_classes == results[0]
    assert conversion == results[1]
    assert new_cmap == results[2]


@pytest.mark.parametrize(
    ["in_labels", "in_pred", "out_labels", "out_pred", "out_classes"],
    [
        (
            [1, 1, 3, 5, 1, 4, 1, 5, 3],
            [1, 4, 1, 5, 1, 4, 1, 5, 1],
            [1, 1, 3, 0, 1, 2, 1, 0, 3],
            [1, 2, 1, 0, 1, 2, 1, 0, 1],
            {0: "5", 1: "1", 2: "4", 3: "3"},
        ),
        (
            [2, 4, 5, 1, 1, 3, 0, 2, 1, 5, 1],
            [2, 1, 5, 1, 3, 3, 0, 1, 1, 5, 1],
            [2, 4, 5, 1, 1, 3, 0, 2, 1, 5, 1],
            [2, 1, 5, 1, 3, 3, 0, 1, 1, 5, 1],
            lazy_fixture("exp_classes"),
        ),
    ],
)
def test_check_test_empty(
    exp_classes: Dict[int, str],
    in_labels: List[int],
    in_pred: List[int],
    out_labels: List[int],
    out_pred: List[int],
    out_classes: Dict[int, str],
) -> None:
    results = utils.check_test_empty(in_pred, in_labels, exp_classes)

    assert_array_equal(results[0], out_pred)
    assert_array_equal(results[1], out_labels)
    assert results[2] == out_classes

    assert_array_equal(np.array(results[0]), np.array(out_pred))
    assert_array_equal(np.array(results[1]), np.array(out_labels))
    assert np.array(results[2]) == np.array(out_classes)


def test_find_modes(exp_classes: Dict[int, str]) -> None:
    labels = [1, 1, 3, 5, 1, 4, 1, 5, 3, 3]

    class_dist = utils.find_modes(labels, plot=True)

    assert class_dist == [(1, 4), (3, 3), (5, 2), (4, 1)]

    utils.print_class_dist(class_dist, exp_classes)


def test_file_check() -> None:
    fn = Path("tests", "test.txt")
    with open(fn, "x") as f:
        f.write("")

    utils.exist_delete_check(fn)

    assert fn.exists() is False


@pytest.mark.parametrize(["input", "output"], [(1, 1), (5, 0), (3, 3), (4, 2)])
def test_class_transform(input: int, output: int) -> None:
    matrix = {1: 1, 3: 3, 4: 2, 5: 0}

    assert utils.class_transform(input, matrix) == output


@pytest.mark.parametrize(
    ["param", "comparator", "expect"],
    [
        ([0, 0, 0], [0, 0, 0], [0, 0, 0]),
        ([0, 0], [0, 0, 0], [0, 0, 0]),
        (0, [0, 1, 0], [0, 0, 0]),
        (1, [3, 4, 2], [1, 1, 1]),
    ],
)
def test_check_len(param: Any, comparator: Any, expect: Any) -> None:
    assert utils.check_len(param, comparator) == expect


@pytest.mark.parametrize(
    ["func", "module"],
    [("typing", "Union"), ("datetime", "datetime"), ("torch.utils.data", "DataLoader")],
)
def test_func_by_str(func: str, module: str) -> None:
    assert callable(utils.func_by_str(func, module))


@pytest.mark.parametrize(
    "fmt", ["%d-%m-%Y_%H%M", "%d-%m-%Y", "%Y-%m-%d", "%H%M", "%H:%M"]
)
def test_timestamp_now(fmt: str) -> None:
    assert utils.timestamp_now(fmt) == datetime.now().strftime(fmt)


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


@pytest.mark.parametrize(
    ["input", "exp_len"],
    [
        (np.random.rand(256, 256), 256 * 256),
        (np.random.rand(16, 128, 128), 16 * 128 * 128),
        (list(np.random.rand(256, 256)), 256 * 256),
    ],
)
def test_batch_flatten(input, exp_len: int) -> None:
    assert len(utils.batch_flatten(input)) == exp_len


@pytest.mark.parametrize(
    ["x", "y", "src_crs", "dest_crs", "exp_x", "exp_y"],
    [
        (
            [-1.3958972757520531],
            50.936371897509154,
            utils.WGS84,
            CRS.from_epsg(3857),
            [-155390.57],
            [6610046.36],
        ),
        (
            [-1.3958972757520531, 0.0],
            [50.936371897509154, 51.47687968807581],
            utils.WGS84,
            CRS.from_epsg(3857),
            [-155390.57, 0.0],
            [6610046.36, 6706085.70],
        ),
        (0.0, 6706085.70, CRS.from_epsg(3857), utils.WGS84, 0.0, 51.47687968807581),
    ],
)
def test_transform_coordinates(
    x: Union[List[float], float],
    y: Union[List[float], float],
    src_crs: CRS,
    dest_crs: CRS,
    exp_x: Union[List[float], float],
    exp_y: Union[List[float], float],
) -> None:
    out_x, out_y = utils.transform_coordinates(x, y, src_crs, dest_crs)

    assert out_x == pytest.approx(exp_x)
    assert out_y == pytest.approx(exp_y)


@pytest.mark.parametrize(
    ["bbox", "expected"],
    [
        (
            BoundingBox(1.0, 2.0, 1.0, 2.0, 1.0, 2.0),
            BoundingBox(1.0, 2.0, 1.0, 2.0, 1.0, 2.0),
        ),
        (
            BoundingBox(1.0, 4.0, 1.0, 2.0, 1.0, 2.0),
            BoundingBox(1.0, 3.0, 1.0, 2.0, 1.0, 2.0),
        ),
        (
            BoundingBox(-1.0, 1.0, -1.0, 4.0, 0.0, 4.0),
            BoundingBox(0.0, 1.0, 0.0, 3.0, 0.0, 4.0),
        ),
    ],
)
def test_check_within_bounds(bbox: BoundingBox, expected: BoundingBox) -> None:
    bounds = BoundingBox(0.0, 3.0, 0.0, 3.0, 0.0, 3.0)
    assert utils.check_within_bounds(bbox, bounds) == expected


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


@pytest.mark.parametrize(
    ["lat", "lon", "loc"],
    [
        (53.02324371916741, -1.482418942412615, "Amber Valley, England"),  # Belper, UK.
        (
            str(53.02324371916741),
            str(-1.482418942412615),
            "Amber Valley, England",
        ),  # Belper, UK.
        (
            51.51331165954196,
            -0.08889921085815589,
            "City of London, England",
        ),  # City of London.
        (36.53849331792166, -102.65475905788739, ""),  # Random point in Ohio.
        (30.45028570174185, -76.49581035362436, ""),  # Bermuda Triangle.
        (41.90204312927206, 12.45644780021287, "Civitas Vaticana"),  # Vatican City.
        (-77.844504, 166.707506, "McMurdo Station"),  # McMurdo Station, Antartica.
    ],
)
def test_lat_lon_to_loc(
    lat: Union[float, str], lon: Union[float, str], loc: str
) -> None:
    try:
        requests.head("http://www.google.com/", timeout=1.0)
    except (requests.ConnectionError, requests.ReadTimeout):
        pass
    else:
        try:
            assert utils.lat_lon_to_loc(lat, lon) == loc
        except GeocoderUnavailable:
            pass

    with no_connection(), pytest.raises(
        GeocoderUnavailable, match="Geocoder unavailable"
    ):
        _ = utils.lat_lon_to_loc(lat, lon)


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

    fpr: Dict[Any, NDArray[Any, Any]] = {
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

    tpr: Dict[Any, NDArray[Any, Any]] = {
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

    path = Path(os.getcwd(), "tmp")

    path.mkdir(parents=True, exist_ok=True)
    fn = f"{path}/roc_curve.png"

    visutils.make_roc_curves(
        probs, labels, class_names, cmap_dict, filename=fn, show=True
    )

    # CLean up.
    shutil.rmtree(path)


@pytest.mark.parametrize(
    ["key", "outcome"],
    [
        ("does_not_exist", False),
        ("exist_none", False),
        ("exist_false", False),
        ("exist_true", True),
        ("exist_value", True),
    ],
)
def test_check_dict_key(key: str, outcome: bool) -> None:
    dictionary = {
        "exist_none": None,
        "exist_false": False,
        "exist_true": True,
        "exist_value": 42,
    }

    assert utils.check_dict_key(dictionary, key) is outcome


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
        utils.RESULTS_DIR.mkdir(parents=True)
    except FileExistsError:
        pass

    utils.mkexpdir(name)

    assert (utils.RESULTS_DIR / name).is_dir()

    utils.mkexpdir(name)

    (utils.RESULTS_DIR / name).rmdir()


def test_get_dataset_name() -> None:
    assert utils.get_dataset_name() == "Chesapeake7"


def test_run_tensorboard() -> None:
    try:
        env_name = Path(os.environ["CONDA_DEFAULT_ENV"]).name
    except KeyError:
        env_name = "base"

    assert utils.run_tensorboard("non_exp", env_name=env_name) is None

    exp_name = "exp1"

    path = Path(tempfile.gettempdir(), exp_name)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    assert (
        utils.run_tensorboard(
            exp_name, env_name=env_name, path=tempfile.gettempdir(), _testing=True
        )
        == 0
    )

    results_dir = CONFIG["dir"]["results"]
    del CONFIG["dir"]["results"]

    assert utils.run_tensorboard(exp_name, env_name=env_name) is None

    utils.CONFIG["dir"]["results"] = results_dir

    path.rmdir()


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
    utils.print_config()
    utils.print_config(utils.CLASSES)


def test_calc_grad(exp_mlp: MinervaModel) -> None:
    batch_size = 16
    x = torch.rand(batch_size, (64))
    y = torch.LongTensor(np.random.randint(0, 8, size=batch_size))

    optimiser = torch.optim.SGD(exp_mlp.parameters(), lr=1.0e-3)
    exp_mlp.set_optimiser(optimiser)
    _ = exp_mlp.step(x, y, train=True)

    grad = utils.calc_grad(exp_mlp)
    assert type(grad) is float
    assert grad != 0.0
    assert utils.calc_grad(42) is None  # type: ignore[arg-type]


def test_tsne_cluster() -> None:
    clusters = utils.tsne_cluster(np.random.rand(10, 100))
    assert isinstance(clusters, NDArray)
    assert clusters.shape == (10, 2)


def test_calc_norm_euc_dist() -> None:
    a1 = np.random.random_integers(0, 8, 10)
    b1 = np.random.random_integers(0, 8, 10)

    b2 = np.random.random_integers(0, 8, 8)

    assert isinstance(utils.calc_norm_euc_dist(a1, b1), float)

    with pytest.raises(AssertionError):
        utils.calc_norm_euc_dist(a1, b2)
