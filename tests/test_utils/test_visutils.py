# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2024 Harry Baker

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
r"""Tests for :mod:`minerva.utils.visutils`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import matplotlib as mlp
import numpy as np
import pytest
import torch
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.image import AxesImage
from nptyping import NDArray, Shape
from numpy.testing import assert_array_equal
from rasterio.crs import CRS
from torchgeo.datasets import GeoDataset
from torchgeo.samplers import get_random_bounding_box

from minerva.utils import utils, visutils


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_de_interlace() -> None:
    x_1 = [1, 1, 1, 1, 1]
    x_2 = [2, 2, 2, 2, 2]
    x_3 = [3, 3, 3, 3, 3]

    x = [x_1, x_2, x_3, x_1, x_2, x_3]

    x2: NDArray[Shape["30"], Any] = np.array([x_1, x_1, x_2, x_2, x_3, x_3]).flatten()

    assert_array_equal(visutils.de_interlace(x, 3), x2)


def test_dec_extent_to_deg(bounds_for_test_img) -> None:
    shape = (224, 224)
    new_crs = CRS.from_epsg(26918)

    corners, lat, lon = visutils.dec_extent_to_deg(
        shape, bounds_for_test_img, src_crs=visutils.WGS84, new_crs=new_crs
    )

    correct_lat = [
        8557744.60333314,
        8558100.10001212,
        8558455.5966911,
        8558811.09337008,
        8559166.59004907,
        8559522.08672805,
        8559877.58340703,
        8560233.08008601,
    ]

    correct_lon = [
        4974378.15360064,
        4974110.85466129,
        4973843.55572193,
        4973576.25678258,
        4973308.95784323,
        4973041.65890387,
        4972774.35996452,
        4972507.06102517,
    ]

    assert corners == (0, 224, 0, 224)
    assert lat == pytest.approx(list(correct_lat))
    assert lon == pytest.approx(list(correct_lon))


def test_get_mlp_cmap() -> None:
    og_cmap = mlp.colormaps["viridis"]  # type: ignore

    cmap = visutils.get_mlp_cmap(og_cmap, 8)
    assert isinstance(cmap, Colormap)


def test_discrete_heatmap(
    random_mask, exp_classes: Dict[int, str], exp_cmap_dict: Dict[int, str]
) -> None:
    cmap = ListedColormap(exp_cmap_dict.values())  # type: ignore
    visutils.discrete_heatmap(random_mask, list(exp_classes.values()), cmap_style=cmap)


def test_stack_rgb() -> None:
    red: NDArray[Shape["3, 3"], Any] = np.array(
        [[25.0, 12.0, 11.0], [34.0, 55.0, 89.0], [23.0, 18.0, 76.0]]
    )

    blue: NDArray[Shape["3, 3"], Any] = np.array(
        [[16.0, 17.0, 18.0], [19.0, 23.0, 24.0], [78.0, 67.0, 54.0]]
    )

    green: NDArray[Shape["3, 3"], Any] = np.array(
        [[3.0, 2.0, 1.0], [9.0, 11.0, 34.0], [23.0, 15.0, 128.0]]
    )

    image_1: NDArray[Shape["3, 3, 3"], Any] = np.array([red, green, blue])

    correct = np.dstack((blue, green, red)) / 255.0
    result_1 = visutils.stack_rgb(image_1, max_value=255)

    assert_array_equal(result_1, correct)


def test_make_rgb_image(random_image) -> None:
    assert type(visutils.make_rgb_image(random_image)) is AxesImage


def test_labelled_rgb_image(
    random_mask,
    random_image,
    bounds_for_test_img,
    exp_classes: Dict[int, str],
    exp_cmap_dict: Dict[int, str],
) -> None:
    path = tempfile.gettempdir()
    name = "pretty_pic"
    cmap = ListedColormap(exp_cmap_dict.values())  # type: ignore

    fn = visutils.labelled_rgb_image(
        random_image,
        random_mask,
        bounds_for_test_img,
        visutils.WGS84,
        path,
        name,
        list(exp_classes.values()),
        cmap_style=cmap,
    )

    correct_fn = Path(path, f"{name}_RGBHM.png")
    assert fn == correct_fn


def test_make_gif(
    bounds_for_test_img, exp_classes: Dict[int, str], exp_cmap_dict: Dict[int, str]
) -> None:
    dates = ["2018-01-15", "2018-07-03", "2018-11-30"]
    images = np.random.rand(3, 32, 32, 3)
    masks = np.random.randint(0, 7, size=(3, 32, 32))

    path = Path(os.getcwd(), "tmp")

    path.mkdir(parents=True, exist_ok=True)

    # Creates the GIF filename.
    gif_fn = f"{path}/new_gif.gif"

    cmap = ListedColormap(exp_cmap_dict.values())  # type: ignore

    visutils.make_gif(
        dates,
        images,
        masks,
        bounds_for_test_img,
        visutils.WGS84,
        list(exp_classes.values()),
        gif_fn,
        path,
        cmap,
    )

    # CLean up.
    shutil.rmtree(path)


def test_prediction_plot(
    random_image, random_mask, bounds_for_test_img, exp_classes: Dict[int, str]
) -> None:
    pred = np.random.randint(0, 8, size=(32, 32))

    src_crs = utils.WGS84

    sample = {
        "image": random_image,
        "mask": random_mask,
        "pred": pred,
        "bounds": bounds_for_test_img,
    }
    visutils.prediction_plot(sample, "101", exp_classes, src_crs)


def test_seg_plot(
    results_root: Path,
    default_dataset: GeoDataset,
    exp_classes: Dict[int, str],
    exp_cmap_dict: Dict[int, str],
    monkeypatch,
) -> None:

    batch_size = 2
    n_batches = 2

    size = (32, 32)

    bounds = default_dataset.bounds
    bbox = get_random_bounding_box(bounds, size, res=1.0)

    z = list(np.random.randint(0, 7, size=(n_batches, batch_size, *size)))
    ids = []

    for i in range(n_batches):
        ids.append([f"{i}-{j}" for j in range(batch_size)])

    bboxes = [bbox] * int(n_batches * batch_size)

    fn_prefix = results_root / "seg_plot"

    monkeypatch.setattr(visutils, "_MAX_SAMPLES", 3)
    visutils.seg_plot(
        z=z,
        y=z,
        ids=ids,  # type: ignore[arg-type]
        bounds=bboxes,
        task_name="test-test",
        classes=exp_classes,
        colours=exp_cmap_dict,
        fn_prefix=fn_prefix,
        frac=1.0,
    )


def test_plot_subpopulations(
    exp_classes: Dict[int, str], exp_cmap_dict: Dict[int, str]
) -> None:
    class_dist = [(1, 25000), (0, 1300), (2, 100), (3, 2)]

    fn = Path("plot.png")

    visutils.plot_subpopulations(
        class_dist, exp_classes, cmap_dict=exp_cmap_dict, filename=fn, save=True
    )

    fn.unlink(missing_ok=True)


def test_plot_history() -> None:
    train_loss = {"x": list(range(1, 11)), "y": np.random.rand(10)}
    train_acc = {"x": list(range(1, 11)), "y": np.random.rand(10)}

    val_loss = {"x": list(range(1, 11)), "y": np.random.rand(10)}
    val_acc = {"x": list(range(1, 11)), "y": np.random.rand(10)}

    metrics = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    filename = Path("plot.png")

    visutils.plot_history(metrics, filename, show=True)

    filename.unlink(missing_ok=True)


def test_make_confusion_matrix(exp_classes: Dict[int, str]) -> None:
    batch_size = 2
    patch_size = (32, 32)

    pred_1 = np.random.randint(0, 8, size=batch_size * patch_size[0] * patch_size[1])
    labels_1 = np.random.randint(0, 8, size=batch_size * patch_size[0] * patch_size[1])

    pred_2 = np.random.randint(0, 6, size=batch_size * patch_size[0] * patch_size[1])

    fn = Path("cm.png")

    visutils.make_confusion_matrix(
        pred_1, labels_1, exp_classes, filename=fn, save=True
    )

    visutils.make_confusion_matrix(pred_2, labels_1, exp_classes)

    fn.unlink(missing_ok=True)


def test_format_names() -> None:
    timestamp = "01-01-1970"
    model_name = "tester"
    path = ["test", "path"]
    names = visutils.format_plot_names(model_name, timestamp, path)

    filenames = {
        "History": Path(f"test/path/{model_name}_{timestamp}_MH.png"),
        "Pred": Path(f"test/path/{model_name}_{timestamp}_TP.png"),
        "CM": Path(f"test/path/{model_name}_{timestamp}_CM.png"),
        "ROC": Path(f"test/path/{model_name}_{timestamp}_ROC.png"),
        "Mask": Path(f"test/path/Masks/{model_name}_{timestamp}_Mask"),
        "PvT": Path(f"test/path/PvTs/{model_name}_{timestamp}_PvT"),
        "TSNE": Path(f"test/path/{model_name}_{timestamp}_TSNE.png"),
    }

    assert filenames == names


def test_plot_results(default_dataset: GeoDataset, exp_classes: Dict[int, str], exp_cmap_dict: Dict[int, str]) -> None:
    batch_size = 2
    patch_size = (32, 32)
    n_classes = 8

    plots = {
        "History": True,
        "Pred": True,
        "CM": True,
        "ROC": True,
        "micro": True,
        "macro": True,
        "Mask": False,
        "TSNE": True,
    }
    z = np.random.randint(0, n_classes, size=batch_size * patch_size[0] * patch_size[1])
    y = np.random.randint(0, n_classes, size=batch_size * patch_size[0] * patch_size[1])

    train_loss = {"x": list(range(1, 11)), "y": np.random.rand(10)}
    train_acc = {"x": list(range(1, 11)), "y": np.random.rand(10)}

    val_loss = {"x": list(range(1, 11)), "y": np.random.rand(10)}
    val_acc = {"x": list(range(1, 11)), "y": np.random.rand(10)}

    metrics = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }

    probs = np.random.rand(batch_size, *patch_size, len(exp_classes))

    embeddings = torch.rand([4, 152]).numpy()
    bounds = np.array(
        [get_random_bounding_box(default_dataset.bounds, 12.0, 1.0) for _ in range(4)]
    )

    visutils.plot_results(
        plots,
        z,
        y,
        metrics,
        probs=probs,
        bounds=bounds,
        embeddings=embeddings,
        task_name="test-test",
        class_names=exp_classes,
        colours=exp_cmap_dict,
        save=False,
    )


def test_plot_embeddings(results_root: Path, default_dataset: GeoDataset) -> None:
    embeddings = torch.rand([4, 152])
    bounds = [
        get_random_bounding_box(default_dataset.bounds, 12.0, 1.0) for _ in range(4)
    ]

    visutils.plot_embedding(
        embeddings,
        bounds,
        "test-test",
        show=True,
        filename=results_root / "tsne_cluster_vis.png",
        title="test_plot",
    )
