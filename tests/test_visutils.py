from datetime import date
from minerva.utils import visutils, utils
import os
import tempfile
import shutil
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from rasterio.crs import CRS
from torchgeo.datasets.utils import BoundingBox
from matplotlib.colors import ListedColormap
from matplotlib.image import AxesImage


def test_de_interlace() -> None:
    x_1 = [1, 1, 1, 1, 1]
    x_2 = [2, 2, 2, 2, 2]
    x_3 = [3, 3, 3, 3, 3]

    x = [x_1, x_2, x_3, x_1, x_2, x_3]

    x2 = np.array([x_1, x_1, x_2, x_2, x_3, x_3]).flatten()

    assert assert_array_equal(visutils.de_interlace(x, 3), x2) is None


def test_dec_extent_to_deg() -> None:
    shape = [224, 224]
    new_crs = CRS.from_epsg(26918)
    bounds = BoundingBox(
        -1.4153283567520825,
        -1.3964510733477618,
        50.91896360773007,
        50.93781998522083,
        1.0,
        2.0,
    )

    corners, lat, lon = visutils.dec_extent_to_deg(
        shape, bounds, src_crs=visutils.WGS84, new_crs=new_crs
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


def test_discrete_heatmap() -> None:
    data = np.random.randint(0, 7, size=(224, 224))
    cmap = ListedColormap(utils.CMAP_DICT.values())
    assert (
        visutils.discrete_heatmap(data, utils.CLASSES.values(), cmap_style=cmap) is None
    )


def test_stack_rgb() -> None:
    red = np.array([[25.0, 12.0, 11.0], [34.0, 55.0, 89.0], [23.0, 18.0, 76.0]])

    blue = np.array([[16.0, 17.0, 18.0], [19.0, 23.0, 24.0], [78.0, 67.0, 54.0]])

    green = np.array([[3.0, 2.0, 1.0], [9.0, 11.0, 34.0], [23.0, 15.0, 128.0]])

    image_1 = np.array([red, green, blue])
    rgb_1 = {"R": 0, "G": 1, "B": 2}

    image_2 = np.array([blue, red, green])
    rgb_2 = {"G": 2, "B": 0, "R": 1}

    correct = np.dstack((blue, green, red)) / 255.0
    result_1 = visutils.stack_rgb(image_1, rgb_1, max_value=255)
    result_2 = visutils.stack_rgb(image_2, rgb_2, max_value=255)

    assert assert_array_equal(result_1, correct) is None
    assert assert_array_equal(result_2, correct) is None


def test_make_rgb_image() -> None:
    image = np.random.rand(3, 224, 224)
    rgb = {"R": 0, "G": 1, "B": 2}

    assert type(visutils.make_rgb_image(image, rgb)) is AxesImage


def test_labelled_rgb_image() -> None:
    image = np.random.rand(224, 224, 3)
    mask = np.random.randint(0, 7, size=(224, 224))
    bounds = BoundingBox(
        -1.4153283567520825,
        -1.3964510733477618,
        50.91896360773007,
        50.93781998522083,
        1.0,
        2.0,
    )

    path = tempfile.gettempdir()
    if not isinstance(path, str):
        path = os.path.join(*path)

    name = "pretty_pic"
    cmap = ListedColormap(utils.CMAP_DICT.values())

    fn = visutils.labelled_rgb_image(
        image,
        mask,
        bounds,
        visutils.WGS84,
        path,
        name,
        utils.CLASSES.values(),
        cmap_style=cmap,
    )

    correct_fn = os.path.join(path, name) + "_RGBHM.png"

    assert fn == correct_fn


def test_make_gif() -> None:
    dates = ["2018-01-15", "2018-07-03", "2018-11-30"]
    images = np.random.rand(3, 32, 32, 3)
    masks = np.random.randint(0, 7, size=(3, 32, 32))
    bounds = BoundingBox(
        -1.4153283567520825,
        -1.3964510733477618,
        50.91896360773007,
        50.93781998522083,
        1.0,
        2.0,
    )

    path = os.getcwd()
    if isinstance(path, str):
        path = os.path.join(path, "tmp")
    else:
        path = os.path.join(*path, "tmp")

    os.makedirs(path, exist_ok=True)

    # Creates the GIF filename.
    gif_fn = f"{path}/new_gif.gif"

    cmap = ListedColormap(utils.CMAP_DICT.values())

    assert (
        visutils.make_gif(
            dates,
            images,
            masks,
            bounds,
            visutils.WGS84,
            utils.CLASSES.values(),
            gif_fn,
            path,
            cmap,
        )
        is None
    )

    # CLean up.
    shutil.rmtree(path)


def test_prediction_plot() -> None:
    image = np.random.rand(224, 224, 3)
    mask = np.random.randint(0, 7, size=(224, 224))
    pred = np.random.randint(0, 7, size=(224, 224))
    bounds = BoundingBox(
        -1.4153283567520825,
        -1.3964510733477618,
        50.91896360773007,
        50.93781998522083,
        1.0,
        2.0,
    )

    src_crs = utils.WGS84

    sample = {"image": image, "mask": mask, "pred": pred, "bounds": bounds}
    assert visutils.prediction_plot(sample, "101", utils.CLASSES, src_crs) is None


"""
def test_seg_plot() -> None:
    z = np.random.randint(0, 7, size=25*224*224)
    y = np.random.randint(0, 7, size=25*224*224)
    ids = ["ID"] * 25
    bbox = BoundingBox(
        -1.4153283567520825,
        -1.3964510733477618,
        50.91896360773007,
        50.93781998522083,
        1.0,
        2.0,
    )

    bounds = [bbox] * 25

    assert visutils.seg_plot(z, y, ids, bounds, ) is None
"""


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

    filename = "plot.png"

    assert visutils.plot_history(metrics, filename, show=True) is None

    os.remove(filename)


def test_make_confusion_matrix() -> None:
    pred_1 = np.random.randint(0, 8, size=16 * 224 * 224)
    labels_1 = np.random.randint(0, 8, size=16 * 224 * 224)

    pred_2 = np.random.randint(0, 6, size=16 * 224 * 224)

    fn = "cm.png"

    assert (
        visutils.make_confusion_matrix(
            pred_1, labels_1, utils.CLASSES, filename=fn, save=True
        )
        is None
    )

    assert visutils.make_confusion_matrix(pred_2, labels_1, utils.CLASSES) is None

    os.remove(fn)


def test_format_names() -> None:
    timestamp = "01-01-1970"
    model_name = "tester"
    path = ["test", "path"]
    names = visutils.format_plot_names(model_name, timestamp, path)

    filenames = {
        "History": f"test/path/{model_name}_{timestamp}_MH.png",
        "Pred": f"test/path/{model_name}_{timestamp}_TP.png",
        "CM": f"test/path/{model_name}_{timestamp}_CM.png",
        "ROC": f"test/path/{model_name}_{timestamp}_ROC.png",
        "Mask": f"test/path/Masks/{model_name}_{timestamp}_Mask",
        "PvT": f"test/path/PvTs/{model_name}_{timestamp}_PvT",
    }

    assert filenames == names


def test_plot_results() -> None:
    plots = {
        "History": True,
        "Pred": True,
        "CM": True,
        "ROC": True,
        "micro": True,
        "macro": True,
        "Mask": False,
    }
    z = np.random.randint(0, 8, size=16 * 224 * 224)
    y = np.random.randint(0, 8, size=16 * 224 * 224)

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

    probs = np.random.rand(16, 224, 224, len(utils.CLASSES))

    assert (
        visutils.plot_results(
            plots,
            z,
            y,
            metrics,
            probs=probs,
            class_names=utils.CLASSES,
            colours=utils.CMAP_DICT,
            save=False,
        )
        is None
    )
