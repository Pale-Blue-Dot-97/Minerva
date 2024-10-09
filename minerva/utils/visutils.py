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
#
"""Module to visualise .tiff images, label masks and results from the fitting of neural networks for remote sensing.

Attributes:
    WGS84 (~rasterio.crs.CRS): WGS84 co-ordinate reference system acting as a
        default :class:`~rasterio.crs.CRS` for transformations.
"""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "WGS84",
    "de_interlace",
    "dec_extent_to_deg",
    "get_mlp_cmap",
    "discrete_heatmap",
    "stack_rgb",
    "make_rgb_image",
    "labelled_rgb_image",
    "make_gif",
    "prediction_plot",
    "seg_plot",
    "plot_subpopulations",
    "plot_history",
    "make_confusion_matrix",
    "make_roc_curves",
    "plot_embedding",
    "format_plot_names",
    "plot_results",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
import random
from pathlib import Path
from typing import Any, Optional, Sequence

import imageio
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch
from geopy.exc import GeocoderUnavailable
from matplotlib import offsetbox
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from nptyping import Float, Int, NDArray, Shape
from numpy.typing import ArrayLike
from omegaconf import OmegaConf
from rasterio.crs import CRS
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay, multilabel_confusion_matrix
from torchgeo.datasets import GeoDataset, NonGeoDataset
from torchgeo.datasets.utils import BoundingBox
from tqdm import tqdm, trange

from minerva.utils import universal_path, utils

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
WGS84 = CRS.from_epsg(4326)

# Automatically fixes the layout of the figures to accommodate the colour bar legends.
plt.rcParams["figure.constrained_layout.use"] = True

# Increases DPI to avoid strange plotting errors for class heatmaps.
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# Removes margin in x-axis of plots.
plt.rcParams["axes.xmargin"] = 0

# Filters out all TensorFlow messages other than errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_MAX_SAMPLES = 25


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def de_interlace(x: Sequence[Any], f: int) -> NDArray[Any, Any]:
    """Separates interlaced arrays, ``x`` at a frequency of ``f`` from each other.

    Args:
        x (~typing.Sequence[~typing.Any]): Array of data to be de-interlaced.
        f (int): Frequency at which interlacing occurs. Equivalent to number of sources interlaced together.

    Returns:
        ~numpy.ndarray[~typing.Any]: De-interlaced array. Each source array is now sequentially connected.
    """
    new_x: list[NDArray[Any, Any]] = []
    for i in range(f):
        x_i = []
        for j in np.arange(start=i, stop=len(x), step=f):
            x_i.append(x[j])
        new_x.append(np.array(x_i).flatten())

    return np.array(new_x).flatten()


def dec_extent_to_deg(
    shape: tuple[int, int],
    bounds: BoundingBox,
    src_crs: CRS,
    new_crs: CRS = WGS84,
    spacing: int = 32,
) -> tuple[tuple[int, int, int, int], NDArray[Any, Float], NDArray[Any, Float]]:
    """Gets the extent of the image with ``shape`` and with ``bounds`` in latitude, longitude of system ``new_crs``.

    Args:
        shape (tuple[int, int]): 2D shape of image to be used to define the extents of the composite image.
        bounds (~torchgeo.datasets.utils.BoundingBox): Object describing a geospatial bounding box.
            Must contain ``minx``, ``maxx``, ``miny`` and ``maxy`` parameters.
        src_crs (~rasterio.crs.CRS): Source co-ordinate reference system (CRS).
        new_crs (~rasterio.crs.CRS): Optional; The co-ordinate reference system (CRS) to transform to.
        spacing (int): Spacing of the lat - lon ticks.

    Returns:
        tuple[tuple[int, int, int, int], ~numpy.ndarray[float], ~numpy.ndarray[float]]:
            * The corners of the image in pixel co-ordinates e.g. ``(0, 256, 0, 256)``.
            * The latitude extent of the image with ticks at intervals defined by ``spacing``.
            * The longitude extent of the image with ticks at intervals defined by ``spacing``.
    """
    # Defines the 'extent' for a composite image based on the size of shape.
    extent = 0, shape[0], 0, shape[1]

    # Gets the co-ordinates of the corners of the image in decimal lat-lon.
    corners = utils.transform_coordinates(
        x=[bounds.minx, bounds.maxx],
        y=[bounds.miny, bounds.maxy],
        src_crs=src_crs,
        new_crs=new_crs,
    )

    # Creates a discrete mapping of the spaced ticks to latitude longitude extent of the image.
    lat_extent = np.around(
        np.linspace(
            start=corners[1][0],
            stop=corners[1][1],
            num=int(shape[0] / spacing) + 1,
            endpoint=True,
        ),
        decimals=3,
    )
    lon_extent = np.around(
        np.linspace(
            start=corners[0][0],
            stop=corners[0][1],
            num=int(shape[0] / spacing) + 1,
            endpoint=True,
        ),
        decimals=3,
    )

    return extent, lat_extent, lon_extent


def get_mlp_cmap(
    cmap_style: Optional[Colormap | str] = None, n_classes: Optional[int] = None
) -> Optional[Colormap]:
    """Creates a cmap from query

    Args:
        cmap_style (~matplotlib.colors.Colormap | str): Optional; :mod:`matplotlib` colourmap style to get.
        n_classes (int): Optional; Number of classes in data to assign colours to.

    Returns:
        ~matplotlib.colors.Colormap | None:
        * If ``cmap_style`` and ``n_classes`` provided, returns a :class:`~matplotlib.colors.ListedColormap` instance.
        * If ``cmap_style`` provided but no ``n_classes``, returns a :class:`~matplotlib.colors.Colormap` instance.
        * If neither arguments are provided, ``None`` is returned.
    """
    cmap: Optional[Colormap] = None

    if cmap_style:
        if isinstance(cmap_style, str):
            cmap = mlp.colormaps[cmap_style]  # type: ignore
        else:
            cmap = cmap_style

        if n_classes:
            assert isinstance(cmap, Colormap)
            cmap = cmap.resampled(n_classes)  # type: ignore

    return cmap


def discrete_heatmap(
    data: NDArray[Shape["*, *"], Int],  # noqa: F722
    classes: list[str] | tuple[str, ...],
    cmap_style: Optional[str | ListedColormap] = None,
    block_size: int = 32,
) -> None:
    """Plots a heatmap with a discrete colour bar. Designed for Radiant Earth MLHub 256x256 SENTINEL images.

    Args:
        data (~numpy.ndarray[int]): 2D Array of data to be plotted as a heat map.
        classes (list[str]): Optional; List of all possible class labels.
        cmap_style (str | ~matplotlib.colors.ListedColormap): Optional; Name or object for colour map style.
        block_size (int): Optional; Size of block image subdivision in pixels.
    """
    # Initialises a figure.
    plt.figure()

    # Creates a cmap from query.
    cmap = get_mlp_cmap(cmap_style, len(classes))

    # Plots heatmap onto figure.
    heatmap = plt.imshow(data, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5)  # type: ignore[arg-type]

    # Sets tick intervals to block size. Default 32 x 32.
    plt.xticks(np.arange(0, data.shape[0] + 1, block_size))
    plt.yticks(np.arange(0, data.shape[1] + 1, block_size))

    # Add grid overlay.
    plt.grid(which="both", color="#CCCCCC", linestyle=":")

    # Plots colour bar onto figure.
    clb = plt.colorbar(heatmap, ticks=np.arange(0, len(classes)), shrink=0.77)

    # Sets colour bar ticks to class labels.
    clb.ax.set_yticklabels(classes)

    # Display figure.
    plt.show(block=False)

    # Close figure.
    plt.close()


def stack_rgb(
    image: NDArray[Shape["3, *, *"], Float],  # noqa: F722
    max_value: int = 255,
) -> NDArray[Shape["*, *, 3"], Float]:  # noqa: F722
    """Stacks together red, green and blue image bands to create a RGB array.

    Args:
        image (~numpy.ndarray[float]): Image of separate channels to be normalised
            and reshaped into stacked RGB image.
        max_value (int): Optional; The maximum pixel value in ``image``. e.g. for 8 bit this will be 255.

    Returns:
        ~numpy.ndarray[float]: Normalised and stacked red, green, blue arrays into RGB array.
    """
    # Stack together RGB bands. Assumes RGB bands are in dimensions 0-2. Ignores any other bands.
    # Note that it has to be order BGR not RGB due to the order numpy stacks arrays.
    rgb_image: NDArray[Shape["3, *, *"], Any] = np.dstack(  # noqa: F722
        (image[2], image[1], image[0])
    )
    assert isinstance(rgb_image, np.ndarray)

    # Normalise.
    return rgb_image / max_value


def make_rgb_image(
    image: NDArray[Shape["3, *, *"], Float],  # noqa: F722
    block_size: int = 32,
    max_pixel_value: int = 255,
) -> AxesImage:
    """Creates an RGB image from a composition of red, green and blue bands.

    Args:
        image (~numpy.ndarray[int]): Array representing the image of shape ``(bands x height x width)``.
        block_size (int): Optional; Size of block image sub-division in pixels.
        max_value (int): Optional; The maximum pixel value in ``image``. e.g. for 8 bit this will be 255.

    Returns:
        ~matplotlib.image.AxesImage: Plotted RGB image object.
    """
    # Stack RGB image data together.
    rgb_image_array = stack_rgb(image, max_pixel_value)

    # Create RGB image.
    rgb_image = plt.imshow(rgb_image_array)

    # Sets tick intervals to block size. Default 32 x 32.
    plt.xticks(np.arange(0, rgb_image_array.shape[0] + 1, block_size))
    plt.yticks(np.arange(0, rgb_image_array.shape[1] + 1, block_size))

    # Add grid overlay.
    plt.grid(which="both", color="#CCCCCC", linestyle=":")

    plt.show(block=False)

    return rgb_image


def labelled_rgb_image(
    image: NDArray[Shape["*, *, 3"], Float],  # noqa: F722
    mask: NDArray[Shape["*, *"], Int],  # noqa: F722
    bounds: BoundingBox,
    src_crs: CRS,
    path: str | Path,
    name: str,
    classes: list[str] | tuple[str, ...],
    cmap_style: Optional[str | ListedColormap] = None,
    new_crs: Optional[CRS] = WGS84,
    block_size: int = 32,
    alpha: float = 0.5,
    show: bool = True,
    save: bool = True,
    figdim: tuple[int | float, int | float] = (8.02, 10.32),
) -> Path:
    """Produces a layered image of an RGB image, and it's associated label mask heat map alpha blended on top.

    Args:
        image (~numpy.ndarray[int]): Array representing the image of shape ``(height x width x bands)``.
        mask (~numpy.ndarray[int]): Ground truth mask. Should be of shape (height x width) matching ``image``.
        bounds (~torchgeo.datasets.utils.BoundingBox): Object describing a geospatial bounding box.
            Must contain ``minx``, ``maxx``, ``miny`` and ``maxy`` parameters.
        src_crs (~rasterio.crs.CRS): Source co-ordinate reference system (CRS).
        path (str): Path to where to save created figure.
        name (str): Name of figure. Will be used for title and in the filename.
        classes (list[str]): Optional; List of all possible class labels.
        cmap_style (str | ~matplotlib.colors.ListedColormap): Optional; Name or object for colour map style.
        new_crs (~rasterio.crs.CRS): Optional; The co-ordinate reference system (CRS) to transform to.
        block_size (int): Optional; Size of block image subdivision in pixels.
        alpha (float): Optional; Fraction determining alpha blending of label mask.
        show (bool): Optional; Show the figure when plotted.
        save (bool): Optional; Save the figure to ``path``.
        figdim (tuple[int | float, int | float]): Optional; Figure (height, width) in inches.

    Returns:
        str: Path to figure save location.
    """
    # Checks that the mask and image shapes will align.
    mask_shape: tuple[int, int] = mask.shape  # type: ignore[assignment]
    assert mask_shape == image.shape[:2]

    assert new_crs is not None

    # Gets the extent of the image in pixel, lattitude and longitude dimensions.
    extent, lat_extent, lon_extent = dec_extent_to_deg(
        mask_shape,
        bounds=bounds,
        src_crs=src_crs,
        spacing=block_size,
        new_crs=new_crs,
    )

    # Initialises a figure.
    fig, ax1 = plt.subplots()

    # Create RGB image.
    ax1.imshow(image, extent=extent)

    # Creates a cmap from query.
    cmap = get_mlp_cmap(cmap_style, len(classes))

    # Plots heatmap onto figure.
    heatmap = ax1.imshow(
        mask,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(classes) - 0.5,
        extent=extent,
        alpha=alpha,  # type: ignore[arg-type]
    )

    # Sets tick intervals to standard 32x32 block size.
    ax1.set_xticks(np.arange(0, mask.shape[0] + 1, block_size))
    ax1.set_yticks(np.arange(0, mask.shape[1] + 1, block_size))

    # Creates a secondary x and y-axis to hold lat-lon.
    ax2 = ax1.twiny().twinx()

    # Plots an invisible line across the diagonal of the image to create the secondary axis for lat-lon.
    ax2.plot(
        lon_extent,
        lat_extent,
        " ",
        clip_box=Bbox.from_extents(
            lon_extent[0], lat_extent[0], lon_extent[-1], lat_extent[-1]
        ),
    )

    # Set ticks for lat-lon.
    ax2.set_xticks(lon_extent)
    ax2.set_yticks(lat_extent)

    # Sets the limits of the secondary axis, so they should align with the primary.
    ax2.set_xlim(left=lon_extent[0], right=lon_extent[-1])
    ax2.set_ylim(top=lat_extent[-1], bottom=lat_extent[0])

    # Converts the decimal lat-lon into degrees, minutes, seconds to label the axis.
    lat_labels = utils.dec2deg(lat_extent, axis="lat")
    lon_labels = utils.dec2deg(lon_extent, axis="lon")

    # Sets the secondary axis tick labels.
    ax2.set_xticklabels(lon_labels, fontsize=11)
    ax2.set_yticklabels(lat_labels, fontsize=10, rotation=-30, ha="left")

    # Add grid overlay.
    ax1.grid(which="both", color="#CCCCCC", linestyle=":")

    # Plots colour bar onto figure.
    clb = plt.colorbar(
        heatmap, ticks=np.arange(0, len(classes)), shrink=0.9, aspect=75, drawedges=True
    )

    # Sets colour bar ticks to class labels.
    clb.ax.set_yticklabels(classes, fontsize=11)

    # Bodge to get a figure title by using the colour bar title.
    clb.ax.set_title(f"{name}\nLand Cover", loc="left", fontsize=15)

    # Set axis labels.
    ax1.set_xlabel("(x) - Pixel Position", fontsize=14)
    ax1.set_ylabel("(y) - Pixel Position", fontsize=14)
    ax2.set_ylabel("Latitude", fontsize=14, rotation=270, labelpad=12)
    ax2.set_title("Longitude")  # Bodge

    # Manual trial and error fig size which fixes aspect ratio issue.
    fig.set_figheight(figdim[0])
    fig.set_figwidth(figdim[1])

    # Display figure.
    if show:
        plt.show(block=False)

    # Path and file name of figure.
    fn = Path(f"{path}/{name}_RGBHM.png")

    # If true, save file to fn.
    if save:
        # Checks if file already exists. Deletes if true.
        utils.exist_delete_check(fn)

        # Save figure to fn.
        fig.savefig(fn)

    # Close figure.
    plt.close()

    return fn


def make_gif(
    dates: Sequence[str],
    images: NDArray[Shape["*, *, *, 3"], Any],  # noqa: F722
    masks: NDArray[Shape["*, *, *"], Any],  # noqa: F722
    bounds: BoundingBox,
    src_crs: CRS,
    classes: list[str] | tuple[str, ...],
    gif_name: str,
    path: str | Path,
    cmap_style: Optional[str | ListedColormap] = None,
    fps: float = 1.0,
    new_crs: Optional[CRS] = WGS84,
    alpha: float = 0.5,
    figdim: tuple[int | float, int | float] = (8.02, 10.32),
) -> None:
    """Wrapper to :func:`labelled_rgb_image` to make a GIF for a patch out of scenes.

    Args:
        dates (~typing.Sequence[str]): Dates of scenes to be used as the frames in the GIF.
        images (~numpy.ndarray[~typing.Any]): All the frames of imagery to make the GIF from.
            Leading dimension must be the same length as ``dates`` and ``masks``.
        masks (~numpy.ndarray[~typing.Any]): The masks for each frame of the GIF.
            Leading dimension must be the same length as ``dates`` and ``image``.
        bounds (~torchgeo.datasets.utils.BoundingBox): The bounding box (in the ``src_crs`` CRS) of the
            :term:`patch` the ``GIF`` will be of.
        src_crs (~rasterio.crs.CRS): Source co-ordinate reference system (CRS).
        classes (list[str]): List of all possible class labels.
        gif_name (str): Path to and name of GIF to be made.
        path (~pathlib.Path | str]): Path to where to save frames of the ``GIF``.
        cmap_style (str | ~matplotlib.colors.ListedColormap): Optional; Name or object for colour map style.
        fps (float): Optional; Frames per second of ``GIF``.
        new_crs (~rasterio.crs.CRS): Optional; The co-ordinate reference system (CRS) to transform to.
        alpha (float): Optional; Fraction determining alpha blending of label mask.
        figdim (tuple[int | float, int | float]): Optional; Figure (height, width) in inches.

    Returns:
        None
    """
    # Changes to `imagio` now mean we need the duration of the GIF and not the `fps`.
    duration = len(dates) / fps

    # List to hold filenames and paths of images created.
    frames = []

    with trange(len(dates)) as t:
        for i in t:
            # Update progress bar with current scene.
            t.set_description("SCENE ON %s" % dates[i])

            # Create a frame of the GIF for a scene of the patch.
            frame = labelled_rgb_image(
                images[i],
                masks[i],
                bounds,
                src_crs,
                path,
                name=f"{i}",
                classes=classes,
                cmap_style=cmap_style,
                new_crs=new_crs,
                alpha=alpha,
                save=True,
                show=False,
                figdim=figdim,
            )

            # Read in frame just created and add to list of frames.
            frames.append(imageio.imread(frame))

    # Checks GIF doesn't already exist. Deletes if it does.
    utils.exist_delete_check(gif_name)

    print("MAKING PATCH GIF")

    # Create GIF.
    imageio.mimwrite(gif_name, frames, format=".gif", duration=duration)  # type: ignore


def prediction_plot(
    sample: dict[str, Any],
    sample_id: str,
    classes: dict[int, str],
    src_crs: CRS,
    new_crs: CRS = WGS84,
    path: str = "",
    cmap_style: Optional[str | ListedColormap] = None,
    exp_id: Optional[str] = None,
    fig_dim: Optional[tuple[int | float, int | float]] = None,
    block_size: int = 32,
    show: bool = True,
    save: bool = True,
    fn_prefix: Optional[str | Path] = None,
) -> None:
    """
    Produces a figure containing subplots of the predicted label mask, the ground truth label mask
    and a reference RGB image of the same patch.

    Args:
        sample (dict[str, ~typing.Any]): Dictionary holding the ``"image"``, ground truth (``"mask"``)
            and predicted (``"prediction"``) masks and the bounding box for this sample.
        sample_id (str): ID for the sample.
        classes (dict[int, str]): Dictionary mapping class labels to class names.
        src_crs (~rasterio.crs.CRS): Existing co-ordinate system of the image.
        new_crs(~rasterio.crs.CRS): Optional; Co-ordinate system to convert image to and use for labelling.
        exp_id (str): Optional; Unique ID for the experiment run that predictions and labels come from.
        block_size (int): Optional; Size of block image sub-division in pixels.
        cmap_style (str | ~matplotlib.colors.ListedColormap): Optional; Name or object for colour map style.
        show (bool): Optional; Show the figure when plotted.
        save (bool): Optional; Save the figure to file to ``fn_prefix``.
        fig_dim (tuple[float, float]): Optional; Figure (height, width) in inches.
        fn_prefix (str | ~pathlib.Path): Optional; Common filename prefix (including path to file) for all plots of
            this type from this experiment. Appended with the sample ID to give the filename to save the plot to.

    Returns:
        None
    """
    # Stacks together the R, G, & B bands to form an array of the RGB image.
    rgb_image = sample["image"]
    z = sample["prediction"]
    y = sample["mask"]
    bounds = sample["index"]

    extent, lat_extent, lon_extent = dec_extent_to_deg(
        y.shape, bounds, src_crs, new_crs=new_crs, spacing=block_size
    )

    centre = utils.transform_coordinates(
        *utils.get_centre_loc(bounds), src_crs=src_crs, new_crs=new_crs
    )

    # Initialises a figure.
    fig: Figure = plt.figure(figsize=fig_dim)

    gs = GridSpec(nrows=2, ncols=2, figure=fig)

    axes: NDArray[Shape["3"], Axes] = np.array(  # type: ignore[type-var, assignment]
        [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :]),
        ]
    )

    cmap = get_mlp_cmap(cmap_style, len(classes))

    # Plots heatmap onto figure.
    z_heatmap = axes[0].imshow(z, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5)
    _ = axes[1].imshow(y, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5)

    # Create RGB image.
    axes[2].imshow(rgb_image, extent=extent)

    # Sets tick intervals to standard 32x32 block size.
    axes[0].set_xticks(np.arange(0, z.shape[0] + 1, block_size))
    axes[0].set_yticks(np.arange(0, z.shape[1] + 1, block_size))

    axes[1].set_xticks(np.arange(0, y.shape[0] + 1, block_size))
    axes[1].set_yticks(np.arange(0, y.shape[1] + 1, block_size))

    axes[2].set_xticks(np.arange(0, rgb_image.shape[0] + 1, block_size))
    axes[2].set_yticks(np.arange(0, rgb_image.shape[1] + 1, block_size))

    # Add grid overlay.
    axes[0].grid(which="both", color="#CCCCCC", linestyle=":")
    axes[1].grid(which="both", color="#CCCCCC", linestyle=":")
    axes[2].grid(which="both", color="#CCCCCC", linestyle=":")

    # Converts the decimal lat-lon into degrees, minutes, seconds to label the axis.
    lat_labels = utils.dec2deg(lat_extent, axis="lat")
    lon_labels = utils.dec2deg(lon_extent, axis="lon")

    # Sets the secondary axis tick labels.
    axes[2].set_xticklabels(lon_labels, fontsize=9, rotation=30)
    axes[2].set_yticklabels(lat_labels, fontsize=9)

    # Plots colour bar onto figure.
    clb = fig.colorbar(
        z_heatmap,
        ax=axes.ravel().tolist(),
        location="top",
        ticks=np.arange(0, len(classes)),
        aspect=75,
        drawedges=True,
    )

    # Sets colour bar ticks to class labels.
    clb.ax.set_xticklabels(classes.values(), fontsize=9)

    # Set figure title and subplot titles.
    loc: str
    try:
        loc = utils.lat_lon_to_loc(lat=str(centre[1]), lon=str(centre[0]))
    except GeocoderUnavailable:  # pragma: no cover
        loc = ""

    fig.suptitle(
        f"{sample_id}: {loc}",
        fontsize=15,
    )
    axes[0].set_title("Predicted", fontsize=13)
    axes[1].set_title("Ground Truth", fontsize=13)
    axes[2].set_title("Reference Imagery", fontsize=13)

    # Set axis labels.
    axes[0].set_xlabel("(x) - Pixel Position", fontsize=10)
    axes[0].set_ylabel("(y) - Pixel Position", fontsize=10)
    axes[1].set_xlabel("(x) - Pixel Position", fontsize=10)
    axes[1].set_ylabel("(y) - Pixel Position", fontsize=10)
    axes[2].set_xlabel("Longitude", fontsize=10)
    axes[2].set_ylabel("Latitude", fontsize=10)

    # Display figure.
    if show:
        plt.show(block=False)

    if fn_prefix is None:
        _path = universal_path(path)
        fn_prefix = str(_path / f"{exp_id}_{utils.timestamp_now()}_Mask")

    # Path and file name of figure.
    fn = Path(f"{fn_prefix}_{sample_id}.png").absolute()

    # If true, save file to fn.
    if save:
        # Checks if file already exists. Deletes if true.
        utils.exist_delete_check(fn)

        # Save figure to fn.
        fig.savefig(fn)

    # Close figure.
    plt.close()


def seg_plot(
    z: list[int] | NDArray[Any, Any],
    y: list[int] | NDArray[Any, Any],
    ids: list[str],
    index: Sequence[Any] | NDArray[Any, Any],
    data_dir: Path | str,
    dataset_params: dict[str, Any],
    classes: dict[int, str],
    colours: dict[int, str],
    fn_prefix: Optional[str | Path],
    x: Optional[list[int] | NDArray[Any, Any]] = None,
    frac: float = 0.05,
    fig_dim: Optional[tuple[int | float, int | float]] = (9.3, 10.5),
    model_name: str = "",
    path: str = "",
    max_pixel_value: int = 255,
    cache_dir: Optional[str | Path] = None,
) -> None:
    """Custom function for pre-processing the outputs from image segmentation testing for data visualisation.

    Args:
        z (list[float]): Predicted segmentation masks by the network.
        y (list[float]): Corresponding ground truth masks.
        ids (list[str]): Corresponding patch IDs for the test data supplied to the network.
        bounds (list[~torchgeo.datasets.utils.BoundingBox] | ~numpy.ndarray[~torchgeo.datasets.utils.BoundingBox]):
            Array of objects describing a geospatial bounding box.
            Must contain ``minx``, ``maxx``, ``miny`` and ``maxy`` parameters.
        task_name (str): Name of the task that samples are from.
        classes (dict[int, str]): Dictionary mapping class labels to class names.
        colours (dict[int, str]): Dictionary mapping class labels to colours.
        fn_prefix (str | ~pathlib.Path): Common filename prefix (including path to file) for all plots of this type
            from this experiment to use.
        frac (float): Optional; Fraction of patch samples to plot.
        fig_dim (tuple[float, float]): Optional; Figure (height, width) in inches.
        cache_dir (str | ~pathlib.Path): Optional; Path to the directory to load the cached dataset from.
            Defaults to None (so will create dataset from scratch).

    Returns:
        None
    """
    # TODO: This is a very naughty way of avoiding a circular import.
    # Need to reorganise package to avoid need for this.
    from minerva.datasets import make_dataset

    if not isinstance(z, np.ndarray):
        z = np.array(z)

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    z = np.reshape(z, (z.shape[0] * z.shape[1], z.shape[2], z.shape[3]))
    y = np.reshape(y, (y.shape[0] * y.shape[1], y.shape[2], y.shape[3]))
    flat_ids: NDArray[Any, Any] = np.array(ids).flatten()

    print("\nRE-CONSTRUCTING DATASET")
    if cache_dir is not None:
        cache = True
    else:
        cache = False
        cache_dir = ""

    dataset, _ = make_dataset(
        data_dir, dataset_params, cache=cache, cache_dir=cache_dir
    )

    # Limits number of masks to produce to a fractional number of total and no more than `_MAX_SAMPLES`.
    n_samples = int(frac * len(flat_ids))
    if n_samples > _MAX_SAMPLES:
        n_samples = _MAX_SAMPLES

    print("\nPRODUCING PREDICTED MASKS")

    # Plots the predicted versus ground truth labels for all test patches supplied.
    with tqdm(total=n_samples) as pbar:
        for i in random.sample(range(len(flat_ids)), n_samples):
            if isinstance(dataset, GeoDataset):
                image = stack_rgb(
                    torch.Tensor(x[i])
                    if x is not None
                    else dataset[index[i]]["image"].numpy(),
                    max_pixel_value,
                )
                sample = {
                    "image": image,
                    "prediction": z[i],
                    "mask": y[i],
                    "index": index[i],
                }
                prediction_plot(
                    sample,
                    flat_ids[i],
                    classes=classes,
                    src_crs=dataset.crs,
                    exp_id=model_name,
                    show=False,
                    fn_prefix=fn_prefix,
                    fig_dim=fig_dim,
                    cmap_style=ListedColormap(colours.values(), N=len(colours)),  # type: ignore
                    path=path,
                )

            elif isinstance(dataset, NonGeoDataset) and hasattr(dataset, "plot"):
                sample = {
                    "image": torch.Tensor(x[i])
                    if x is not None
                    else dataset[index[i]]["image"],
                    "prediction": torch.LongTensor(z[i]),
                    "mask": torch.LongTensor(y[i]),
                    "index": index[i],
                }
                fig = dataset.plot(
                    sample,
                    show_titles=True,
                    suptitle=sample["index"],
                    classes=classes,
                    colours=colours,
                )

                if fn_prefix is None:
                    _path = universal_path(path)
                    fn_prefix = str(
                        _path / f"{model_name}_{utils.timestamp_now()}_Mask"
                    )

                sample_id = sample["index"]

                # Path and file name of figure.
                fn = Path(f"{fn_prefix}_{sample_id}.png").absolute()

                # Checks if file already exists. Deletes if true.
                utils.exist_delete_check(fn)

                # Save figure to fn.
                fig.savefig(fn)

                # Close figure.
                plt.close()
            else:
                raise NotImplementedError()

        pbar.update()


def plot_subpopulations(
    class_dist: list[tuple[int, int]],
    class_names: Optional[dict[int, str]] = None,
    cmap_dict: Optional[dict[int, str]] = None,
    filename: Optional[str | Path] = None,
    save: bool = True,
    show: bool = False,
) -> None:
    """Creates a pie chart of the distribution of the classes within the data.

    Args:
        class_dist (list[tuple[int, int]]): Modal distribution of classes in the dataset provided.
        class_names (dict[int, str]): Optional; Dictionary mapping class labels to class names.
        cmap_dict (dict[int, str]): Optional; Dictionary mapping class labels to class colours.
        filename (str): Optional; Name of file to save plot to.
        show (bool): Optional; Whether to show plot.
        save (bool): Optional; Whether to save plot to file.

    Returns:
        None
    """
    # List to hold the name and percentage distribution of each class in the data as str.
    class_data = []

    # List to hold the total counts of each class.
    counts = []

    # List to hold colours of classes in the correct order.
    colours: Optional[list[str]] = []

    if class_names is None:
        class_numbers = [x[0] for x in class_dist]
        class_names = {i: f"class {i}" for i in class_numbers}

    # Finds total number of samples to normalise data.
    n_samples = 0
    for mode in class_dist:
        n_samples += mode[1]

    # For each class, find the percentage of data that is that class and the total counts for that class.
    for label in class_dist:
        # Sets percentage label to <0.01% for classes matching that equality.
        if (label[1] * 100.0 / n_samples) > 0.01:
            class_data.append(
                "{} \n{:.2f}%".format(
                    class_names[label[0]], (label[1] * 100.0 / n_samples)
                )
            )
        else:
            class_data.append(f"{class_names[label[0]]} \n<0.01%")
        counts.append(label[1])

        if cmap_dict:
            assert colours is not None
            colours.append(cmap_dict[label[0]])

    if cmap_dict is None:
        colours = None

    # Locks figure size.
    plt.figure(figsize=(6, 5))

    # Plot a pie chart of the data distribution amongst the classes.
    patches, _ = plt.pie(  # type: ignore[misc]
        counts, colors=colours, explode=[i * 0.05 for i in range(len(class_data))]
    )

    # Adds legend.
    plt.legend(
        patches, class_data, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False
    )

    # Shows and/or saves plot.
    if show:
        plt.show(block=False)
    if save:
        plt.savefig(filename)
        plt.close()


def plot_history(
    metrics: dict[str, Any],
    filename: Optional[str | Path] = None,
    save: bool = True,
    show: bool = False,
) -> None:
    """Plots model history based on metrics supplied.

    Args:
        metrics (dict[str, ~typing.Any]): Dictionary containing the names and results of the metrics
            by which model was assessed.
        filename (str): Optional; Name of file to save plot to.
        show (bool): Optional; Whether to show plot.
        save (bool): Optional; Whether to save plot to file.

    Returns:
        None
    """
    # Initialise figure.
    ax = plt.figure().gca()

    # Plots each metric in metrics, appending their artist handles.
    handles = []
    labels = []
    for key in metrics:
        # Checks that the length of x matches y and is greater than 1 so can be plotted.
        if len(metrics[key]["x"]) == len(metrics[key]["y"]) >= 1.0:
            # Plot metric.
            handles.append(ax.plot(metrics[key]["x"], metrics[key]["y"])[0])
            labels.append(key)

    # Creates legend from plot artist handles and names of metrics.
    ax.legend(handles=handles, labels=labels)

    # Forces x-axis ticks to be integers.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adds a grid overlay with green dashed lines.
    ax.grid(color="green", linestyle="--", linewidth=0.5)  # For some funky gridlines

    # Adds axis labels.
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss/Accuracy")

    # Shows and/or saves plot.
    if show:
        plt.show(block=False)
    if save:
        plt.savefig(filename)
        plt.close()


def make_confusion_matrix(
    pred: list[int] | NDArray[Any, Int],
    labels: list[int] | NDArray[Any, Int],
    classes: dict[int, str],
    filename: Optional[str | Path] = None,
    cmap_style: str = "Blues",
    figsize: tuple[int, int] = (2, 2),
    show: bool = True,
    save: bool = False,
) -> None:
    """Creates a heat-map of the confusion matrix of the given model.

    Args:
        pred (list[int]): Predictions made by model on test images.
        labels (list[int]): Accompanying ground truth labels for testing images.
        classes (dict[int, str]): Dictionary mapping class labels to class names.
        filename (str): Optional; Name of file to save plot to.
        cmap_style (str): Colourmap style to use in the confusion matrix.
        show (bool): Optional; Whether to show plot.
        save (bool): Optional; Whether to save plot to file.

    Returns:
        None
    """
    _pred, _labels, new_classes = utils.check_test_empty(pred, labels, classes)

    # Extract class names from dict in numeric order to ensure labels match matrix.
    class_names = [new_classes[key] for key in range(len(new_classes.keys()))]

    # Creates the figure to plot onto.
    ax = plt.figure(figsize=figsize).gca()

    # Get a matplotlib colourmap based on the style specified to use for the confusion matrix.
    cmap = get_mlp_cmap(cmap_style)

    # Creates, plots and normalises the confusion matrix.
    cm = ConfusionMatrixDisplay.from_predictions(
        _labels,
        _pred,
        labels=list(new_classes.keys()),
        normalize="true",
        display_labels=class_names,
        cmap=cmap,
        ax=ax,
    )

    # Normalises the colourbar to between [0, 1] for consistent clarity.
    cm.ax_.get_images()[0].set_clim(0, 1)

    # Shows and/or saves plot.
    if show:
        plt.show(block=False)
    if save:
        plt.savefig(filename)
        plt.close()


def make_multilabel_confusion_matrix(
    preds: list[int] | NDArray[Any, Int],
    labels: list[int] | NDArray[Any, Int],
    classes: dict[int, str],
    filename: Optional[str | Path] = None,
    cmap_style: str = "Blues",
    figsize: tuple[int, int] = (2, 2),
    show: bool = True,
    save: bool = False,
) -> None:
    """Creates a heat-map of the confusion matrix of the given model.

    Args:
        probs (list[int]): Output made by the model on test images.
        labels (list[int]): Accompanying ground truth labels for testing images.
        classes (dict[int, str]): Dictionary mapping class labels to class names.
        filename (str): Optional; Name of file to save plot to.
        cmap_style (str): Colourmap style to use in the confusion matrix.
        show (bool): Optional; Whether to show plot.
        save (bool): Optional; Whether to save plot to file.

    Returns:
        None
    """
    # Find a pair of integer values that are most square-like for the dimensions
    # of the sub-plot array that fits with the number of classes.
    dimensions = utils.closest_factors(len(classes))

    # Creates the figure to plot onto.
    fig, axes = plt.subplots(dimensions[0], dimensions[1], figsize=figsize)
    axes = axes.ravel()

    # Get a matplotlib colourmap based on the style specified to use for the confusion matrix.
    cmap = get_mlp_cmap(cmap_style)

    if isinstance(labels, list):
        labels = np.ndarray(labels)
    if isinstance(preds, list):
        preds = np.ndarray(preds)

    # Create the confusion matrices for each class.
    cm = multilabel_confusion_matrix(
        labels.reshape(-1, labels.shape[-1]),
        preds.reshape(-1, preds.shape[-1]),
        labels=list(classes.keys()),
    )

    for i in range(len(classes)):
        # Creates the confusion matrix.
        sub_cm = ConfusionMatrixDisplay(cm[i], display_labels=["N", "Y"])

        # Plot confusion matrix.
        sub_cm.plot(cmap=cmap, ax=axes[i])

        # Set title for each sub-plot to the class number (labels will not fit).
        sub_cm.ax_.set_title(f"Class {i}")

        # Delete individual colourbars for each sub-plot.
        sub_cm.im_.colorbar.remove()

    # Add colourbar for whole figure.
    fig.colorbar(sub_cm.im_, ax=axes)

    # Delete empty sub_plots if the n_classes was an odd number > 5.
    for i in range(len(classes), np.prod(dimensions)):
        fig.delaxes(axes[i])

    # Shows and/or saves plot.
    if show:
        plt.show(block=False)
    if save:
        plt.savefig(filename)
        plt.close()


def make_roc_curves(
    probs: ArrayLike,
    labels: Sequence[int] | NDArray[Any, Int],
    class_names: dict[int, str],
    colours: dict[int, str],
    micro: bool = True,
    macro: bool = True,
    filename: Optional[str | Path] = None,
    show: bool = False,
    save: bool = True,
) -> None:
    """Plots ROC curves for each class, the micro and macro average ROC curves and accompanying AUCs.

    Adapted from Scikit-learn's example at:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Args:
        probs (list | ~numpy.ndarray[int]): Array of probabilistic predicted classes from model where each sample
            should have a list of the predicted probability for each class.
        labels (list | ~numpy.ndarray[int]): List of corresponding ground truth labels.
        class_names (dict[int, str]): Dictionary mapping class labels to class names.
        colours (dict[int, str]): Dictionary mapping class labels to colours.
        micro (bool): Optional; Whether to compute and plot the micro average ROC curves.
        macro (bool): Optional; Whether to compute and plot the macro average ROC curves.
        filename (str | ~pathlib.Path): Optional; Name of file to save plot to.
        save (bool): Optional; Whether to save the plots to file.
        show (bool): Optional; Whether to show the plots.

    Returns:
        None
    """
    # Gets the class labels as a list from the class_names dict.
    class_labels = [key for key in class_names.keys()]

    # Reshapes the probabilities to be (n_samples, n_classes).
    probs = np.reshape(probs, (len(labels), len(class_labels)))

    # Computes all class, micro and macro average ROC curves and AUCs.
    fpr, tpr, roc_auc = utils.compute_roc_curves(
        probs, labels, class_labels, micro=micro, macro=macro
    )

    # Plot all ROC curves
    print("\nPlotting ROC Curves")
    plt.figure()

    if micro:
        # Plot micro average ROC curves.
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="Micro-average (AUC = {:.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle="dotted",
        )

    if macro:
        # Plot macro average ROC curves.
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="Macro-average (AUC = {:.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle="dotted",
        )

    # Plot all class ROC curves.
    for key in class_labels:
        try:
            plt.plot(
                fpr[key],
                tpr[key],
                color=colours[key],
                label=f"{class_names[key]} " + "(AUC = {:.2f})".format(roc_auc[key]),
            )
        except KeyError:
            pass

    # Plot random classifier diagonal.
    plt.plot([0, 1], [0, 1], "k--")

    # Set limits.
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Set axis labels.
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Position legend in lower right corner of figure where no classifiers should exist.
    plt.legend(loc="lower right")

    # Shows and/or saves plot.
    if show:
        plt.show(block=False)
    if save:
        plt.savefig(filename)
        print("ROC Curves plot SAVED")
        plt.close()


def plot_embedding(
    embeddings: Any,
    index: Sequence[BoundingBox] | Sequence[int],
    data_dir: Path | str,
    dataset_params: dict[str, Any],
    title: Optional[str] = None,
    show: bool = False,
    save: bool = True,
    filename: Optional[Path | str] = None,
    max_pixel_value: int = 255,
    cache_dir: Optional[Path | str] = None,
) -> None:
    """Using TSNE Clustering, visualises the embeddings from a model.

    Args:
        embeddings (~typing.Any): Embeddings from a model.
        bounds (~typing.Sequence[~torchgeo.datasets.utils.BoundingBox] | ~numpy.ndarray[~torchgeo.datasets.utils.BoundingBox]):  # noqa: E501
            Array of objects describing a geospatial bounding box.
            Must contain ``minx``, ``maxx``, ``miny`` and ``maxy`` parameters.
        task_name (str): Name of the task that the samples are from.
        title (str): Optional; Title of plot.
        show (bool): Optional; Whether to show plot.
        save (bool): Optional; Whether to save plot to file.
        filename (str): Optional; Name of file to save plot to.
        cache_dir (str | ~pathlib.Path): Optional; Path to the directory to load the cached dataset from.
            Defaults to None (so will create dataset from scratch).

    Returns:
        None
    """

    x = utils.tsne_cluster(embeddings)

    # TODO: This is a very naughty way of avoiding a circular import.
    # Need to reorganise package to avoid need for this.
    from minerva.datasets import make_dataset

    print("\nRE-CONSTRUCTING DATASET")
    if cache_dir is not None:
        cache = True
    else:
        cache = False
        cache_dir = ""

    dataset, _ = make_dataset(
        data_dir, dataset_params, cache=cache, cache_dir=cache_dir
    )

    images = []
    targets = []

    # Plots the predicted versus ground truth labels for all test patches supplied.
    for i in trange(len(x)):
        sample = dataset[index[i]]
        images.append(stack_rgb(sample["image"].numpy()))
        targets.append(
            [
                int(stats.mode(mask.flatten(), keepdims=False).mode)
                for mask in sample["mask"].numpy()
            ]
        )

    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    for i in range(len(x)):
        plt.text(
            x[i, 0],
            x[i, 1],
            str(targets[i]),
            color=plt.cm.Set1(targets[i][0] / 10.0),  # type: ignore
            fontdict={"weight": "bold", "size": 9},
        )

    if hasattr(offsetbox, "AnnotationBbox"):
        # only print thumbnails with matplotlib > 1.0
        shown_images: NDArray[Any, Any] = np.array([[1.0, 1.0]])  # just something big

        for i, image in enumerate(images):
            dist = np.sum((x[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don’t show points that are too close
                continue  # pragma: no cover

            shown_images = np.r_[shown_images, [x[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(image, cmap=mlp.colormaps["Greys_r"]),
                x[i],  # type: ignore
            )

            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])  # type: ignore

    if title is not None:
        plt.title(title)

    # Shows and/or saves plot.
    if show:
        plt.show(block=False)
    if save:
        if filename is None:  # pragma: no cover
            filename = "tsne_cluster_vis.png"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
        print("TSNE cluster visualisation SAVED")
        plt.close()


def format_plot_names(
    model_name: str, timestamp: str, path: Sequence[str] | str | Path
) -> dict[str, Path]:
    """Creates unique filenames of plots in a standardised format.

    Args:
        model_name (str): Name of model. e.g. ``"MLP-MkVI"``.
        timestamp (str): Time and date to be used to identify experiment.
        path (list[str] | str | ~pathlib.Path]): Path to the directory for storing plots as a :class:`list`
            of strings for each level.

    Returns:
        filenames (dict[str, ~pathlib.Path]): Formatted filenames for plots.
    """

    def standard_format(plot_type: str, *sub_dir) -> str:
        """Creates a unique filename for a plot in a standardised format.

        Args:
            plot_type (str): Plot type to use in filename.
            sub_dir (str): Additional subdirectories to add to path to filename.

        Returns:
            str: String of path to filename of the form ``"{model_name}_{timestamp}_{plot_type}.{file_ext}"``
        """
        filename = f"{model_name}_{timestamp}_{plot_type}"
        return str(universal_path(path) / universal_path(sub_dir) / filename)

    filenames = {
        "History": Path(standard_format("MH") + ".png"),
        "Pred": Path(standard_format("TP") + ".png"),
        "CM": Path(standard_format("CM") + ".png"),
        "ROC": Path(standard_format("ROC" + ".png")),
        "Mask": Path(standard_format("Mask", "Masks")),
        "PvT": Path(standard_format("PvT", "PvTs")),
        "TSNE": Path(standard_format("TSNE") + ".png"),
    }

    return filenames


def plot_results(
    plots: dict[str, bool],
    x: Optional[NDArray[Any, Int]] = None,
    y: Optional[list[int] | NDArray[Any, Int]] = None,
    z: Optional[list[int] | NDArray[Any, Int]] = None,
    metrics: Optional[dict[str, Any]] = None,
    ids: Optional[list[str]] = None,
    index: Optional[NDArray[Any, Any]] = None,
    probs: Optional[list[float] | NDArray[Any, Float]] = None,
    embeddings: Optional[NDArray[Any, Any]] = None,
    class_names: Optional[dict[int, str]] = None,
    colours: Optional[dict[int, str]] = None,
    save: bool = True,
    show: bool = False,
    model_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    results_dir: Optional[Sequence[str] | str | Path] = None,
    task_cfg: Optional[dict[str, Any]] = None,
    global_cfg: Optional[dict[str, Any]] = None,
) -> None:
    """Orchestrates the creation of various plots from the results of a model fitting.

    Args:
        plots (dict[str, bool]): Dictionary defining which plots to make.
        x (list[list[int]] | ~numpy.ndarray[~numpy.ndarray[int]]): Optional; List of images supplied to the model.
        y (list[list[int]] | ~numpy.ndarray[~numpy.ndarray[int]]): Optional; List of corresponding ground truth labels.
        z (list[list[int]] | ~numpy.ndarray[~numpy.ndarray[int]]): Optional; List of predicted label masks.
        metrics (dict[str, ~typing.Any]): Optional; Dictionary containing a log of various metrics used to assess
            the performance of a model.
        ids (list[str]): Optional; List of IDs defining the origin of samples to the model.
            Maybe either patch IDs or scene tags.
        task_name (str): Optional; Name of task that samples are from.
        index (~numpy.ndarray[int] | ~numpy.ndarray[~torchgeo.datasets.utils.BoundingBox]): Optional; Array of objects
            describing a geospatial bounding box for each sample or a sequence of indexes.
        probs (list[float] | ~numpy.ndarray[float]): Optional; Array of probabilistic predicted classes
            from model where each sample should have a list of the predicted probability for each class.
        embeddings (~numpy.ndarray[~typing.Any]): Embeddings from the model to visualise with TSNE clustering.
        class_names (dict[int, str]): Optional; Dictionary mapping class labels to class names.
        colours (dict[int, str]): Optional; Dictionary mapping class labels to colours.
        save (bool): Optional; Save the plots to file.
        show (bool): Optional; Show the plots.
        model_name (str): Optional; Name of model. e.g. MLP-MkVI.
        timestamp (str): Optional; Time and date to be used to identify experiment.
            If not specified, the current date-time is used.
        results_dir (list[str] | str | ~pathlib.Path): Optional; Path to the directory for storing plots.

    Notes:
        ``save==True``, ``show==False`` regardless of input for plots made for each sample such as PvT or Mask plots.

    Returns:
        None
    """
    if not show:
        # Ensures that there is no attempt to display figures incase no display is present.
        try:
            mlp.use("agg")
        except ImportError:  # pragma: no cover
            pass

    if OmegaConf.is_config(task_cfg):
        task_cfg = OmegaConf.to_object(task_cfg)  # type: ignore[assignment]

    if OmegaConf.is_config(global_cfg):
        global_cfg = OmegaConf.to_object(global_cfg)  # type: ignore[assignment]

    assert isinstance(task_cfg, dict)
    assert isinstance(global_cfg, dict)

    model_type = utils.fallback_params("model_type", task_cfg, global_cfg)

    flat_y = None
    flat_z = None

    if x is not None:
        x = x.reshape(-1, *x.shape[-3:])
    if y is not None:
        flat_y = utils.batch_flatten(y)
    if z is not None:
        flat_z = utils.batch_flatten(z)

    if timestamp is None:
        timestamp = utils.timestamp_now(fmt="%d-%m-%Y_%H%M")

    if model_name is None:
        model_name = utils.fallback_params("model_name", task_cfg, global_cfg, "_name_")

    assert model_name is not None

    if results_dir is None:
        results_dir = utils.fallback_params("results_dir", task_cfg, global_cfg)

    assert isinstance(results_dir, (Sequence, str, Path))

    data_root = utils.fallback_params("data_root", task_cfg, global_cfg)

    filenames = format_plot_names(model_name, timestamp, results_dir)

    try:
        universal_path(results_dir).mkdir(parents=True, exist_ok=True)
    except FileExistsError as err:
        print(err)

    if plots.get("History", False):
        assert metrics is not None

        print("\nPLOTTING MODEL HISTORY")
        plot_history(metrics, filename=filenames["History"], save=save, show=show)

    if plots.get("CM", False):
        assert class_names is not None
        assert flat_y is not None
        assert flat_z is not None

        print("\nPLOTTING CONFUSION MATRIX")

        if utils.check_substrings_in_string(model_type, "multilabel"):
            make_multilabel_confusion_matrix(
                labels=y,  # type: ignore[arg-type]
                preds=z,  # type: ignore[arg-type]
                classes=class_names,
                filename=filenames["CM"],
                save=save,
                show=show,
                figsize=task_cfg["data_config"]["fig_sizes"]["CM"],
            )
        else:
            make_confusion_matrix(
                labels=flat_y,
                pred=flat_z,
                classes=class_names,
                filename=filenames["CM"],
                save=save,
                show=show,
                figsize=task_cfg["data_config"]["fig_sizes"]["CM"],
            )

    if plots.get("Pred", False):
        assert class_names is not None
        assert colours is not None
        assert flat_z is not None

        print("\nPLOTTING CLASS DISTRIBUTION OF PREDICTIONS")
        plot_subpopulations(
            utils.find_modes(flat_z),
            class_names=class_names,
            cmap_dict=colours,
            filename=filenames["Pred"],
            save=save,
            show=show,
        )

    if plots.get("ROC", False):
        assert class_names is not None
        assert colours is not None
        assert probs is not None
        assert flat_y is not None

        print("\nPLOTTING ROC CURVES")
        make_roc_curves(
            probs,
            flat_y,
            class_names=class_names,
            colours=colours,
            filename=filenames["ROC"],
            micro=plots["micro"],
            macro=plots["macro"],
            save=save,
            show=show,
        )

    if plots.get("Mask", False):
        assert class_names is not None
        assert colours is not None
        assert flat_z is not None
        assert flat_y is not None
        assert ids is not None
        assert index is not None

        print("\nPRODUCING PRED VS GROUND TRUTH MASK PLOTS")

        if task_cfg:
            try:
                figsize = task_cfg["data_config"]["fig_sizes"]["Mask"]
            except KeyError:
                figsize = None
        else:
            figsize = None

        flat_bbox = utils.batch_flatten(index)
        (universal_path(results_dir) / "Masks").mkdir(parents=True, exist_ok=True)
        seg_plot(
            z,  # type: ignore[arg-type]
            y,  # type: ignore[arg-type]
            ids,
            flat_bbox,
            data_root,
            task_cfg["dataset_params"],
            fn_prefix=filenames["Mask"],
            classes=class_names,
            colours=colours,
            x=x,
            frac=task_cfg.get("seg_plot_samples_frac", 0.05),
            fig_dim=figsize,
            model_name=model_name,
        )

    if plots.get("TSNE", False):
        assert embeddings is not None
        assert index is not None

        print("\nPERFORMING TSNE CLUSTERING")
        plot_embedding(
            embeddings,
            index,  # type: ignore[arg-type]
            data_root,
            task_cfg["dataset_params"],
            show=show,
            save=save,
            filename=filenames["TSNE"],
        )
