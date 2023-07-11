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
#
# TODO: Reduce boilerplate.
#
"""Module to visualise .tiff images, label masks and results from the fitting of neural networks for remote sensing.

Attributes:
    DATA_CONFIG (dict): Config defining the properties of the data used in the experiment.
    IMAGERY_CONFIG (dict): Config defining the properties of the imagery used in the experiment.
    DATA_DIR (list[str] | str): Path to directory holding dataset.
    BAND_IDS (dict): Band IDs and position in sample image.
    MAX_PIXEL_VALUE (int): Maximum pixel value (e.g. 255 for 8-bit integer).
    WGS84 (~rasterio.crs.CRS): WGS84 co-ordinate reference system acting as a
        default :class:`~rasterio.crs.CRS` for transformations.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "DATA_CONFIG",
    "IMAGERY_CONFIG",
    "DATA_DIR",
    "BAND_IDS",
    "MAX_PIXEL_VALUE",
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import imageio
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
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
from rasterio.crs import CRS
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay
from torchgeo.datasets.utils import BoundingBox

from minerva.utils import AUX_CONFIGS, CONFIG, universal_path, utils

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
DATA_CONFIG = AUX_CONFIGS.get("data_config")
IMAGERY_CONFIG = AUX_CONFIGS["imagery_config"]

# Path to directory holding dataset.
DATA_DIR = CONFIG["dir"]["data"]

# Band IDs and position in sample image.
BAND_IDS = IMAGERY_CONFIG["data_specs"]["band_ids"]

# Maximum pixel value (e.g. 255 for 8-bit integer).
MAX_PIXEL_VALUE = IMAGERY_CONFIG["data_specs"]["max_value"]

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
    new_x: List[NDArray[Any, Any]] = []
    for i in range(f):
        x_i = []
        for j in np.arange(start=i, stop=len(x), step=f):
            x_i.append(x[j])
        new_x.append(np.array(x_i).flatten())

    return np.array(new_x).flatten()


def dec_extent_to_deg(
    shape: Tuple[int, int],
    bounds: BoundingBox,
    src_crs: CRS,
    new_crs: CRS = WGS84,
    spacing: int = 32,
) -> Tuple[Tuple[int, int, int, int], NDArray[Any, Float], NDArray[Any, Float]]:
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
    cmap_style: Optional[Union[Colormap, str]] = None, n_classes: Optional[int] = None
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
    classes: Union[List[str], Tuple[str, ...]],
    cmap_style: Optional[Union[str, ListedColormap]] = None,
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
    rgb: Dict[str, int] = BAND_IDS,
    max_value: int = MAX_PIXEL_VALUE,
) -> NDArray[Shape["*, *, 3"], Float]:  # noqa: F722
    """Stacks together red, green and blue image bands to create a RGB array.

    Args:
        image (~numpy.ndarray[float]): Image of separate channels to be normalised
            and reshaped into stacked RGB image.
        rgb (dict[str, int]): Optional; Dictionary of which channels in image are the R, G & B bands.
        max_value (int): Optional; The maximum pixel value in ``image``. e.g. for 8 bit this will be 255.

    Returns:
        ~numpy.ndarray[float]: Normalised and stacked red, green, blue arrays into RGB array.
    """

    # Extract R, G, B bands from image and normalise.
    channels: List[Any] = []
    for channel in ["R", "G", "B"]:
        band = image[rgb[channel]] / max_value
        channels.append(band)

    # Stack together RGB bands.
    # Note that it has to be order BGR not RGB due to the order numpy stacks arrays.
    rgb_image: NDArray[Shape["3, *, *"], Any] = np.dstack(  # noqa: F722
        (channels[2], channels[1], channels[0])
    )
    assert isinstance(rgb_image, np.ndarray)
    return rgb_image


def make_rgb_image(
    image: NDArray[Shape["3, *, *"], Float],  # noqa: F722
    rgb: Dict[str, int],
    block_size: int = 32,
) -> AxesImage:
    """Creates an RGB image from a composition of red, green and blue bands.

    Args:
        image (~numpy.ndarray[int]): Array representing the image of shape ``(bands x height x width)``.
        rgb (dict[str, int]): Dictionary of channel numbers of R, G & B bands within ``image``.
        block_size (int): Optional; Size of block image sub-division in pixels.

    Returns:
        ~matplotlib.image.AxesImage: Plotted RGB image object.
    """
    # Stack RGB image data together.
    rgb_image_array = stack_rgb(image, rgb)

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
    path: Union[str, Path],
    name: str,
    classes: Union[List[str], Tuple[str, ...]],
    cmap_style: Optional[Union[str, ListedColormap]] = None,
    new_crs: Optional[CRS] = WGS84,
    block_size: int = 32,
    alpha: float = 0.5,
    show: bool = True,
    save: bool = True,
    figdim: Tuple[Union[int, float], Union[int, float]] = (8.02, 10.32),
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
    mask_shape: Tuple[int, int] = mask.shape  # type: ignore[assignment]
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
        mask, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5, extent=extent, alpha=alpha  # type: ignore[arg-type]
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
    classes: Union[List[str], Tuple[str, ...]],
    gif_name: str,
    path: Union[str, Path],
    cmap_style: Optional[Union[str, ListedColormap]] = None,
    fps: float = 1.0,
    new_crs: Optional[CRS] = WGS84,
    alpha: float = 0.5,
    figdim: Tuple[Union[int, float], Union[int, float]] = (8.02, 10.32),
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

    # Initialise progress bar.
    with alive_bar(len(dates), bar="blocks") as bar:
        # List to hold filenames and paths of images created.
        frames = []
        for i in range(len(dates)):
            # Update progress bar with current scene.
            bar.text("SCENE ON %s" % dates[i])

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

            # Update bar with step completion.
            bar()

    # Checks GIF doesn't already exist. Deletes if it does.
    utils.exist_delete_check(gif_name)

    # Create a 'unknown' bar to 'spin' while the GIF is created.
    with alive_bar(unknown="waves") as bar:
        # Add current operation to spinner bar.
        bar.text("MAKING PATCH GIF")

        # Create GIF.
        imageio.mimwrite(gif_name, frames, format=".gif", duration=duration)  # type: ignore


def prediction_plot(
    sample: Dict[str, Any],
    sample_id: str,
    classes: Dict[int, str],
    src_crs: CRS,
    new_crs: CRS = WGS84,
    cmap_style: Optional[Union[str, ListedColormap]] = None,
    exp_id: Optional[str] = None,
    fig_dim: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    block_size: int = 32,
    show: bool = True,
    save: bool = True,
    fn_prefix: Optional[Union[str, Path]] = None,
) -> None:
    """
    Produces a figure containing subplots of the predicted label mask, the ground truth label mask
    and a reference RGB image of the same patch.

    Args:
        sample (dict[str, ~typing.Any]): Dictionary holding the ``"image"``, ground truth (``"mask"``)
            and predicted (``"pred"``) masks and the bounding box for this sample.
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
    z = sample["pred"]
    y = sample["mask"]
    bounds = sample["bounds"]

    extent, lat_extent, lon_extent = dec_extent_to_deg(
        y.shape, bounds, src_crs, new_crs=new_crs, spacing=block_size
    )

    centre = utils.transform_coordinates(
        *utils.get_centre_loc(bounds), src_crs=src_crs, new_crs=new_crs
    )

    # Initialises a figure.
    fig: Figure = plt.figure(figsize=fig_dim)

    gs = GridSpec(nrows=2, ncols=2, figure=fig)

    axes: NDArray[Shape["3"], Axes] = np.array(
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
        path = universal_path(CONFIG["dir"]["results"])
        fn_prefix = str(path / f"{exp_id}_{utils.timestamp_now()}_Mask")

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
    z: Union[List[int], NDArray[Any, Any]],
    y: Union[List[int], NDArray[Any, Any]],
    ids: List[str],
    bounds: Union[Sequence[Any], NDArray[Any, Any]],
    mode: str,
    classes: Dict[int, str],
    colours: Dict[int, str],
    fn_prefix: Union[str, Path],
    frac: float = 0.05,
    fig_dim: Optional[Tuple[Union[int, float], Union[int, float]]] = (9.3, 10.5),
) -> None:
    """Custom function for pre-processing the outputs from image segmentation testing for data visualisation.

    Args:
        z (list[float]): Predicted segmentation masks by the network.
        y (list[float]): Corresponding ground truth masks.
        ids (list[str]): Corresponding patch IDs for the test data supplied to the network.
        bounds (list[~torchgeo.datasets.utils.BoundingBox] | ~numpy.ndarray[~torchgeo.datasets.utils.BoundingBox]):
            Array of objects describing a geospatial bounding box.
            Must contain ``minx``, ``maxx``, ``miny`` and ``maxy`` parameters.
        mode (str): Mode samples are from. Must be ``'train'``, ``'val'`` or ``'test'``.
        classes (dict[int, str]): Dictionary mapping class labels to class names.
        colours (dict[int, str]): Dictionary mapping class labels to colours.
        fn_prefix (str | ~pathlib.Path): Common filename prefix (including path to file) for all plots of this type
            from this experiment to use.
        frac (float): Optional; Fraction of patch samples to plot.
        fig_dim (tuple[float, float]): Optional; Figure (height, width) in inches.

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
    dataset, _ = make_dataset(CONFIG["dir"]["data"], CONFIG["dataset_params"][mode])

    # Create a new projection system in lat-lon.
    crs = dataset.crs

    print("\nPRODUCING PREDICTED MASKS")

    # Limits number of masks to produce to a fractional number of total and no more than `_MAX_SAMPLES`.
    n_samples = int(frac * len(flat_ids))
    if n_samples > _MAX_SAMPLES:
        n_samples = _MAX_SAMPLES

    # Initialises a progress bar for the epoch.
    with alive_bar(n_samples, bar="blocks") as bar:
        # Plots the predicted versus ground truth labels for all test patches supplied.
        for i in random.sample(range(len(flat_ids)), n_samples):
            image = stack_rgb(dataset[bounds[i]]["image"].numpy())
            sample = {"image": image, "pred": z[i], "mask": y[i], "bounds": bounds[i]}

            prediction_plot(
                sample,
                flat_ids[i],
                classes=classes,
                src_crs=crs,
                exp_id=CONFIG["model_name"],
                show=False,
                fn_prefix=fn_prefix,
                fig_dim=fig_dim,
                cmap_style=ListedColormap(colours.values(), N=len(colours)),  # type: ignore
            )

            bar()


def plot_subpopulations(
    class_dist: List[Tuple[int, int]],
    class_names: Dict[int, str],
    cmap_dict: Dict[int, str],
    filename: Optional[Union[str, Path]] = None,
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
    colours = []

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
            class_data.append("{} \n<0.01%".format(class_names[label[0]]))
        counts.append(label[1])
        colours.append(cmap_dict[label[0]])

    # Locks figure size.
    plt.figure(figsize=(6, 5))

    # Plot a pie chart of the data distribution amongst the classes.
    patches, _ = plt.pie(
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
    metrics: Dict[str, Any],
    filename: Optional[Union[str, Path]] = None,
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
    pred: Union[List[int], NDArray[Any, Int]],
    labels: Union[List[int], NDArray[Any, Int]],
    classes: Dict[int, str],
    filename: Optional[Union[str, Path]] = None,
    cmap_style: str = "Blues",
    show: bool = True,
    save: bool = False,
) -> None:
    """Creates a heat-map of the confusion matrix of the given model.

    Args:
        pred(list[int]): Predictions made by model on test images.
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

    if DATA_CONFIG is not None:
        figsize = DATA_CONFIG["fig_sizes"]["CM"]
    else:  # pragma: no cover
        figsize = None

    # Creates the figure to plot onto.
    ax = plt.figure(figsize=figsize).gca()

    # Get a matplotlib colourmap based on the style specified to use for the confusion matrix.
    cmap = get_mlp_cmap(cmap_style)

    # Creates, plots and normalises the confusion matrix.
    cm = ConfusionMatrixDisplay.from_predictions(
        _labels,
        _pred,
        labels=list(new_classes.keys()),
        normalize="all",
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


def make_roc_curves(
    probs: ArrayLike,
    labels: Union[Sequence[int], NDArray[Any, Int]],
    class_names: Dict[int, str],
    colours: Dict[int, str],
    micro: bool = True,
    macro: bool = True,
    filename: Optional[Union[str, Path]] = None,
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
    bounds: Union[Sequence[BoundingBox], NDArray[Any, Any]],
    mode: str,
    title: Optional[str] = None,
    show: bool = False,
    save: bool = True,
    filename: Optional[Union[Path, str]] = None,
) -> None:
    """Using TSNE Clustering, visualises the embeddings from a model.

    Args:
        embeddings (~typing.Any): Embeddings from a model.
        bounds (~typing.Sequence[~torchgeo.datasets.utils.BoundingBox] | ~numpy.ndarray[~torchgeo.datasets.utils.BoundingBox]):  # noqa: E501
            Array of objects describing a geospatial bounding box.
            Must contain ``minx``, ``maxx``, ``miny`` and ``maxy`` parameters.
        mode (str): Mode samples are from. Must be ``'train'``, ``'val'`` or ``'test'``.
        title (str): Optional; Title of plot.
        show (bool): Optional; Whether to show plot.
        save (bool): Optional; Whether to save plot to file.
        filename (str): Optional; Name of file to save plot to.

    Returns:
        None
    """

    x = utils.tsne_cluster(embeddings)

    # TODO: This is a very naughty way of avoiding a circular import.
    # Need to reorganise package to avoid need for this.
    from minerva.datasets import make_dataset

    print("\nRE-CONSTRUCTING DATASET")
    dataset, _ = make_dataset(CONFIG["dir"]["data"], CONFIG["dataset_params"][mode])

    images = []
    targets = []

    # Initialises a progress bar for the epoch.
    with alive_bar(len(x), bar="blocks") as bar:
        # Plots the predicted versus ground truth labels for all test patches supplied.
        for i in range(len(x)):
            sample = dataset[bounds[i]]
            images.append(stack_rgb(sample["image"].numpy()))
            targets.append(
                [
                    int(stats.mode(mask.flatten(), keepdims=False).mode)
                    for mask in sample["mask"].numpy()
                ]
            )

            bar()

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

        for i in range(len(images)):
            dist = np.sum((x[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don’t show points that are too close
                continue  # pragma: no cover

            shown_images = np.r_[shown_images, [x[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r), x[i]  # type: ignore
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
        os.makedirs(Path(filename).parent, exist_ok=True)
        plt.savefig(filename)
        print("TSNE cluster visualisation SAVED")
        plt.close()


def format_plot_names(
    model_name: str, timestamp: str, path: Union[Sequence[str], str, Path]
) -> Dict[str, Path]:
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
    plots: Dict[str, bool],
    z: Optional[Union[List[int], NDArray[Any, Int]]] = None,
    y: Optional[Union[List[int], NDArray[Any, Int]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    ids: Optional[List[str]] = None,
    mode: str = "test",
    bounds: Optional[NDArray[Any, Any]] = None,
    probs: Optional[Union[List[float], NDArray[Any, Float]]] = None,
    embeddings: Optional[NDArray[Any, Any]] = None,
    class_names: Optional[Dict[int, str]] = None,
    colours: Optional[Dict[int, str]] = None,
    save: bool = True,
    show: bool = False,
    model_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    results_dir: Optional[Union[Sequence[str], str, Path]] = None,
) -> None:
    """Orchestrates the creation of various plots from the results of a model fitting.

    Args:
        plots (dict[str, bool]): Dictionary defining which plots to make.
        z (list[list[int]] | ~numpy.ndarray[~numpy.ndarray[int]]): List of predicted label masks.
        y (list[list[int]] | ~numpy.ndarray[~numpy.ndarray[int]]): List of corresponding ground truth label masks.
        metrics (dict[str, ~typing.Any]): Optional; Dictionary containing a log of various metrics used to assess
            the performance of a model.
        ids (list[str]): Optional; List of IDs defining the origin of samples to the model.
            Maybe either patch IDs or scene tags.
        mode (str): Optional; Mode samples are from. Must be ``'train'``, ``'val'`` or ``'test'``.
        bounds (~numpy.ndarray[~torchgeo.datasets.utils.BoundingBox]): Optional; Array of objects describing
            a geospatial bounding box for each sample.
            Must contain ``minx``, ``maxx``, ``miny`` and ``maxy`` parameters.
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

    flat_z = None
    flat_y = None

    if z is not None:
        flat_z = utils.batch_flatten(z)

    if y is not None:
        flat_y = utils.batch_flatten(y)

    if timestamp is None:
        timestamp = utils.timestamp_now(fmt="%d-%m-%Y_%H%M")

    if model_name is None:
        model_name = CONFIG["model_name"]
    assert model_name is not None

    if results_dir is None:
        results_dir = CONFIG["dir"]["results"]
        assert isinstance(results_dir, (Sequence, str, Path))

    filenames = format_plot_names(model_name, timestamp, results_dir)

    try:
        os.mkdir(universal_path(results_dir))
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
        make_confusion_matrix(
            labels=flat_y,
            pred=flat_z,
            classes=class_names,
            filename=filenames["CM"],
            save=save,
            show=show,
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
        assert z is not None
        assert y is not None
        assert ids is not None
        assert bounds is not None
        assert mode is not None

        figsize = None
        if DATA_CONFIG is not None:
            figsize = DATA_CONFIG["fig_sizes"]["Mask"]

        flat_bbox = utils.batch_flatten(bounds)
        os.makedirs(universal_path(results_dir) / "Masks", exist_ok=True)
        seg_plot(
            z,
            y,
            ids,
            flat_bbox,
            mode,
            fn_prefix=filenames["Mask"],
            classes=class_names,
            colours=colours,
            fig_dim=figsize,
        )

    if plots.get("TSNE", False):
        assert embeddings is not None
        assert bounds is not None
        assert mode is not None

        print("\nPERFORMING TSNE CLUSTERING")
        plot_embedding(
            embeddings,
            bounds,
            mode,
            show=show,
            save=save,
            filename=filenames["TSNE"],
        )
