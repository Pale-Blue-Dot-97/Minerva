# -*- coding: utf-8 -*-
# Copyright (C) 2023 Harry Baker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
#
# TODO: Add exception handling where appropriate.
"""Module to handle all utility functions for training, testing and evaluation of a model.

Attributes:
    IMAGERY_CONFIG_PATH (str | ~typing.Sequence[str]): Path to the imagery config ``YAML`` file.
    DATA_CONFIG_PATH (str | ~typing.Sequence[str]): Path to the data config ``YAML`` file.
    IMAGERY_CONFIG (dict[str, ~typing.Any]): Config defining the properties of the imagery used in the experiment.
    DATA_CONFIG (dict[str, ~typing.Any]): Config defining the properties of the data used in the experiment.
    DATA_DIR (str): Path to directory holding dataset.
    CACHE_DIR (str): Path to cache directory.
    RESULTS_DIR (str): Path to directory to output plots to.
    BAND_IDS (list[int] | tuple[int, ...] | dict[str, ~typing.Any]): Band IDs and position in sample image.
    IAMGE_SIZE (int | tuple[int, int] | list[int]): Defines the shape of the images.
    CLASSES (dict[str, ~typing.Any]): Mapping of class labels to class names.
    CMAP_DICT (dict[str, ~typing.Any]): Mapping of class labels to colours.
    WGS84 (~rasterio.crs.CRS): WGS84 co-ordinate reference system acting as a default :class:`~rasterio.crs.CRS`
        for transformations.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU LGPLv3"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "IMAGERY_CONFIG_PATH",
    "DATA_CONFIG_PATH",
    "DATA_CONFIG",
    "IMAGERY_CONFIG",
    "CLASSES",
    "CONFIG",
    "CMAP_DICT",
    "return_updated_kwargs",
    "pair_collate",
    "dublicator",
    "tg_to_torch",
    "pair_return",
    "is_notebook",
    "get_cuda_device",
    "set_seeds",
    "exist_delete_check",
    "mkexpdir",
    "check_dict_key",
    "datetime_reformat",
    "get_dataset_name",
    "transform_coordinates",
    "check_within_bounds",
    "deg_to_dms",
    "dec2deg",
    "get_centre_loc",
    "lat_lon_to_loc",
    "find_tensor_mode",
    "labels_to_ohe",
    "mask_to_ohe",
    "class_weighting",
    "find_empty_classes",
    "eliminate_classes",
    "load_data_specs",
    "class_transform",
    "mask_transform",
    "check_test_empty",
    "class_dist_transform",
    "class_frac",
    "threshold_scene_select",
    "find_best_of",
    "timestamp_now",
    "find_modes",
    "modes_from_manifest",
    "func_by_str",
    "check_len",
    "print_class_dist",
    "batch_flatten",
    "make_classification_report",
    "run_tensorboard",
    "compute_roc_curves",
    "find_geo_similar",
    "print_config",
    "tsne_cluster",
    "calc_norm_euc_dist",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
# ---+ Inbuilt +-------------------------------------------------------------------------------------------------------
import cmath
import functools
import importlib
import inspect
import math
import os
import random
import re as regex
import subprocess
import sys
import webbrowser
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable
from typing import Counter as CounterType
from typing import (
    Dict,
    Iterable,
    List,
    Match,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

# ---+ 3rd Party +-----------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import psutil
import rasterio as rt
import torch
import yaml
from alive_progress import alive_bar
from geopy.exc import GeocoderUnavailable
from geopy.geocoders import Photon
from nptyping import Float, Int, NDArray, Shape
from numpy.typing import ArrayLike
from pandas import DataFrame
from rasterio.crs import CRS
from scipy.spatial import distance
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.manifold import TSNE
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.preprocessing import label_binarize
from tabulate import tabulate
from torch import LongTensor, Tensor
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.types import _device
from torchgeo.datasets.utils import BoundingBox
from urllib3.exceptions import NewConnectionError

# ---+ Minerva +-------------------------------------------------------------------------------------------------------
from minerva.utils import AUX_CONFIGS, CONFIG, universal_path, visutils

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
IMAGERY_CONFIG_PATH: Union[str, Sequence[str]] = CONFIG["dir"]["configs"][
    "imagery_config"
]

DATA_CONFIG_PATH: Optional[Path]
_data_config_path: Optional[Union[str, Path]] = CONFIG["dir"]["configs"].get(
    "data_config"
)
if _data_config_path:
    DATA_CONFIG_PATH = universal_path(_data_config_path)
else:  # pragma: no cover
    assert _data_config_path is None
    DATA_CONFIG_PATH = _data_config_path

DATA_CONFIG: Optional[Dict[str, Any]] = AUX_CONFIGS.get("data_config")
IMAGERY_CONFIG: Dict[str, Any] = AUX_CONFIGS["imagery_config"]

# Path to directory holding dataset.
DATA_DIR: Path = universal_path(CONFIG["dir"]["data"])

# Path to cache directory.
CACHE_DIR: Path = universal_path(CONFIG["dir"]["cache"])

# Path to directory to output plots to.
RESULTS_DIR: Path = universal_path(CONFIG["dir"]["results"])

# Band IDs and position in sample image.
BAND_IDS: Union[int, Tuple[int, int], List[int]] = IMAGERY_CONFIG["data_specs"][
    "band_ids"
]

# Defines size of the images to determine the number of batches.
IMAGE_SIZE: Union[int, Tuple[int, int], List[int]] = IMAGERY_CONFIG["data_specs"][
    "image_size"
]

CLASSES: Dict[int, str]
CMAP_DICT: Dict[int, str]

if DATA_CONFIG:
    CLASSES = DATA_CONFIG["classes"]
    CMAP_DICT = DATA_CONFIG["colours"]
else:  # pragma: no cover
    CLASSES = {}
    CMAP_DICT = {}

# WGS84 co-ordinate reference system acting as a default CRS for transformations.
WGS84: CRS = CRS.from_epsg(4326)

# Filters out all TensorFlow messages other than errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# =====================================================================================================================
#                                                   DECORATORS
# =====================================================================================================================
def return_updated_kwargs(
    func: Callable[..., Tuple[Any, ...]]
) -> Callable[..., Tuple[Any, ...]]:
    """Decorator that allows the `kwargs` supplied to the wrapped function to be returned with updated values.

    Assumes that the wrapped function returns a :class:`dict` in the last position of the
    :class:`tuple` of returns with keys in ``kwargs`` that have new values.

    Args:
        func (~typing.Callable[..., tuple[~typing.Any, ...]): Function to be wrapped. Must take `kwargs` and return
            a :class:`dict` with updated ``kwargs`` in the last position of the :class:`tuple`.

    Returns:
        ~typing.Callable[..., tuple[~typing.Any, ...]: Wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        kwargs.update(results[-1])
        return *results[:-1], kwargs

    return wrapper


def pair_collate(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wraps a collator function so that it can handle paired samples.

    .. warning::
        *NOT* compatible with :class:`~torch.nn.parallel.DistributedDataParallel` due to it's use of :mod:`pickle`.
        Use :func:`~minerva.datasets.stack_sample_pairs` instead as a direct replacement
        for :func:`~torchgeo.datasets.utils.stack_samples`.

    Args:
        func (~typing.Callable[[~typing.Any], ~typing.Any]): Collator function to be wrapped.

    Returns:
        ~typing.Callable[[~typing.Any], ~typing.Any]: Wrapped collator function.
    """

    @functools.wraps(func)
    def wrapper(samples: Iterable[Tuple[Any, Any]]) -> Tuple[Any, Any]:
        a, b = tuple(zip(*samples))
        return func(a), func(b)

    return wrapper


def dublicator(cls):
    """Dublicates decorated transform object to handle paired samples."""

    @functools.wraps(cls, updated=())
    class Wrapper:
        def __init__(self, *args, **kwargs) -> None:
            self.wrap = cls(*args, **kwargs)

        def __call__(self, pair: Tuple[Any, Any]) -> Tuple[Any, Any]:
            a, b = pair

            return self.wrap.__call__(a), self.wrap.__call__(b)

        def __repr__(self) -> str:
            return f"dublicator({self.wrap.__repr__()})"

    return Wrapper


def tg_to_torch(cls, keys: Optional[Sequence[str]] = None):
    """Ensures wrapped transform can handle both :class:`~torch.Tensor` and :mod:`torchgeo` style :class:`dict` inputs.

    .. warning::
        *NOT* compatible with :class:`~torch.nn.parallel.DistributedDataParallel` due to it's use of :mod:`pickle`.
        This functionality is now handled within :class:`~minerva.transforms.MinervaCompose`.

    Args:
        keys (~typing.Optional[~typing.Sequence[str]]): Keys to fields within :class:`dict` inputs
            to transform values in. Defaults to ``None``.

    Raises:
        TypeError: If input is not a :class:`dict` or :class:`~torch.Tensor`.
    """

    @functools.wraps(cls, updated=())
    class Wrapper:
        def __init__(self, *args, **kwargs) -> None:
            self.wrap: Callable[
                [
                    Union[Dict[str, Any], Tensor],
                ],
                Dict[str, Any],
            ] = cls(*args, **kwargs)
            self.keys = keys

        @overload
        def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            ...  # pragma: no cover

        @overload
        def __call__(self, batch: Tensor) -> Dict[str, Any]:
            ...  # pragma: no cover

        def __call__(self, batch: Union[Dict[str, Any], Tensor]) -> Dict[str, Any]:
            if isinstance(batch, Tensor):
                return self.wrap(batch)

            elif isinstance(batch, dict) and isinstance(self.keys, Sequence):
                aug_batch: Dict[str, Any] = {}
                for key in self.keys:
                    aug_batch[key] = self.wrap(batch.pop(key))

                return {**batch, **aug_batch}

            else:
                raise TypeError(
                    f"Inputted batch has type {type(batch)}"
                    + " -- batch must be either a `Tensor` or `dict`"
                )

        def __repr__(self) -> str:
            return self.wrap.__repr__()

    return Wrapper


def pair_return(cls):
    """Wrapper for :class:`~torchgeo.datasets.GeoDataset` classes to be able to handle pairs of queries and returns.

    .. warning::
        *NOT* compatible with :class:`~torch.nn.parallel.DistributedDataParallel` due to it's use of :mod:`pickle`.
        Use :class:`~minerva.datasets.PairedDataset` directly instead, supplying the dataset to `wrap` on init.

    Raises:
        AttributeError: If an attribute cannot be found in either the :class:`Wrapper` or the wrapped ``dataset``.
    """

    @functools.wraps(cls, updated=())
    class Wrapper:
        def __init__(self, *args, **kwargs) -> None:
            self.wrap = cls(*args, **kwargs)

        def __getitem__(self, queries: Any = None) -> Tuple[Any, Any]:
            return self.wrap[queries[0]], self.wrap[queries[1]]

        def __getattr__(self, item):
            if item in self.__dict__:
                return getattr(self, item)  # pragma: no cover
            elif item in self.wrap.__dict__:
                return getattr(self.wrap, item)
            else:
                raise AttributeError

        def __repr__(self) -> Any:
            return self.wrap.__repr__()

    return Wrapper


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def _print_banner(print_func: Callable[..., None] = print) -> None:
    """Prints the :mod:`minerva` banner to ``stdout``.

    Args:
        print_func (~typing.Callable[..., None]): Function to use to print the banner. Defaults to :func:`print`.

    Raises:
        FileNotFoundError: If ``banner.txt`` cannot be found.
    """
    banner_path = Path(__file__).parent.parent.parent / "banner.txt"
    if banner_path.exists():
        with open(banner_path, "r") as f:
            print_func(f.read())
    else:  # pragma: no cover
        raise FileNotFoundError("Cannot find the banner.txt file")


@overload
def _optional_import(
    module: str,
    *,
    name: None,
    package: str,
) -> ModuleType:
    ...  # pragma: no cover


@overload
def _optional_import(
    module: str,
    *,
    name: str,
    package: str,
) -> Callable[..., Any]:
    ...  # pragma: no cover


@overload
def _optional_import(
    module: str,
    *,
    name: None,
    package: None,
) -> ModuleType:
    ...  # pragma: no cover


@overload
def _optional_import(
    module: str,
    *,
    name: str,
    package: None,
) -> Callable[..., Any]:
    ...  # pragma: no cover


@overload
def _optional_import(
    module: str,
    *,
    name: str,
) -> Callable[..., Any]:
    ...  # pragma: no cover


@overload
def _optional_import(
    module: str,
    *,
    package: str,
) -> ModuleType:
    ...  # pragma: no cover


def _optional_import(
    module: str, *, name: Optional[str] = None, package: Optional[str] = None
) -> Union[ModuleType, Callable[..., Any]]:
    try:
        _module: ModuleType = importlib.import_module(module)
        return _module if name is None else getattr(_module, name)
    except (ImportError, AttributeError) as e:  # pragma: no cover
        if package is None:
            package = module
        msg = f"install the '{package}' package to make use of this feature"
        raise ImportError(msg) from e


def check_optional_import_exist(package: str) -> bool:
    """Checks if a package is installed. Useful for optional dependencies.

    Args:
        module (str): Name of the package to check if installed.

    Returns:
        bool: ``True`` if package installed, ``False`` if not.
    """
    try:
        _ = importlib.metadata.version(package)
        return True
    except ImportError:  # pragma: no cover
        return False


def extract_class_type(var: Any) -> type:
    if inspect.isclass(var):
        return var
    else:
        return type(var)


def is_notebook() -> bool:
    """Check if this code is being executed from a Juypter Notebook or not.

    Adapted from https://gist.github.com/thomasaarholt/e5e2da71ea3ee412616b27d364e3ae82

    Returns:
        bool: ``True`` if executed by Juypter kernel. ``False`` if not.
    """
    try:
        from IPython.core.getipython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
        if "VSCODE_PID" in os.environ:  # pragma: no cover
            return False
    except:  # noqa: E722
        return False
    else:  # pragma: no cover
        return True


def get_cuda_device(device_sig: Union[int, str] = "cuda:0") -> _device:
    """Finds and returns the ``CUDA`` device, if one is available. Else, returns CPU as device.
    Assumes there is at most only one ``CUDA`` device.

    Args:
        device_sig (int | str): Optional; Either the GPU number or string representing
            the :mod:`torch` device to find. Defaults to ``'cuda:0'``.
    Returns:
        ~torch.device: ``CUDA`` device, if found. Else, CPU device.
    """
    use_cuda = torch.cuda.is_available()
    device: _device = torch.device(device_sig if use_cuda else "cpu")  # type: ignore[attr-defined]

    return device


def exist_delete_check(fn: Union[str, Path]) -> None:
    """Checks if given file exists then deletes if true.

    Args:
        fn (str): Path to file to have existence checked then deleted.

    Returns:
        None
    """
    # Checks if file exists. Deletes if True. No action taken if False
    Path(fn).unlink(missing_ok=True)


def mkexpdir(name: str) -> None:
    """Makes a new directory below the results directory with name provided. If directory already exists,
    no action is taken.

    Args:
        name (str): Name of new directory.

    Returns:
        None
    """
    try:
        (RESULTS_DIR / name).mkdir(parents=True)
    except FileExistsError:
        pass


def set_seeds(seed: int) -> None:
    """Set :mod:`torch`, :mod:`numpy` and :mod:`random` seeds for reproducibility.

    Args:
        seed (int): Seed number to set all seeds to.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)


def check_dict_key(dictionary: Dict[Any, Any], key: Any) -> bool:
    """Checks if a key exists in a dictionary and if it is ``None`` or ``False``.

    Args:
        dictionary (dict[~typing.Any, ~typing.Any]): Dictionary to check key for.
        key (~typing.Any): Key to be checked.

    Returns:
        bool: ``True`` if key exists and is not ``None`` or ``False``. ``False`` if else.
    """
    if key in dictionary:
        if dictionary[key] is None:
            return False
        elif dictionary[key] is False:
            return False
        else:
            return True
    else:
        return False


def datetime_reformat(timestamp: str, fmt1: str, fmt2: str) -> str:
    """Takes a :class:`str` representing a time stamp in one format and returns it reformatted into a second.

    Args:
        timestamp (str): Datetime string to be reformatted.
        fmt1 (str): Format of original datetime.
        fmt2 (str): New format for datetime.

    Returns:
        str: Datetime reformatted to ``fmt2``.
    """
    return datetime.strptime(timestamp, fmt1).strftime(fmt2)


def get_dataset_name() -> Optional[Union[str, Any]]:
    """Gets the name of the dataset to be used from the config name.

    Returns:
        ~typing.Optional[str | ~typing.Any]: Name of dataset as string.
    """
    data_config_fn: str = ""
    try:
        assert DATA_CONFIG_PATH is not None
        data_config_fn = DATA_CONFIG_PATH.name
    except AssertionError as err:  # pragma: no cover
        print(err)
        print(
            "DATA_CONFIG_PATH is empty! This is needed here to obtain the dataset name."
            + "\nSpecify the path to the data config in the experiment config"
        )

    match: Optional[Match[str]] = regex.search(r"(.*?)\.yml", data_config_fn)

    if match is None:  # pragma: no cover
        print("\nDataset not found!")
        return None
    else:
        return match.group(1)


@overload
def transform_coordinates(
    x: Sequence[float],
    y: Sequence[float],
    src_crs: CRS,
    new_crs: CRS = WGS84,
) -> Tuple[Sequence[float], Sequence[float]]:
    ...  # pragma: no cover


@overload
def transform_coordinates(
    x: Sequence[float],
    y: float,
    src_crs: CRS,
    new_crs: CRS = WGS84,
) -> Tuple[Sequence[float], Sequence[float]]:
    ...  # pragma: no cover


@overload
def transform_coordinates(
    x: float,
    y: Sequence[float],
    src_crs: CRS,
    new_crs: CRS = WGS84,
) -> Tuple[Sequence[float], Sequence[float]]:
    ...  # pragma: no cover


@overload
def transform_coordinates(
    x: float, y: float, src_crs: CRS, new_crs: CRS = WGS84
) -> Tuple[float, float]:
    ...  # pragma: no cover


def transform_coordinates(
    x: Union[Sequence[float], float],
    y: Union[Sequence[float], float],
    src_crs: CRS,
    new_crs: CRS = WGS84,
) -> Union[Tuple[Sequence[float], Sequence[float]], Tuple[float, float]]:
    """Transforms co-ordinates from one :class:`~rasterio.crs.CRS` to another.

    Args:
        x (~typing.Sequence[float] | float): The x co-ordinate(s).
        y (~typing.Sequence[float] | float): The y co-ordinate(s).
        src_crs (~rasterio.crs.CRS): The source co-orinates reference system (CRS).
        new_crs (~rasterio.crs.CRS): Optional; The new CRS to transform co-ordinates to.
            Defaults to ``wgs_84``.

    Returns:
        tuple[~typing.Sequence[float], ~typing.Sequence[float] | tuple[float, float]: The transformed co-ordinates.
        A :class:`tuple` if only one ``x`` and ``y`` were provided,
        sequence of tuples if sequence of ``x`` and ``y`` provided.
    """
    single = False

    # Checks if x is a float. Places x in a list if True.
    if type(x) is float:
        x = [x]
        single = True

    # Check that len(y) == len(x). Ensure y is in a list if a float.
    y = check_len(y, x)

    # Transform co-ordinates from source to new CRS and returns a tuple of (x, y)
    co_ordinates: Tuple[Sequence[float], Sequence[float]] = rt.warp.transform(  # type: ignore
        src_crs=src_crs, dst_crs=new_crs, xs=x, ys=y
    )

    assert isinstance(co_ordinates, tuple)
    assert isinstance(co_ordinates[0], Sequence)
    assert isinstance(co_ordinates[1], Sequence)

    if single:
        x_2: float = co_ordinates[0][0]
        y_2: float = co_ordinates[1][0]

        return x_2, y_2

    else:
        return co_ordinates


def check_within_bounds(bbox: BoundingBox, bounds: BoundingBox) -> BoundingBox:
    """Ensures that the a bounding box is within another.

    Args:
        bbox (~torchgeo.datasets.utils.BoundingBox): First bounding box that needs to be within the second.
        bounds (~torchgeo.datasets.utils.BoundingBox): Second outer bounding box to use as the bounds.

    Returns:
        ~torchgeo.datasets.utils.BoundingBox: Copy of ``bbox`` if it is within ``bounds`` or a new
        bounding box that has been limited to the dimensions of ``bounds`` if those of ``bbox`` exceeded them.
    """
    minx, maxx, miny, maxy = bbox.minx, bbox.maxx, bbox.miny, bbox.maxy
    if minx < bounds.minx:
        minx = bounds.minx
    if maxx > bounds.maxx:
        maxx = bounds.maxx
    if miny < bounds.miny:
        miny = bounds.miny
    if maxy > bounds.maxy:
        maxy = bounds.maxy

    return BoundingBox(minx, maxx, miny, maxy, bbox.mint, bbox.maxt)


def deg_to_dms(deg: float, axis: str = "lat") -> str:
    """Converts between decimal degrees of lat/lon to degrees, minutes, seconds.

    Credit to Gustavo Gonçalves on Stack Overflow.
    https://stackoverflow.com/questions/2579535/convert-dd-decimal-degrees-to-dms-degrees-minutes-seconds-in-python

    Args:
        deg (float): Decimal degrees of latitude or longitude.
        axis (str): Identifier between latitude (``"lat"``) or longitude (``"lon"``) for N-S, E-W direction identifier.

    Returns:
        str: String of inputted ``deg`` in degrees, minutes and seconds in the form Degreesº Minutes Seconds Hemisphere.
    """
    # Split decimal degrees into units and decimals
    decimals, number = math.modf(deg)

    # Compute degrees, minutes and seconds
    d = int(number)
    m = int(decimals * 60)
    s = (deg - d - m / 60) * 3600.00

    # Define cardinal directions between latitude and longitude
    compass = {"lat": ("N", "S"), "lon": ("E", "W")}

    # Select correct hemisphere
    compass_str = compass[axis][0 if d >= 0 else 1]

    # Return formatted str
    return "{}º{}'{:.0f}\"{}".format(abs(d), abs(m), abs(s), compass_str)


def dec2deg(
    dec_co: Union[Sequence[float], NDArray[Shape["*"], Float]],  # noqa: F722
    axis: str = "lat",
) -> List[str]:
    """Wrapper for :func:`deg_to_dms`.

    Args:
        dec_co (list[float]): Array of either latitude or longitude co-ordinates in decimal degrees.
        axis (str): Identifier between latitude (``"lat"``) or longitude (``"lon"``) for N-S, E-W identifier.

    Returns:
        list[str]: List of formatted strings in degrees, minutes and seconds.
    """
    deg_co: List[str] = []
    for co in dec_co:
        deg_co.append(deg_to_dms(co, axis=axis))

    return deg_co


def get_centre_loc(bounds: BoundingBox) -> Tuple[float, float]:
    """Gets the centre co-ordinates of the parsed bounding box.

    Args:
        bounds (~torchgeo.datasets.utils.BoundingBox): Bounding box to find the centre co-ordinates.

    Returns:
        tuple[float, float]: :class:`tuple` of the centre x, y co-ordinates of the bounding box.
    """
    mid_x = bounds.maxx - abs((bounds.maxx - bounds.minx) / 2)
    mid_y = bounds.maxy - abs((bounds.maxy - bounds.miny) / 2)

    return mid_x, mid_y


def lat_lon_to_loc(lat: Union[str, float], lon: Union[str, float]) -> str:
    """Takes a latitude - longitude co-ordinate and returns a string of the semantic location.

    Args:
        lat (str | float): Latitude of location.
        lon (str | float): Longitude of location.

    Returns:
        str: Semantic location of co-ordinates e.g. "Belper, Derbyshire, UK".
    """
    try:
        # Creates a geolocator object to query the server.
        geolocator = Photon(user_agent="geoapiExercises")

        # Query to server with lat-lon co-ordinates.
        query = geolocator.reverse(f"{lat},{lon}")

    # If there is no internet connection (i.e. on a compute cluster) this exception will likely be raised.
    except (GeocoderUnavailable, NewConnectionError):
        print("\nGeocoder unavailable")
        return ""

    else:
        if query is None:
            print("No location found!")
            return ""

        location = query.raw["properties"]  # type: ignore

        # Attempts to add possible fields to address of the location. Not all will be present for every query.
        locs: List[str] = []
        try:
            locs.append(location["city"])
        except KeyError:
            try:
                locs.append(location["county"])
            except KeyError:
                pass
        try:
            locs.append(location["state"])
        except KeyError:
            try:
                locs.append(location["country"])
            except KeyError:
                pass

        # If more than one line in the address, join together with comma seperaters.
        if len(locs) > 1:
            return ", ".join(locs)
        # If one line, just return this field as the location.
        elif len(locs) == 1:
            return locs[0]
        # If no fields found for query, return empty string.
        else:  # pragma: no cover
            return ""


def find_tensor_mode(mask: LongTensor) -> LongTensor:
    """Finds the mode value in a :class:`~torch.LongTensor`.

    Args:
        mask (~torch.LongTensor): Tensor to find modal value in.

    Returns:
        ~torch.LongTensor: A 0D, 1-element tensor containing the modal value.

    .. versionadded:: 0.22
    """
    mode = torch.mode(torch.flatten(mask)).values
    assert isinstance(mode, LongTensor)
    return mode


def labels_to_ohe(labels: Sequence[int], n_classes: int) -> NDArray[Any, Any]:
    """Convert an iterable of indices to one-hot encoded (:term:`OHE`) labels.

    Args:
        labels (~typing.Sequence[int]): Sequence of class number labels to be converted to :term:`OHE`.
        n_classes (int): Number of classes to determine length of :term:`OHE` label.

    Returns:
        ~numpy.ndarray[~typing.Any]: Labels in OHE form.
    """
    targets: NDArray[Any, Any] = np.array(labels).reshape(-1)
    ohe_labels = np.eye(n_classes)[targets]
    assert isinstance(ohe_labels, np.ndarray)
    return ohe_labels


def mask_to_ohe(mask: LongTensor, n_classes: Optional[int] = None) -> LongTensor:
    """Converts a segmentation mask to one-hot-encoding (OHE).

    Args:
        mask (~torch.LongTensor): Segmentation mask to convert.
        n_classes (int): Optional; Number of classes in total across dataset.
            If not provided, the number of classes is infered from those found in
            ``mask``.

    Note:
        It is advised that one provides ``n_classes`` as there is a fair chance that
        not all possible classes are in ``mask``. Infering from the classes present in ``mask``
        therefore is likely to result in shaping issues between masks in a batch.

    Returns:
        ~torch.LongTensor: ``mask`` converted to OHE. The one-hot-encoding is placed in the leading
        dimension. (CxHxW) where C is the number of classes.

    .. versionadded:: 0.23
    """
    if not n_classes:
        n_classes = len(CLASSES)

    ohe_mask = torch.movedim(F.one_hot(mask, num_classes=n_classes), 2, 0)
    assert isinstance(ohe_mask, LongTensor)
    return ohe_mask


def class_weighting(
    class_dist: List[Tuple[int, int]], normalise: bool = False
) -> Dict[int, float]:
    """Constructs weights for each class defined by the distribution provided.

    Note:
        Each class weight is the inverse of the number of samples of that class.
        This will most likely mean that the weights will not sum to unity.

    Args:
        class_dist (list[list[int]] or tuple[tuple[int]]): 2D iterable which should be of the form as that
            created from :meth:`collections.Counter.most_common`.
        normalise (bool): Optional; Whether to normalise class weights to total number of samples or not.

    Returns:
        dict[int, float]: Dictionary mapping class number to its weight.
    """
    # Finds total number of samples to normalise data
    n_samples: int = 0
    if normalise:
        for mode in class_dist:
            n_samples += mode[1]

    # Constructs class weights. Each weight is 1 / number of samples for that class.
    class_weights: Dict[int, float] = {}
    if normalise:
        for mode in class_dist:
            class_weights[mode[0]] = n_samples / mode[1]
    else:
        for mode in class_dist:
            class_weights[mode[0]] = 1.0 / mode[1]

    return class_weights


def find_empty_classes(
    class_dist: List[Tuple[int, int]], class_names: Dict[int, str] = CLASSES
) -> List[int]:
    """Finds which classes defined by config files are not present in the dataset.

    Args:
        class_dist (list[tuple[int, int]]): Optional; 2D iterable which should be of the form created
            from :meth:`collections.Counter.most_common`.
        class_names (dict[int, str]): Optional; Dictionary mapping the class numbers to class names.

    Returns:
        list[int]: List of classes not found in ``class_dist`` and are thus empty/ not present in dataset.
    """
    empty: List[int] = []

    # Checks which classes are not present in class_dist
    for label in class_names.keys():
        # If not present, add class label to empty.
        if label not in [mode[0] for mode in class_dist]:
            empty.append(label)

    return empty


def eliminate_classes(
    empty_classes: Union[List[int], Tuple[int, ...], NDArray[Any, Int]],
    old_classes: Optional[Dict[int, str]] = None,
    old_cmap: Optional[Dict[int, str]] = None,
) -> Tuple[Dict[int, str], Dict[int, int], Dict[int, str]]:
    """Eliminates empty classes from the class text label and class colour dictionaries and re-normalise.

    This should ensure that the remaining list of classes is still a linearly spaced list of numbers.

    Args:
        empty_classes (list[int]): List of classes not found in class_dist and are thus empty/ not present in dataset.
        old_classes (dict[int, str]): Optional; Previous mapping of class labels to class names.
        old_cmap (dict[int, str]): Optional; Previous mapping of class labels to colours.

    Returns:
        tuple[dict[int, str], dict[int, int], dict[int, str]]: :class:`tuple` of dictionaries:
            * Mapping of remaining class labels to class names.
            * Mapping from old to new classes.
            * Mapping of remaining class labels to RGB colours.
    """
    if old_classes is None:
        old_classes = CLASSES
    if old_cmap is None:
        old_cmap = CMAP_DICT

    if len(empty_classes) == 0:
        return old_classes, {}, old_cmap

    else:
        # Makes deep copies of the class and cmap dicts.
        new_classes = {key: value[:] for key, value in old_classes.items()}
        new_colours = {key: value[:] for key, value in old_cmap.items()}

        # Deletes empty classes from copied dicts.
        for label in empty_classes:
            del new_classes[label]
            del new_colours[label]

        # Holds keys that are over the length of the shortened dict.
        # i.e If there were 8 classes before and now there are 6 but class number 7 remains, it is an over key.
        over_keys = [
            key for key in new_classes.keys() if key >= len(new_classes.keys())
        ]

        # Creates OrderedDicts of the key-value pairs of the over keys.
        over_classes = OrderedDict({key: new_classes[key] for key in over_keys})
        over_colours = OrderedDict({key: new_colours[key] for key in over_keys})

        reordered_classes = {}
        reordered_colours = {}
        conversion = {}

        # Goes through the length of the remaining classes (not the keys).
        for i in range(len(new_classes.keys())):
            # If there is a remaining class present at this number, copy those corresponding values across to new dicts.
            if i in new_classes:
                reordered_classes[i] = new_classes[i]
                reordered_colours[i] = new_colours[i]
                conversion[i] = i

            # If there is no remaining class at this number (because it has been deleted),
            # fill this gap with one of the over-key classes.
            if i not in new_classes:
                class_key, class_value = over_classes.popitem()
                _, colour_value = over_colours.popitem()

                reordered_classes[i] = class_value
                reordered_colours[i] = colour_value

                conversion[class_key] = i

        return reordered_classes, conversion, reordered_colours


def load_data_specs(
    class_dist: List[Tuple[int, int]], elim: bool = False
) -> Tuple[Dict[int, str], Dict[int, int], Dict[int, str]]:
    """Loads the ``classes``, ``forwards`` (if ``elim`` is true) and ``cmap_dict`` dictionaries.

    Args:
        class_dist (list[tuple[int, int]]): Optional; 2D iterable which should be of the form created
            from :func:`Counter.most_common`.
        elim (bool): Whether to eliminate classes with no samples in.

    Returns:
        tuple[dict[int, str], dict[int, int], dict[int, str]]: The ``classes``, ``forwards`` and ``cmap_dict``
        dictionaries transformed to new classes if ``elim`` is true. Else, the ``forwards`` dict is empty
        and ``classes`` and ``cmap_dict`` are unaltered.
    """
    if elim:
        return eliminate_classes(find_empty_classes(class_dist=class_dist))
    else:
        return CLASSES, {}, CMAP_DICT


def class_transform(label: int, matrix: Dict[int, int]) -> int:
    """Transforms labels from one schema to another mapped by a supplied dictionary.

    Args:
        label (int): Label to be transformed.
        matrix (dict[int, int]): Dictionary mapping old labels to new.

    Returns:
        int: Label transformed by matrix.
    """
    return matrix[label]


@overload
def mask_transform(
    array: NDArray[Any, Int], matrix: Dict[int, int]
) -> NDArray[Any, Int]:
    ...  # pragma: no cover


@overload
def mask_transform(array: LongTensor, matrix: Dict[int, int]) -> LongTensor:
    ...  # pragma: no cover


def mask_transform(
    array: Union[NDArray[Any, Int], LongTensor],
    matrix: Dict[int, int],
) -> Union[NDArray[Any, Int], LongTensor]:
    """Transforms all labels of an N-dimensional array from one schema to another mapped by a supplied dictionary.

    Args:
        array (~numpy.ndarray[int]): N-dimensional array containing labels to be transformed.
        matrix (dict[int, int]): Dictionary mapping old labels to new.

    Returns:
        ~numpy.ndarray[int]: Array of transformed labels.
    """
    for key in matrix.keys():
        array[array == key] = matrix[key]

    return array


def check_test_empty(
    pred: Union[Sequence[int], NDArray[Any, Int]],
    labels: Union[Sequence[int], NDArray[Any, Int]],
    class_labels: Dict[int, str],
    p_dist: bool = True,
) -> Tuple[NDArray[Any, Int], NDArray[Any, Int], Dict[int, str]]:
    """Checks if any of the classes in the dataset were not present in both the predictions and ground truth labels.
    Returns corrected and re-ordered predictions, labels and class labels.

    Args:
        pred (~typing.Sequence[int] | ~numpy.ndarray[int]): List of predicted labels.
        labels (~typing.Sequence[int] | ~numpy.ndarray[int]): List of corresponding ground truth labels.
        class_labels (dict[int, str]): Dictionary mapping class labels to class names.
        p_dist (bool): Optional; Whether to print to screen the distribution of classes within each dataset.

    Returns:
        tuple[~numpy.ndarray[int], ~numpy.ndarray[int], dict[int, str]]: :class:`tuple` of:
            * List of predicted labels transformed to new classes.
            * List of corresponding ground truth labels transformed to new classes.
            * Dictionary mapping new class labels to class names.
    """
    # Finds the distribution of the classes within the data.
    labels_dist = find_modes(labels)
    pred_dist = find_modes(pred)

    if p_dist:
        # Prints class distributions of ground truth and predicted labels to stdout.
        print("\nGROUND TRUTH:")
        print_class_dist(labels_dist, class_labels=class_labels)
        print("\nPREDICTIONS:")
        print_class_dist(pred_dist, class_labels=class_labels)

    empty = []

    # Checks which classes are not present in labels and predictions and adds to empty.
    for label in class_labels.keys():
        if label not in [mode[0] for mode in labels_dist] and label not in [
            mode[0] for mode in pred_dist
        ]:
            empty.append(label)

    # Eliminates and reorganises classes based on those not present during testing.
    new_class_labels, transform, _ = eliminate_classes(empty, old_classes=class_labels)

    # Converts labels to new classes after the elimination of empty classes.
    new_labels = mask_transform(np.array(labels), transform)
    new_pred = mask_transform(np.array(pred), transform)

    return new_pred, new_labels, new_class_labels


def class_dist_transform(
    class_dist: List[Tuple[int, int]], matrix: Dict[int, int]
) -> List[Tuple[int, int]]:
    """Transforms the class distribution from an old schema to a new one.

    Args:
        class_dist (list[tuple[int, int]]): 2D iterable which should be of the form as that
            created from :meth:`collections.Counter.most_common`.
        matrix (dict[int, int]): Dictionary mapping old labels to new.

    Returns:
        list[tuple[int, int]]: Class distribution updated to new labels.
    """
    new_class_dist: List[Tuple[int, int]] = []
    for mode in class_dist:
        new_class_dist.append((class_transform(mode[0], matrix), mode[1]))

    return new_class_dist


def class_frac(patch: pd.Series) -> Dict[Any, Any]:
    """Computes the fractional sizes of the classes of the given :term:`patch` and returns a
    :class:`dict` of the results.

    Args:
        patch (~pandas.Series): Row of :class:`~pandas.DataFrame` representing the entry for a :term:`patch`.

    Returns:
        Mapping: Dictionary-like object with keys as class numbers and associated values
        of fractional size of class plus a key-value pair for the :term:`patch` ID.
    """
    new_columns: Dict[Any, Any] = dict(patch.to_dict())
    counts = 0
    for mode in patch["MODES"]:
        counts += mode[1]

    for mode in patch["MODES"]:
        new_columns[mode[0]] = mode[1] / counts

    return new_columns


def cloud_cover(scene: NDArray[Any, Any]) -> Any:
    """Calculates percentage cloud cover for a given scene based on its scene CLD.

    Args:
        scene (~numpy.ndarray[~typing.Any]): Cloud cover mask for a particular scene.

    Returns:
        float: Percentage cloud cover of scene.
    """
    return np.sum(scene) / scene.size


def threshold_scene_select(df: DataFrame, thres: float = 0.3) -> List[str]:
    """Selects all scenes in a :term:`patch` with a cloud cover less than the threshold provided.

    Args:
        df (~pandas.DataFrame): :class:`~pandas.DataFrame` containing all scenes and their cloud cover percentages.
        thres (float): Optional; Fractional limit of cloud cover below which scenes shall be selected.

    Returns:
        list[str]: List of strings representing dates of the selected scenes in ``YY_MM_DD`` format.
    """
    dates = df.loc[df["COVER"] < thres]["DATE"].tolist()
    assert type(dates) == list
    return dates


def find_best_of(
    patch_id: str,
    manifest: DataFrame,
    selector: Callable[[DataFrame], List[str]] = threshold_scene_select,
    **kwargs,
) -> List[str]:
    """Finds the scenes sorted by cloud cover using selector function supplied.

    Args:
        patch_id (str): Unique patch ID.
        manifest (~pandas.DataFrame): :class:`~pandas.DataFrame` outlining cloud cover percentages
            for all scenes in the patches desired.
        selector (~typing.Callable[[~pandas.DataFrame], list[str]]): Optional; Function to use to select scenes.
            Must take an appropriately constructed :class:`~pandas.DataFrame`.
        **kwargs: Kwargs for func.

    Returns:
        list[str]: List of strings representing dates of the selected scenes in ``YY_MM_DD`` format.
    """
    # Select rows in manifest for given patch ID.
    patch_df = manifest[manifest["PATCH"] == patch_id]

    # Re-indexes the DataFrame to datetime
    patch_df.set_index(
        pd.to_datetime(patch_df["DATE"], format="%Y_%m_%d"), drop=True, inplace=True  # type: ignore
    )

    # Sends DataFrame to scene_selection() and returns the selected scenes
    return selector(patch_df, **kwargs)


def timestamp_now(fmt: str = "%d-%m-%Y_%H%M") -> str:
    """Gets the timestamp of the datetime now.

    Args:
        fmt (str): Format of the returned timestamp.

    Returns:
        str: Timestamp of the datetime now.
    """
    return datetime.now().strftime(fmt)


def find_modes(labels: Iterable[int], plot: bool = False) -> List[Tuple[int, int]]:
    """Finds the modal distribution of the classes within the labels provided.

    Can plot the results as a pie chart if ``plot=True``.

    Args:
        labels (Iterable[int]): Class labels describing the data to be analysed.
        plot (bool): Plots distribution of subpopulations if ``True``.

    Returns:
        list[tuple[int, int]]: Modal distribution of classes in input in order of most common class.
    """
    # Finds the distribution of the classes within the data
    class_dist: List[Tuple[int, int]] = Counter(
        np.array(labels).flatten()
    ).most_common()

    if plot:
        # Plots a pie chart of the distribution of the classes within the given list of patches
        visutils.plot_subpopulations(
            class_dist, class_names=CLASSES, cmap_dict=CMAP_DICT, save=False, show=True
        )

    return class_dist


def modes_from_manifest(
    manifest: DataFrame, plot: bool = False
) -> List[Tuple[int, int]]:
    """Uses the dataset manifest to calculate the fractional size of the classes.

    Args:
        manifest (~pandas.DataFrame): :class:`~pandas.DataFrame` containing the fractional sizes
            of classes and centre pixel labels of all samples of the dataset to be used.
        plot (bool): Optional; Whether to plot the class distribution pie chart.

    Returns:
        list[tuple[int, int]]: Modal distribution of classes in the dataset provided.
    """

    def count_samples(cls):
        try:
            return manifest[cls].sum() / len(manifest)
        except KeyError:
            return manifest[f"{cls}"].sum() / len(manifest)

    class_counter: CounterType[int] = Counter()
    for classification in CLASSES.keys():
        try:
            count = count_samples(classification)
            if count == 0.0 or count == 0:
                continue
            else:
                class_counter[classification] = count
        except KeyError:
            continue
    class_dist: List[Tuple[int, int]] = class_counter.most_common()

    if plot:
        # Plots a pie chart of the distribution of the classes within the given list of patches
        visutils.plot_subpopulations(
            class_dist, class_names=CLASSES, cmap_dict=CMAP_DICT, save=False, show=True
        )

    return class_dist


def func_by_str(module_path: str, func: str) -> Callable[..., Any]:
    """Gets the constructor or callable within a module defined by the names supplied.

    Args:
        module_path (str): Name (and path to) of module desired function or class is within.
        func (str): Name of function or class desired.

    Returns:
        ~typing.Callable[[~typing.Any], ~typing.Any]: Pointer to the constructor or function requested.
    """
    # Gets module found from the path/ name supplied.
    module = importlib.import_module(module_path)

    # Returns the constructor/ callable within the module.
    func = getattr(module, func)
    assert callable(func)
    return func


def check_len(param: Any, comparator: Any) -> Union[Any, Sequence[Any]]:
    """Checks the length of one object against a comparator object.

    Args:
        param (~typing.Any): Object to have length checked.
        comparator (~typing.Any): Object to compare length of param to.

    Returns:
        ~typing.Any | ~typing.Sequence[~typing.Any]:
        * ``param`` if length of param == comparator,
        * *or* :class:`list` with ``param[0]`` elements of length comparator if param =! comparator,
        * *or* :class:`list` with param elements of length comparator if param does not have ``__len__``.
    """
    if hasattr(param, "__len__"):
        if len(param) == len(comparator):
            return param
        else:
            return [param[0]] * len(comparator)
    else:
        return [param] * len(comparator)


def calc_grad(model: Module) -> Optional[float]:
    """Calculates and prints to ``stdout`` the 2D grad norm of the model parameters.

    Args:
        model (~torch.nn.Module): :mod:`Torch` model to calculate grad norms from.

    Returns:
        total_norm (float): Total 2D grad norm of the model.

    Raises:
        AttributeError: If model has no attribute ``parameters``.
    """
    total_norm = 0.0

    try:
        # Iterate through all model parameters.
        for p in model.parameters():
            # Calculate 2D grad norm
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)

                # Converts norm to float, squares and adds to total_norm.
                total_norm += param_norm.item() ** 2

        # Square-root to give final total_norm.
        total_norm **= 0.5
        print("Total Norm:", total_norm)

        return total_norm
    except AttributeError:
        print("Model has no attribute 'parameters'. Cannot calculate grad norms")

        return None


def print_class_dist(
    class_dist: List[Tuple[int, int]], class_labels: Dict[int, str] = CLASSES
) -> None:
    """Prints the supplied ``class_dist`` in a pretty table format using :mod:`tabulate`.

    Args:
        class_dist (list[tuple[int, int]]): 2D iterable which should be of the form as that
            created from :meth:`collections.Counter.most_common`.
        class_labels (dict[int, str]): Mapping of class labels to class names.

    """

    def calc_frac(count: float, total: float) -> str:
        """Calculates the percentage size of the class from the number of counts and total counts across the dataset.

        Args:
            count (float): Number of samples in dataset belonging to this class.
            total (float): Total number of samples across dataset.

        Returns:
            str: Formatted string of the percentage size to 2 decimal places.
        """
        return "{:.2f}%".format(count * 100.0 / total)

    # Convert class_dist to dict with class labels.
    rows = [
        {"#": mode[0], "LABEL": class_labels[mode[0]], "COUNT": mode[1]}
        for mode in class_dist
    ]

    # Create pandas DataFrame from dict.
    df = DataFrame(rows)

    # Add percentage size of classes.
    df["SIZE"] = df["COUNT"].apply(calc_frac, total=float(df["COUNT"].sum()))

    # Convert dtype of COUNT from float to int64.
    df = df.astype({"COUNT": "int64"})

    # Set the index to class numbers and sort into ascending order.
    df.set_index("#", drop=True, inplace=True)
    df.sort_values(by="#", inplace=True)

    # Use tabulate to print the DataFrame in a pretty plain text format to stdout.
    print(tabulate(df, headers="keys", tablefmt="psql"))  # type: ignore


def batch_flatten(
    x: Union[NDArray[Any, Any], ArrayLike]
) -> NDArray[Shape["*"], Any]:  # noqa: F722
    """Flattens the supplied array with :func:`numpy`.

    Args:
        x (~numpy.ndarray[~typing.Any] | ~nptyping.ArrayLike]): Array to be flattened.

    Returns:
        ~numpy.ndarray[~typing.Any]: Flattened :class:`~numpy.ndarray`.
    """
    if isinstance(x, np.ndarray):
        x = x.flatten()

    else:
        x = np.array(x).flatten()

    return x


def make_classification_report(
    pred: Union[Sequence[int], NDArray[Any, Int]],
    labels: Union[Sequence[int], NDArray[Any, Int]],
    class_labels: Dict[int, str],
    print_cr: bool = True,
    p_dist: bool = False,
) -> DataFrame:
    """Generates a DataFrame of the precision, recall, f-1 score and support of the supplied predictions
    and ground truth labels.

    Uses scikit-learn's classification_report to calculate the metrics:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

    Args:
        pred (list[int] | ~numpy.ndarray[int]): List of predicted labels.
        labels (list[int] | ~numpy.ndarray[int]): List of corresponding ground truth labels.
        class_labels (dict[int, str]): Dictionary mapping class labels to class names.
        print_cr (bool): Optional; Whether to print a copy of the classification report
            :class:`~pandas.DataFrame` put through :mod:`tabulate`.
        p_dist (bool): Optional; Whether to print to screen the distribution of classes within each dataset.

    Returns:
        ~pandas.DataFrame: Classification report with the precision, recall, f-1 score and support
        for each class in a :class:`~pandas.DataFrame`.
    """
    # Checks if any of the classes in the dataset were not present in both the predictions and ground truth labels.
    # Returns corrected and re-ordered predictions, labels and class_labels.
    pred, labels, class_labels = check_test_empty(
        pred, labels, class_labels, p_dist=p_dist
    )

    # Gets the list of class names from the dict.
    class_names = [class_labels[i] for i in range(len(class_labels))]

    # Uses Sci-kit Learn's classification_report to generate the report as a nested dict.
    cr = classification_report(
        y_true=labels,
        y_pred=pred,
        labels=[i for i in range(len(class_labels))],
        zero_division=0,  # type: ignore
        output_dict=True,
    )

    # Constructs DataFrame from classification report dict.
    cr_df = DataFrame(cr)

    # Delete unneeded columns.
    for column in ("accuracy", "macro avg", "micro avg", "weighted avg"):
        try:
            del cr_df[column]
        except KeyError:
            pass

    # Transpose DataFrame so rows are classes and columns are metrics.
    cr_df = cr_df.T

    # Add column for the class names.
    cr_df["LABEL"] = class_names

    # Re-order the columns so the class names are on the left-hand side.
    cr_df = cr_df[["LABEL", "precision", "recall", "f1-score", "support"]]

    # Prints the DataFrame put through tabulate into a pretty text format to stdout.
    if print_cr:
        print(tabulate(cr_df, headers="keys", tablefmt="psql"))  # type: ignore

    return cr_df


def calc_contrastive_acc(z: Tensor) -> Tensor:
    """Calculates the accuracies of predicted samples in a constrastitive learning framework.

    Note:
        This function has to calculate the loss on the feature embeddings to obtain the gain the
        rankings of the positive samples. This is depsite the likely scenario that the loss has
        already been calculated by the embedded loss function in the model. Unfortuanately, this seemingly
        inefficent computation must be done to obtain certain variables from within the loss calculation
        needed to get the rankings.

    Args:
        z (~torch.Tensor): Feature embeddings to calculate constrastive loss (and thereby accuracy) on.

    Returns:
        ~torch.Tensor: Rankings of positive samples across the batch.
    """
    # Calculates the cosine similarity between samples.
    cos_sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)

    # Mask out cosine similarity to itself.
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)  # type: ignore[attr-defined]
    cos_sim.masked_fill_(self_mask, -9e15)

    # Find positive example -> batch_size//2 away from the original example.
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

    # Get ranking position of positive example.
    comb_sim = torch.cat(  # type: ignore[attr-defined]
        [
            cos_sim[pos_mask][:, None],  # First position positive example
            cos_sim.masked_fill(pos_mask, -9e15),
        ],
        dim=-1,
    )
    rankings = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    assert isinstance(rankings, Tensor)
    return rankings


def run_tensorboard(
    exp_name: str,
    path: Optional[Union[str, List[str], Tuple[str, ...], Path]] = None,
    env_name: str = "env",
    host_num: Union[str, int] = 6006,
    _testing: bool = False,
) -> Optional[int]:
    """Runs the :mod:`TensorBoard` logs and hosts on a local webpage.

    Args:
        exp_name (str): Unique name of the experiment to run the logs of.
        path (str | list[str] | tuple[str, ...] | ~pathlib.Path): Path to the directory holding the log.
            Can be a string or a list of strings for each sub-directory.
        env_name (str): Name of the ``conda`` environment to run :mod:`tensorBoard` in.
        host_num (str | int): Local host number :mod:`tensorBoard` will be hosted on.

    Raises:
        KeyError: If ``path is None`` but the default cannot be found in ``config``, return ``None``.

    Returns:
        int | None: Exitcode for testing purposes. ``None`` under normal use.
    """
    if not path:
        try:
            path = universal_path(CONFIG["dir"]["results"]).parent
        except KeyError:
            print("KeyError: Path not specified and default cannot be found.")
            print("ABORT OPERATION")
            return None

    # Get current working directory.
    cwd = os.getcwd()

    assert path is not None

    # Joins path together if a list or tuple.
    _path: Path = universal_path(path)

    if not (_path / exp_name).exists():
        print(_path / exp_name)
        print("Expermiment directory does not exist!")
        print("ABORT OPERATION")
        return None

    # Changes working directory to that containing the TensorBoard log.
    os.chdir(_path)

    # Activates the correct Conda environment.
    subprocess.Popen(f"conda activate {env_name}", shell=True).wait()  # nosec B602

    if _testing:
        os.chdir(cwd)
        return 0

    else:  # pragma: no cover
        # Runs TensorBoard log.
        subprocess.Popen(f"tensorboard --logdir={exp_name}", shell=True)  # nosec B602

        # Opens the TensorBoard log in a locally hosted webpage of the default system browser.
        webbrowser.open(f"localhost:{host_num}")

        # Changes back to the original CWD.
        os.chdir(cwd)

        return None


def compute_roc_curves(
    probs: NDArray[Any, Float],
    labels: Union[Sequence[int], NDArray[Any, Int]],
    class_labels: List[int],
    micro: bool = True,
    macro: bool = True,
) -> Tuple[Dict[Any, float], Dict[Any, float], Dict[Any, float]]:
    """Computes the false-positive rate, true-positive rate and AUCs for each class using a one-vs-all approach.
    The micro and macro averages are for each of these variables is also computed.

    Adapted from scikit-learn's example at:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Args:
        probs (~numpy.ndarray[float]): Array of probabilistic predicted classes from model
            where each sample should have a list of the predicted probability for each class.
        labels (list[int]): List of corresponding ground truth labels.
        class_labels (list[int]): List of class label numbers.
        micro (bool): Optional; Whether to compute the micro average ROC curves.
        macro (bool): Optional; Whether to compute the macro average ROC curves.

    Returns:
        tuple[dict[~typing.Any, float], dict[~typing.Any, float], dict[~typing.Any, float]]: :class:`tuple` of:
            * Dictionary of false-positive rates for each class and micro and macro averages.
            * Dictionary of true-positive rates for each class and micro and macro averages.
            * Dictionary of AUCs for each class and micro and macro averages.
    """

    print("\nBinarising labels")

    # One-hot-encoders the class labels to match binarised input expected by roc_curve.
    targets = label_binarize(labels, classes=class_labels)

    # Dicts to hold the false-positive rate, true-positive rate and Area Under Curves
    # of each class and micro, macro averages.
    fpr: Dict[Any, Any] = {}
    tpr: Dict[Any, Any] = {}
    roc_auc: Dict[Any, Any] = {}

    # Holds a list of the classes that were in the targets supplied to the model.
    # Avoids warnings about empty targets from sklearn!
    populated_classes: List[int] = []

    # Initialises a progress bar.
    with alive_bar(len(class_labels), bar="blocks") as bar:
        # Compute ROC curve and ROC AUC for each class.
        print("Computing class ROC curves")
        for key in class_labels:
            # Checks if this class was actually in the targets supplied to the model.
            if 1 in targets[:, key]:
                try:
                    # Calculates the true-positive and false-positive rate for this class.
                    fpr[key], tpr[key], _ = roc_curve(
                        targets[:, key], probs[:, key], pos_label=1
                    )

                    # Calculates the AUC for this class from TPR and FPR.
                    roc_auc[key] = auc(fpr[key], tpr[key])

                    # Adds the class to the list of populated classes.
                    populated_classes.append(key)

                    # Step on progress bar.
                    bar()

                except UndefinedMetricWarning:  # pragma: no cover
                    bar("Class empty!")
            else:
                print(f"Class {key} empty!")

    if micro:
        # Get the current memory utilisation of the system.
        sysvmem = psutil.virtual_memory()

        if sys.getsizeof(probs) < 0.25 * sysvmem.free:
            try:
                # Compute micro-average ROC curve and ROC AUC.
                print("Calculating micro average ROC curve")
                fpr["micro"], tpr["micro"], _ = roc_curve(
                    targets.ravel(), probs.ravel()
                )
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            except MemoryError as err:  # pragma: no cover
                print(err)

        else:  # pragma: no cover
            try:
                raise MemoryError(
                    "WARNING: Size of predicted probabilities may exceed free system memory."
                )
            except MemoryError:
                print("Aborting micro averaging.")
                pass

    if macro:
        # Aggregate all false positive rates.
        all_fpr: NDArray[Any, Any] = np.unique(
            np.concatenate([fpr[key] for key in populated_classes])
        )

        # Then interpolate all ROC curves at these points.
        print("Interpolating macro average ROC curve")
        mean_tpr = np.zeros_like(all_fpr)

        # Initialises a progress bar.
        with alive_bar(len(populated_classes), bar="blocks") as bar:
            for key in populated_classes:
                mean_tpr += np.interp(all_fpr, fpr[key], tpr[key])
                bar()

        # Finally, average it and compute AUC
        mean_tpr /= len(populated_classes)

        # Add macro FPR, TPR and AUCs to dicts.
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def find_geo_similar(bbox: BoundingBox, max_r: int = 256) -> BoundingBox:
    """Find an image that is less than or equal to the geo-spatial distance ``r`` from the intial image.

    Based on the the work of GeoCLR https://arxiv.org/abs/2108.06421v1.

    Args:
        bbox (~torchgeo.datasets.utils.BoundingBox): Original bounding box.
        max_r (int): Optional; Maximum distance new bounding box can be from original. Defaults to ``256``.

    Returns:
        ~torchgeo.datasets.utils.BoundingBox: New bounding box translated a random displacement from original.
    """
    # Find a random set of polar co-ordinates within the distance `max_r`.
    r = random.randint(0, max_r)
    phi = random.random() * math.pow(-1, random.randint(1, 2)) * math.pi

    # Convert from polar to cartesian co-ordinates and extract real and imaginary parts.
    z = cmath.rect(r, phi)
    x, y = z.real, z.imag

    # Translate `bbox` by (x, y) and return new `BoundingBox`.
    return BoundingBox(
        minx=bbox.minx + x,
        maxx=bbox.maxx + x,
        miny=bbox.miny + y,
        maxy=bbox.maxy + y,
        mint=bbox.mint,
        maxt=bbox.maxt,
    )


def print_config(conf: Optional[Dict[Any, Any]] = None) -> None:
    """Print function for the configuration file using ``YAML`` dump.

    Args:
        conf (dict[str, ~typing.Any]]): Optional; Config file to print. If ``None``, uses the ``global`` config.
    """
    if conf is None:
        conf = CONFIG

    print(yaml.dump(conf))


def tsne_cluster(
    embeddings: NDArray[Any, Any],
    n_dim: int = 2,
    lr: str = "auto",
    n_iter: int = 1000,
    verbose: int = 1,
    perplexity: int = 30,
) -> Any:
    """Trains a TSNE algorithm on the embeddings passed.

    Args:
        embeddings (~numpy.ndarray[~typing.Any]): Embeddings outputted from the model.
        n_dim (int, optional): Number of dimensions to reduce embeddings to. Defaults to 2.
        lr (str, optional): Learning rate. Defaults to "auto".
        n_iter (int, optional): Number of iterations. Defaults to 1000.
        verbose (int, optional): Verbosity. Defaults to 1.
        perplexity (int, optional): Relates to number of nearest neighbours used.
            Must be less than the length of ``embeddings``.

    Returns:
        ~typing.Any: Embeddings transformed to ``n_dim`` dimensions using TSNE.
    """

    if len(embeddings) < perplexity:
        perplexity = len(embeddings) - 1

    tsne = TSNE(
        n_dim,
        learning_rate=lr,
        n_iter=n_iter,
        verbose=verbose,
        init="random",
        perplexity=perplexity,
    )

    return tsne.fit_transform(embeddings)


def calc_norm_euc_dist(
    a: Union[Sequence[int], NDArray[Shape["*"], Int]],  # noqa: F722
    b: Union[Sequence[int], NDArray[Shape["*"], Int]],  # noqa: F722
) -> float:
    """Calculates the normalised Euclidean distance between two vectors.

    Args:
        a (~typing.Sequence[int] | ~numpy.ndarray[int]): Vector ``A``.
        b (~typing.Sequence[int] | ~numpy.ndarray[int]): Vector ``B``.

    Returns:
        float: Normalised Euclidean distance between vectors ``A`` and ``B``.
    """
    assert len(a) == len(b)
    euc_dist: float = distance.euclidean(a, b) / float(len(a))

    return euc_dist
