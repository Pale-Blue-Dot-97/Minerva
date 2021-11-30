"""Module to handle all utility functions for training, testing and evaluation of a model.

    Copyright (C) 2021 Harry James Baker

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program in LICENSE.txt. If not,
    see <https://www.gnu.org/licenses/>.

Author: Harry James Baker

Email: hjb1d20@soton.ac.uk or hjbaker97@gmail.com

Institution: University of Southampton

Created under a project funded by the Ordnance Survey Ltd.

Attributes:
    config_path (str): Path to master config YAML file.
    config (dict): Master config defining how the experiment should be conducted.
    imagery_config_path (str): Path to the imagery config YAML file.
    data_config_path (str): Path to the data config YAML file.
    imagery_config (dict): Config defining the properties of the imagery used in the experiment.
    data_config (dict): Config defining the properties of the data used in the experiment.
    data_dir (list): Path to directory holding dataset.
    results_dir (list): Path to directory to output plots to.
    patch_dir_prefix (str): Prefix to every patch ID in every patch directory name.
    label_suffix (str): Suffix of the label files identifying which dataset they belong to.
    band_ids (list): Band IDs of images to be used.
    image_size (tuple): Defines the shape of the images.
    classes (dict): Mapping of class labels to class names.
    cmap_dict (dict): Mapping of class labels to colours.
    params (dict): Sub-dict of the master config for the model hyper-parameters.

TODO:
    * Add exception handling where appropriate
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import sys
from typing import Tuple, Union, Optional, Any, Iterator
import functools
from Minerva.utils import visutils
import yaml
import os
import glob
import psutil
import math
import importlib
import webbrowser
import re as regex
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, OrderedDict, Mapping
import rasterio as rt
import rasterio.mask as rtmask
import fiona
from tabulate import tabulate
from osgeo import gdal, osr
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.exceptions import UndefinedMetricWarning
from torch.backends import cudnn
from alive_progress import alive_bar

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'

with open(config_path) as file:
    config = yaml.safe_load(file)

imagery_config_path = config['dir']['configs']['imagery_config']
data_config_path = config['dir']['configs']['data_config']

with open(imagery_config_path) as file:
    imagery_config = yaml.safe_load(file)

with open(data_config_path) as file:
    data_config = yaml.safe_load(file)

# Path to directory holding dataset.
data_dir = os.sep.join(config['dir']['data'])

# Path to directory to output plots to.
results_dir = os.path.join(*config['dir']['results'])

# Prefix to every patch ID in every patch directory name.
patch_dir_prefix = imagery_config['patch_dir_prefix']

label_suffix = data_config['label_suffix']

# Band IDs of SENTINEL-2 images contained in the LandCoverNet dataset.
band_ids = imagery_config['data_specs']['band_ids']

# Defines size of the images to determine the number of batches.
image_size = imagery_config['data_specs']['image_size']

classes = data_config['classes']

cmap_dict = data_config['colours']

# Parameters
params = config['hyperparams']['params']

# Filters out all TensorFlow messages other than errors.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_configs(master_config_path: str) -> Tuple:
    """Loads the master config from YAML. Finds other config paths within and loads them.

    Args:
        master_config_path (str): Path to the master config YAML file.

    Returns:
        Master config and any other configs found from paths in the master config.
    """

    def yaml_load(path: str) -> dict:
        """Loads YAML file from path as dict.
        Args:
            path(str): Path to YAML file.

        Returns:
            yml_file (dict): YAML file loaded as dict.
        """
        with open(path) as f:
            return yaml.safe_load(f)

    def aux_config_load(paths: dict) -> dict:
        """Loads and returns config files from YAML as dicts.

        Args:
            paths (dict): Dictionary mapping config names to paths to their YAML files.

        Returns:
            Config dictionaries loaded from YAML from paths.
        """
        configs = {}
        for config_name in paths.keys():
            # Loads config from YAML as dict.
            configs[config_name] = yaml_load(paths[config_name])
        return configs

    # First loads the master config.
    master_config = yaml_load(master_config_path)

    # Gets the paths for the other configs from master config.
    config_paths = master_config['dir']['configs']

    # Loads and returns the other configs along with master config.
    return master_config, aux_config_load(config_paths)


def get_cuda_device() -> torch.device:
    """Finds and returns the CUDA device, if one is available. Else, returns CPU as device.
    Assumes there is at most only one CUDA device.

    Returns:
        CUDA device, if found. Else, CPU device.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    return device


def exist_delete_check(fn: str) -> None:
    """Checks if given file exists then deletes if true.

    Args:
        fn (str): Path to file to have existence checked then deleted.

    Returns:
        None
    """
    # Checks if file exists. Deletes if True. No action taken if False
    if os.path.exists(fn):
        os.remove(fn)
    else:
        pass


def mkexpdir(name: str) -> None:
    """Makes a new directory below the results directory with name provided. If directory already exists,
    no action is taken.

    Args:
        name (str): Name of new directory.

    Returns:
        None
    """
    try:
        os.mkdir(os.path.join(results_dir, name))
    except FileExistsError:
        pass


def datetime_reformat(timestamp: str, fmt1: str, fmt2: str) -> str:
    """Takes a str representing a time stamp in one format and returns it reformatted into a second.

    Args:
        timestamp (str): Datetime string to be reformatted.
        fmt1 (str): Format of original datetime.
        fmt2 (str): New format for datetime.

    Returns:
        (str): Datetime reformatted to fmt2.
    """
    return datetime.strptime(timestamp, fmt1).strftime(fmt2)


def prefix_format(patch_id: str, scene: str) -> str:
    """Formats a string representing the prefix of a path to any file in a scene.

    Args:
        patch_id (str): Unique patch ID.
        scene (str): Date of scene in YY_MM_DD format.

    Returns:
        prefix (str): Prefix of path to any file in a given scene.
    """
    return os.sep.join([data_dir, patch_dir_prefix + patch_id, scene, patch_id + '_' +
                        datetime_reformat(scene, '%Y_%m_%d', '%Y%m%d')])


def scene_grab(patch_id: str) -> list:
    """Finds all scenes for a given patch.

        Args:
            patch_id (str): Unique patch ID.

        Returns:
            scene_names (list[str]): List of scene dates in YY_MM_DD.
        """
    path = '{}{}{}{}'.format(data_dir, os.sep, patch_dir_prefix, patch_id)

    # Get the name of all the directories for this patch
    scene_dirs = glob.glob('{}{}*{}'.format(path, os.sep, os.sep))

    # Extract the scene names (i.e the dates) from the paths
    return [scene.partition(path)[2].replace(os.sep, '') for scene in scene_dirs]


def cloud_grab(patch_id: str) -> Tuple[list, list]:
    """Finds and loads all CLDs for a given patch.

    Args:
        patch_id (str): Unique patch ID.

    Returns:
        scenes (list): List of CLD masks for each scene.
        scene_names (list): List of scene dates in YY_MM_DD.
    """
    # Extract the scene names (i.e the dates) from the paths
    scene_names = scene_grab(patch_id)

    # List to hold scenes
    scenes = []

    # Finds and appends each CLD of each scene of a patch to scenes
    for date in scene_names:
        scenes.append(load_array('%s_CLD_10m.tif' % prefix_format(patch_id, date), 1))

    return scenes, scene_names


def patch_grab() -> list:
    """Fetches the patch IDs from the directory holding the whole dataset.

    Returns:
        (list): List of unique patch IDs.
    """
    # Fetches the names of the all the patch directories in the dataset
    patch_dirs = glob.glob('%s/%s*/' % (data_dir, patch_dir_prefix))

    # Extracts the patch ID from the directory names and returns the list
    return [(patch.partition(patch_dir_prefix)[2])[:-1] for patch in patch_dirs]


def tile_grab(tile_dir: str, tile_suffix: str) -> list:
    """Finds and returns all unique tile IDs within the specified directory with the matching suffixes.

    Args:
        tile_dir (str): Path to directory containing the tiles.
        tile_suffix (str): Suffix of the filenames of the tiles to be found.

    Returns:
        tile_ids (list): List of unique tile IDs found.
    """
    tiles = glob.glob(os.sep.join([tile_dir, '*_{}.tif'.format(tile_suffix)]))

    return [(tile.partition(tile_suffix)[0])[-4:-1] for tile in tiles]


def get_patches_in_tile(tile_id: str, patch_ids: Optional[Union[list, tuple, np.ndarray]] = None) -> list:
    """Finds all the IDs of patches within the supplied tile.

    Args:
        tile_id (str): Unique ID of tile.
        patch_ids (list[str]): Optional; List of unique patch IDs that are members of the tile ID supplied.

    Returns:
        List of patch IDs belonging to tile_id.
    """
    if patch_ids is None:
        patch_ids = patch_grab()

    return [patch_id for patch_id in patch_ids if tile_id in patch_id]


def date_grab(patch_id: str) -> list:
    """Finds all the name of all the scene directories for a patch and returns a list of the dates reformatted.

    Args:
        patch_id (str): Unique patch ID.

    Returns:
        (list): List of the dates of the scenes in DD.MM.YYYY format for this patch_ID.
    """
    # Extract the scene names (i.e the dates) from the paths
    scene_names = scene_grab(patch_id)

    # Format the dates from US YYYY_MM_DD format into UK DD.MM.YYYY format and return list
    return [datetime_reformat(date, '%Y_%m_%d', '%d.%m.%Y') for date in scene_names]


def get_label_path(patch_id: str) -> str:
    """Gets the path to the file containing the labels for the patch with ID supplied.

    Args:
        patch_id (str): Unique patch ID.

    Returns:
        Path to file containing the labels for the patch.
    """
    return os.sep.join([data_dir, patch_dir_prefix + patch_id, patch_id + '{}.tif'.format(label_suffix)])


def get_dataset_name() -> str:
    """Gets the name of the dataset to be used from the config name.

    Returns:
        Name of dataset as string.
    """
    return regex.search('config/(.*?)\.yml', data_config_path).group(1)


def get_manifest() -> str:
    """Gets the path to the manifest for the dataset to be used.

    Returns:
        Path to manifest as string.
    """
    return os.sep.join([data_dir, '{}_Manifest.csv'.format(get_dataset_name())])


def load_array(path: str, band: int):
    """Extracts an array from opening a specific band of a .tif file.

    Args:
        path (str): Path to file.
        band (int): Band number of .tif file.

    Returns:
        data (np.ndarray[np.ndarray[float]]): 2D array representing the image from the .tif band requested.
    """
    raster = rt.open(path)

    data = raster.read(band)

    return data


def lc_load(patch_id: str):
    """Loads the LC labels for a given patch.

    Args:
        patch_id (str): Unique patch ID.

    Returns:
        LC_label (list): 2D array containing LC labels for each pixel of a patch.
    """
    return load_array(get_label_path(patch_id), 1)


def dataset_lc_load(ids: Union[list, tuple, np.ndarray], func: callable = lc_load) -> list:
    """Loads Labels for the given patch IDs using the provided loading function.

    Args:
        ids (list[str]): List of patch IDs.
        func (function): Optional; Function that loads label(s) via a patch ID input.

    Returns:
        Labels for the given patch IDs.
    """
    return [func(patch_id) for patch_id in ids]


def find_centre_label(patch_id: str) -> int:
    """Gets the annual land cover label for the central pixel of the given patch.

    Args:
        patch_id (str): Unique ID for the patch to find the centre label from.

    Returns:
        The land cover label of the central pixel of the patch.
    """
    labels = np.array(lc_load(patch_id))

    return labels[int(labels.shape[0] / 2)][int(labels.shape[1] / 2)]


def centre_pixel_only(image: Union[list, np.ndarray]) -> np.ndarray:
    """Returns a copy of the image containing only zeros and the original central pixel.

    Args:
        image (list of np.ndarray): Image to be modified.

    Returns:
        Image of only zeros with the exception of the original central pixel.
    """
    new_image = np.zeros((*image_size, len(band_ids)))

    new_image[int(image_size[0] / 2.0)][int(image_size[1] / 2.0)] = image[int(image_size[0] / 2.0)][
        int(image_size[1] / 2.0)]

    return new_image


def cut_to_extents(patch_id: str, tile_id: str, tile_dir: str) -> None:
    """Uses the extents of the patch defined by patch_id to cut out and save a new patch
        at the same location and size from the tile defined by tile_id.

    Args:
        patch_id (str): Unique patch ID.
        tile_id (str): Unique tile ID.
        tile_dir (list[str]): Path to directory containing tile as a list of strings for each level.

    Returns:
        None
    """
    # Path to patch dir and patch ID prefix.
    patch_path = os.sep.join([data_dir, patch_dir_prefix + patch_id, patch_id])

    # Path to tile to be cut from.
    tile_path = os.sep.join([tile_dir, tile_id + '_20200101-20210101.tif'])

    # Constructs shape file from the provided patch label TIFF.
    os.system('gdaltindex {}_SHP.shp {}_2018_LC_10m.tif '.format(patch_path, patch_path))

    # Loads shape file.
    with fiona.open("{}_SHP.shp".format(patch_path), "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    # Uses shape file to cut patch out from tile.
    with rt.open(tile_path) as src:
        out_image, out_transform = rtmask.mask(src, shapes, crop=True)
        out_meta = src.meta

    # Updates metadata of cut out patch with correct image size.
    out_meta.update({"driver": "GTiff",
                     "height": image_size[0],
                     "width": image_size[1],
                     "transform": out_transform})

    # Deletes any previous versions of this cut out patch.
    exist_delete_check("{}_2020_LC_10m.tif".format(patch_path))

    # Saves new cut out patch to patch dir.
    with rt.open("{}_2020_LC_10m.tif".format(patch_path), "w", **out_meta) as dest:
        dest.write(out_image)

    # Deletes temporary shape files.
    for fn in ['{}_SHP.shp'.format(patch_path), '{}_SHP.shx'.format(patch_path),
               '{}_SHP.dbf'.format(patch_path), '{}_SHP.prj'.format(patch_path)]:
        os.remove(fn)


def transform_coordinates(path: str, new_cs: osr.SpatialReference) -> list:
    """Extracts the co-ordinates of a GeoTiff file from path and returns the co-ordinates of the corners of that file
    in the new co-ordinates system provided.

    Args:
        path (str): Path to GeoTiff to extract and transform co-ordinates from.
        new_cs(osr.SpatialReference): Co-ordinate system to convert GeoTiff co-ordinates from.

    Returns:
        The corners of the image in the new co-ordinate system.
    """
    # Open GeoTiff in GDAL
    ds = gdal.Open(path)
    w = ds.RasterXSize
    h = ds.RasterYSize

    # Create SpatialReference object
    old_cs = osr.SpatialReference()

    # Fetch projection system from GeoTiff and set to old_cs
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # Create co-ordinate transformation object
    trans = osr.CoordinateTransformation(old_cs, new_cs)

    # Fetch the geospatial data from the GeoTiff file
    gt_data = ds.GetGeoTransform()

    # Calculate the minimum and maximum x and y extent of the file in the old co-ordinate system
    min_x = gt_data[0]
    min_y = gt_data[3] + w * gt_data[4] + h * gt_data[5]
    max_x = gt_data[0] + w * gt_data[1] + h * gt_data[2]
    max_y = gt_data[3]

    # Return the transformation of the corners of the file from the old co-ordinate system into the new
    return [[trans.TransformPoint(min_x, max_y)[:2], trans.TransformPoint(max_x, max_y)[:2]],
            [trans.TransformPoint(min_x, min_y)[:2], trans.TransformPoint(max_x, min_y)[:2]]]


def deg_to_dms(deg: float, axis: str = 'lat') -> str:
    """Credit to Gustavo Gonçalves on Stack Overflow.
    https://stackoverflow.com/questions/2579535/convert-dd-decimal-degrees-to-dms-degrees-minutes-seconds-in-python

    Args:
        deg (float): Decimal degrees of latitude or longitude.
        axis (str): Identifier between latitude ('lat') or longitude ('lon') for N-S, E-W direction identifier.

    Returns:
        str of inputted deg in degrees, minutes and seconds in the form DegreesºMinutes Seconds Hemisphere.
    """
    # Split decimal degrees into units and decimals
    decimals, number = math.modf(deg)

    # Compute degrees, minutes and seconds
    d = int(number)
    m = int(decimals * 60)
    s = (deg - d - m / 60) * 3600.00

    # Define cardinal directions between latitude and longitude
    compass = {
        'lat': ('N', 'S'),
        'lon': ('E', 'W')
    }

    # Select correct hemisphere
    compass_str = compass[axis][0 if d >= 0 else 1]

    # Return formatted str
    return '{}º{}\'{:.0f}"{}'.format(abs(d), abs(m), abs(s), compass_str)


def dec2deg(dec_co: Union[list, tuple, np.ndarray], axis: str = 'lat') -> list:
    """Wrapper for deg_to_dms.

    Args:
        dec_co (list[float]): Array of either latitude or longitude co-ordinates in decimal degrees.
        axis (str): Identifier between latitude ('lat') or longitude ('lon') for N-S, E-W direction identifier.

    Returns:
        deg_co (list[str]): List of formatted strings in degrees, minutes and seconds.
    """
    deg_co = []
    for co in dec_co:
        deg_co.append(deg_to_dms(co, axis=axis))

    return deg_co


def labels_to_ohe(labels: Union[list, tuple, np.ndarray], n_classes: int) -> np.ndarray:
    """Convert an iterable of indices to one-hot encoded labels.

    Args:
        labels (list[int], tuple[int], np.ndarray[int]): List of class number labels to be converted to OHE
        n_classes (int): Number of classes to determine length of OHE label

    Returns:
        Labels in OHE form.
    """
    targets = np.array(labels).reshape(-1)
    return np.eye(n_classes)[targets]


def split_data(patch_ids: Optional[Union[list, tuple, np.ndarray]] = None,
               split: Union[list, tuple, np.ndarray] = (0.7, 0.15, 0.15), func: callable = lc_load, seed: int = 42,
               shuffle: bool = True, balance: bool = True, p_dist: bool = False,
               plot: bool = False) -> Tuple[dict, dict]:
    """Splits the patch IDs into train, validation and test id sets.

    Args:
        patch_ids (list[str]): Optional; List of patch IDs that outline the whole dataset to be used. If not provided,
            the patch IDs are inferred from the directory using patch_grab.
        split (list[float] or tuple[float]): Optional; Three values giving the fractional sizes of the datasets, in the
            order (train, validation, test).
        func (callable): Optional;
        seed (int): Optional; Random seed number to fix the shuffling of the data split.
        shuffle (bool): Optional; Whether to shuffle the patch IDs in the splitting of the IDs.
        balance (bool): Optional; If True, uses make_sorted_streams to modify the list of patch IDs that attempt
            to balance the distribution of classes amongst the dataset more evenly based on the majority classes
            in patches.
        p_dist (bool): Optional; Whether to print to screen the distribution of classes within each dataset.
        plot (bool): Optional; Whether or not to plot pie charts of the class distributions within each dataset.

    Returns:
        ids (dict): Dictionary of patch IDs representing train, validation and test datasets.
        class_dists (dict): Dictionary of class dists for the train, validation, test and entire datasets.
    """
    # Fetches all patch IDs in the dataset
    if patch_ids is None:
        patch_ids = patch_grab()

    if balance:
        stream_df = make_sorted_streams(patch_ids, func=func)
        patch_ids = stream_df.to_numpy().flatten().tolist()

    # Splits the dataset into train and val-test
    train_ids, val_test_ids = train_test_split(patch_ids, train_size=split[0], test_size=(split[1] + split[2]),
                                               shuffle=shuffle, random_state=seed)

    # Splits the val-test dataset into validation and test
    val_ids, test_ids = train_test_split(val_test_ids, train_size=(split[1] / (split[1] + split[2])),
                                         test_size=(split[2] / (split[1] + split[2])), shuffle=shuffle,
                                         random_state=seed)

    manifest = pd.read_csv(get_manifest())
    class_dists = {'train': subpopulations_from_manifest(select_df_by_patch(manifest, train_ids), func, plot=plot),
                   'val': subpopulations_from_manifest(select_df_by_patch(manifest, val_ids), func, plot=plot),
                   'test': subpopulations_from_manifest(select_df_by_patch(manifest, test_ids), func, plot=plot),
                   'ALL': subpopulations_from_manifest(select_df_by_patch(manifest, patch_ids), func, plot=plot)}

    # Prints the class sub-populations of each dataset to screen.
    if p_dist or plot:
        print('\nTrain:')
        print_class_dist(class_dists['train'])
        print('\nValidation:')
        print_class_dist(class_dists['val'])
        print('\nTest:')
        print_class_dist(class_dists['test'])
        print('\nALL:')
        print_class_dist(class_dists['ALL'])

    ids = {'train': train_ids,
           'val': val_ids,
           'test': test_ids}

    return ids, class_dists


def class_weighting(class_dist: Union[list, tuple, np.ndarray], normalise: bool = False) -> dict:
    """Constructs weights for each class defined by the distribution provided. Each class weight is the inverse
    of the number of samples of that class. Note: This will most likely mean that the weights will not sum to unity.

    Args:
        class_dist (list[list[int]] or tuple[tuple[int]]): 2D iterable which should be of the form as that
            created from Counter.most_common().
        normalise (bool): Optional; Whether to normalise class weights to total number of samples or not.

    Returns:
        class_weights (dict): Dictionary mapping class number to its weight.
    """
    # Finds total number of samples to normalise data
    n_samples = 0
    if normalise:
        for mode in class_dist:
            n_samples += mode[1]

    # Constructs class weights. Each weight is 1 / number of samples for that class.
    class_weights = {}
    if normalise:
        for mode in class_dist:
            class_weights[mode[0]] = n_samples / mode[1]
    else:
        for mode in class_dist:
            class_weights[mode[0]] = 1.0 / mode[1]

    return class_weights


def weight_samples(scenes: Union[list, tuple, np.ndarray], func: callable = find_centre_label,
                   class_weights: Optional[dict] = None, normalise: bool = False) -> list:
    """Produces a weight for each sample in the dataset defined by the scene - patch ID tuple pairs provided.

    Args:
        scenes (list[list[str]] or tuple[tuple[str]]): List of patch ID - scene date tuple pairs that define
            the dataset.
        func (callable): Optional; Function to load the labels from the dataset with.
        class_weights (dict): Optional; Dictionary mapping class number to its weight.
        normalise (bool): Optional; Whether or not to normalise the weights to the total number of samples.

    Returns:
        sample_weights (list[float]): List of weights for every sample in the dataset defined by patch IDs.
    """
    # Uses class_weighting to generate the class weights given the patch IDs and function provided.
    if class_weights is None:
        patch_ids = [scene[0] for scene in scenes]
        print(find_subpopulations(dataset_lc_load(patch_ids, func)))
        class_weights = class_weighting(find_subpopulations(dataset_lc_load(patch_ids, func)), normalise=normalise)

    sample_weights = []

    # Computes the sample weights
    for scene in scenes:
        # Adds a sample weight per scene based on the corresponding patch class weight of the scene.
        sample_weights.append(class_weights[find_centre_label(scene[0])])

    return sample_weights


def find_empty_classes(patch_ids: Optional[Union[list, tuple, np.ndarray]] = None, func: callable = find_centre_label,
                       class_dist: Optional[Union[list, tuple, np.ndarray]] = None) -> list:
    """Finds which classes defined by config files are not present in the dataset defined by supplied patch IDs.

    Args:
        patch_ids (list[str]): Optional; List of patch IDs that outline the whole dataset to be used. If not provided,
            the patch IDs are inferred from the directory using patch_grab.
        func (callable): Optional; Function to load the labels from the dataset with.
        class_dist (list[list[int]]): Optional; 2D iterable which should be of the form created
            from Counter.most_common(). If not provided, is computed from patch IDs and function provided.

    Returns:
        empty (list[int]): List of classes not found in class_dist and are thus empty/ not present in dataset.
    """
    if class_dist is None:
        if patch_ids is None:
            patch_ids = patch_grab()
        class_dist = find_subpopulations(dataset_lc_load(patch_ids, func), plot=False)

    empty = []

    # Checks which classes are not present in class_dist
    for label in classes.keys():

        # If not present, add class label to empty.
        if label not in [mode[0] for mode in class_dist]:
            empty.append(label)

    return empty


def eliminate_classes(empty_classes: Union[list, tuple, np.ndarray], old_classes: Optional[dict] = None,
                      old_cmap: Optional[dict] = None) -> Tuple[dict, dict, dict]:
    """Eliminates empty classes from the class text label and class colour dictionaries and re-normalise.
    This should ensure that the remaining list of classes is still a linearly spaced list of numbers.

    Args:
        empty_classes (list[int]): List of classes not found in class_dist and are thus empty/ not present in dataset.
        old_classes (dict): Optional; Previous mapping of class labels to class names.
        old_cmap (dict): Optional; Previous mapping of class labels to colours.

    Returns:
        reordered_classes (dict): Mapping of remaining class labels to class names.
        conversion (dict): Mapping from old to new classes.
        reordered_colours (dict): Mapping of remaining class labels to RGB colours.
    """
    if old_classes is None:
        old_classes = classes
    if old_cmap is None:
        old_cmap = cmap_dict

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
        over_keys = [key for key in new_classes.keys() if key >= len(new_classes.keys())]

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
                colour_key, colour_value = over_colours.popitem()

                reordered_classes[i] = class_value
                reordered_colours[i] = colour_value

                conversion[class_key] = i

        return reordered_classes, conversion, reordered_colours


def class_transform(label: int, matrix: dict) -> int:
    """Transforms labels from one schema to another mapped by a supplied dictionary.

    Args:
        label (int): Label to be transformed.
        matrix (dict): Dictionary mapping old labels to new.

    Returns:
        Label transformed by matrix.
    """
    return matrix[label]


def mask_transform(array: Union[list, np.ndarray], matrix: dict) -> Union[list, np.ndarray]:
    """Transforms all labels of an N-dimensional array from one schema to another mapped by a supplied dictionary.

    Args:
        array (list or np.ndarray): N-dimensional array containing labels to be transformed.
        matrix (dict): Dictionary mapping old labels to new.

    Returns:
        Array of transformed labels.
    """
    for key in matrix.keys():
        array[array == key] = matrix[key]

    return array


def check_test_empty(pred: Union[list, np.ndarray], labels: Union[list, np.ndarray], class_labels: dict,
                     p_dist: bool = True) -> Tuple[Union[list, np.ndarray], Union[list, np.ndarray], dict]:
    """Checks if any of the classes in the dataset were not present in both the predictions and ground truth labels.
    Returns corrected and re-ordered predictions, labels and class_labels.

    Args:
        pred (list[int] or np.ndarray[int]): List of predicted labels.
        labels (list[int] or np.ndarray[int]): List of corresponding ground truth labels.
        class_labels (dict): Dictionary mapping class labels to class names.
        p_dist (bool): Optional; Whether to print to screen the distribution of classes within each dataset.

    Returns:
        pred (list[int] or np.ndarray[int]): List of predicted labels transformed to new classes.
        labels (list[int] or np.ndarray[int]): List of corresponding ground truth labels transformed to new classes.
        class_labels (dict): Dictionary mapping new class labels to class names.
    """
    # Finds the distribution of the classes within the data.
    labels_dist = find_subpopulations(labels)
    pred_dist = find_subpopulations(pred)

    if p_dist:
        # Prints class distributions of ground truth and predicted labels to stdout.
        print('\nGROUND TRUTH:')
        print_class_dist(labels_dist, class_labels=class_labels)
        print('\nPREDICTIONS:')
        print_class_dist(pred_dist, class_labels=class_labels)

    empty = []

    # Checks which classes are not present in labels and predictions and adds to empty.
    for label in class_labels.keys():
        if label not in [mode[0] for mode in labels_dist] and label not in [mode[0] for mode in pred_dist]:
            empty.append(label)

    # Eliminates and reorganises classes based on those not present during testing.
    class_labels, transform, _ = eliminate_classes(empty, old_classes=class_labels)

    # Converts labels to new classes after the elimination of empty classes.
    labels = mask_transform(labels, transform)
    pred = mask_transform(pred, transform)

    return pred, labels, class_labels


def class_dist_transform(class_dist: Union[list, tuple, np.ndarray], matrix: dict) -> list:
    """Transforms the class distribution from an old schema to a new one.

    Args:
        class_dist (list[list[int]]): 2D iterable which should be of the form as that
            created from Counter.most_common().
        matrix (dict): Dictionary mapping old labels to new.

    Returns:
        new_class_dist (list[list[int]]): Class distribution updated to new labels.
    """
    new_class_dist = []
    for mode in class_dist:
        new_class_dist.append((class_transform(mode[0], matrix), mode[1]))

    return new_class_dist


def find_patch_modes(patch_id: str) -> list:
    """Finds the distribution of the classes within this patch.

    Args:
        patch_id (str): Unique patch ID.

    Returns:
        Modal distribution of classes in the patch provided in order of most common mode.
    """
    return Counter(np.array(lc_load(patch_id)).flatten()).most_common()


def class_frac(patch: pd.Series) -> Mapping:
    """Computes the fractional sizes of the classes of the given patch and returns a dict of the results

    Args:
        patch (pd.Series): Row of DataFrame representing the entry for a patch

    Returns:
        new_columns (Mapping): Dictionary-like object with keys as class numbers and associated values
            of fractional size of class plus a key-value pair for the patch ID
    """
    new_columns = patch.to_dict()
    for mode in patch['MODES']:
        new_columns[mode[0]] = mode[1] / (image_size[0] * image_size[1])

    return new_columns


def extract_patch_ids(scene: Union[list, tuple, np.ndarray]) -> str:
    """Gets the patch ID from the scene patch ID - date tuple.

    Args:
        scene (list[str] or tuple[str]): Patch ID - date tuple that uniquely identifies a scene sample.

    Returns:
        Patch ID of the scene.
    """
    return scene[0]


def extract_dates(scene: Union[list, tuple, np.ndarray]) -> str:
    """Gets the date from the scene patch ID - date tuple.

    Args:
        scene (list[str] or tuple[str]): Patch ID - date tuple that uniquely identifies a scene sample.

    Returns:
        Date of the scene.
    """
    return scene[1]


def make_sorted_streams(patch_ids: Optional[Union[list, tuple, np.ndarray]] = None,
                        scenes: Optional[Union[list, tuple, np.ndarray]] = None,
                        func: callable = lc_load) -> pd.DataFrame:
    """Creates a DataFrame with columns of patch IDs sorted for each class by class size in those patches.

    Args:
        patch_ids (list[str]): List of patch IDs defining the dataset to be sorted.
        scenes (list[list[str]]): Optional; List of patch ID - scene date tuple pairs that define the dataset.
        func (function): Optional; Function to use to load labels from file with.

    Returns:
        streams_df (pd.DataFrame): Database of list of patch IDs sorted by fractional sizes of class labels.
    """
    sample_type = 'ERROR'
    if scenes is None and patch_ids is not None:
        sample_type = 'PATCH'
    if scenes is not None and patch_ids is None:
        sample_type = 'SCENE'

    df = pd.DataFrame()
    if sample_type == 'PATCH':
        df['PATCH'] = patch_ids
    if sample_type == 'SCENE':
        df['SCENE'] = scenes
        df['PATCH'] = df['SCENE'].apply(extract_patch_ids)
    else:
        raise ValueError
    # Calculates the class modes of each patch.
    df['MODES'] = df['PATCH'].apply(find_patch_modes)

    # Calculates the fractional size of each class in each patch.
    df = pd.DataFrame([row for row in df.apply(class_frac, axis=1)])

    df.fillna(0, inplace=True)

    class_dist = find_subpopulations(dataset_lc_load(df['PATCH'], func=func), plot=False)

    stream_size = int(len(df['PATCH']) / len(class_dist))

    streams = {}

    for mode in reversed(class_dist):
        stream = df.sort_values(by=mode[0], ascending=False)[sample_type][:stream_size]
        streams[mode[0]] = stream.tolist()
        df.drop(stream.index, inplace=True)

    streams_df = pd.DataFrame(streams)

    return streams_df


def hard_balance(scenes: Union[list, tuple, np.ndarray], over_factor: int = 1, seed: int = 42) -> list:
    """Under and over samples supplied scenes to create a new list of scenes that have a perfect class balance.

    Args:
        scenes (list[list[str]]): List of patch ID - scene date tuple pairs that define the dataset.
        over_factor (int): Optional; How many times on average will the same scene from the smallest class be sampled.
        seed (int): Optional; Fixes the random number sequence to use to ensure repeatability.

    Returns:
        List of perfectly balanced scene patch ID - date pairs.
    """
    df = pd.DataFrame()
    df['SCENE'] = scenes
    df['PATCH'] = df['SCENE'].apply(extract_patch_ids)

    # Calculates the class modes of each patch.
    df['LABEL'] = df['PATCH'].apply(find_centre_label)

    class_dist = find_subpopulations(df['LABEL'], plot=False)
    print(class_dist)

    balanced_scenes = []
    scenes_per_class = int(class_dist[-1][1] * over_factor)
    print(scenes_per_class)

    for mode in reversed(class_dist):
        class_scenes_df = df[df['LABEL'] == mode[0]]['SCENE']
        class_scenes = class_scenes_df.sample(n=scenes_per_class, random_state=seed, replace=True)
        class_scenes = class_scenes.tolist()
        for scene in class_scenes:
            balanced_scenes.append(scene)

    return balanced_scenes


def cloud_cover(scene: np.ndarray) -> float:
    """Calculates percentage cloud cover for a given scene based on its scene CLD.

    Args:
        scene (np.ndarray): Cloud cover mask for a particular scene.

    Returns:
        (float): Percentage cloud cover of scene.
    """
    return np.sum(scene) / scene.size


def month_sort(df: pd.DataFrame, month: str) -> str:
    """Finds the the scene with the lowest cloud cover in a given month.

    Args:
        df (pd.DataFrame): Dataframe containing all scenes and their cloud cover percentages.
        month (str): Month of a year to sort.

    Returns:
        (str): Date of the scene with the lowest cloud cover percentage for the given month.
    """
    return df.loc[month].sort_values(by='COVER')['DATE'][0]


def ref_scene_select(df: pd.DataFrame, n_scenes: int = 12) -> list:
    """Selects the scene with the least cloud cover of each month of a patch plus n_scenes more of the remaining scenes.
        Based on REF's 2-step selection criteria.

    Args:
        df (pd.DataFrame): Dataframe containing all scenes and their cloud cover percentages.
        n_scenes (int): Optional; Number of additional scenes across the year to select
            after selecting the best scene of each month.

    Returns:
        List of 12 + n_scene strings representing dates of the selected scenes in YY_MM_DD format.
    """
    # Step 1: Find scene with lowest cloud cover percentage in each month
    step1 = []
    for month in range(1, 13):
        step1.append(month_sort(df, '%d-2018' % month))

    # Step 2: Find the 12 scenes with the lowest cloud cover percentage of the remaining scenes
    df.drop(index=pd.to_datetime(step1, format='%Y_%m_%d'), inplace=True)
    step2 = df.sort_values(by='COVER')['DATE'][:n_scenes].tolist()

    # Return 24 scenes selected by the 2-step REF criteria
    return step1 + step2


def threshold_scene_select(df: pd.DataFrame, thres: float = 0.3) -> list:
    """Selects all scenes in a patch with a cloud cover less than the threshold provided.

    Args:
        df (pd.DataFrame): Dataframe containing all scenes and their cloud cover percentages.
        thres (float): Optional; Fractional limit of cloud cover below which scenes shall be selected.

    Returns:
        List of strings representing dates of the selected scenes in YY_MM_DD format.
    """
    return df.loc[df['COVER'] < thres]['DATE'].tolist()


def find_best_of(patch_id: str, manifest: pd.DataFrame, selector: callable = ref_scene_select, **kwargs):
    """Finds the scenes sorted by cloud cover using selector function supplied.

    Args:
        patch_id (str): Unique patch ID.
        manifest (pd.DataFrame): DataFrame outlining cloud cover percentages for all scenes in the patches desired.
        selector (callable): Optional; Function to use to select scenes.
            Must take an appropriately constructed pd.DataFrame.
        **kwargs: Kwargs for func.

    Returns:
        scene_names (list): List of strings representing dates of the selected scenes in YY_MM_DD format.
    """
    # Select rows in manifest for given patch ID.
    patch_df = manifest[manifest['PATCH'] == patch_id]

    # Re-indexes the DataFrame to datetime
    patch_df.set_index(pd.to_datetime(patch_df['DATE'], format='%Y_%m_%d'), drop=True, inplace=True)

    # Sends DataFrame to scene_selection() and returns the selected scenes
    return selector(patch_df, **kwargs)


def pair_production(patch_id: str, manifest: pd.DataFrame, func: callable = ref_scene_select, **kwargs) -> list:
    """Creates pairs of patch ID and date of scene to define the scenes to load from a patch.

    Args:
        patch_id (str): Unique ID of the patch.
        manifest (pd.DataFrame): DataFrame outlining cloud cover percentages for all scenes in the patches desired.
        func (callable): Optional; Function to use to select scenes.
            Must take an appropriately constructed pd.DataFrame.
        **kwargs: Kwargs for func.

    Returns:
        A list of tuples of pairs of patch ID and date of scene as strings.
    """
    scenes = find_best_of(patch_id, manifest, func, **kwargs)

    return [(patch_id, scene) for scene in scenes]


def scene_extract(patch_ids: Union[list, tuple, np.ndarray], manifest, *args, **kwargs):
    """Uses pair_production to produce patch ID - scene pairs for the whole dataset outlined by patch_ids.

    Args:
        patch_ids (list[str]): List of patch IDs from which to extract scenes from.
        manifest (pd.DataFrame): DataFrame outlining cloud cover percentages for all scenes in the patches desired.
        *args: Args for pair_production
        **kwargs: Kwargs for pair_production

    Returns:
        pairs (list[tuple[str, str]]): List of patch ID - scene pairs defining the dataset.
    """
    pairs = []
    for patch_id in patch_ids:
        # Loads pairs for given patch ID
        patch_pairs = pair_production(patch_id, manifest, *args, **kwargs)

        pairs += patch_pairs

    return pairs


def stack_bands(patch_id: str, scene: str) -> np.ndarray:
    """Stacks together all the bands of the SENTINEL-2 images in a given scene of a patch.

    Args:
        patch_id (str): Unique patch ID.
        scene (str): Date of scene in YY_MM_DD format to stack bands in.

    Returns:
        Normalised and stacked red, green, blue arrays into RGB array.
    """
    bands = []
    # Load R, G, B images from file and normalise
    for band in band_ids:
        image = load_array('%s_%s_10m.tif' % (prefix_format(patch_id, scene), band), 1).astype('float')
        image /= 65535.0
        bands.append(image)

    # Stack together RGB bands
    # Note that it has to be order BGR not RGB due to the order numpy stacks arrays
    return np.dstack(bands)


def make_time_series(patch_id: str, manifest) -> np.ndarray:
    """Makes a time-series of each pixel of a patch across 24 scenes selected by REF's criteria using scene_selection().
     All the bands in the chosen scene are stacked using stack_bands().

    Args:
        patch_id (str): Unique patch ID.
        manifest (pd.DataFrame): DataFrame outlining cloud cover percentages for all scenes in the patches desired.

    Returns:
        (np.ndarray): Array of shape(rows, columns, 24, 12) holding all x for a patch.
    """
    # List of scene dates found by REF's selection criteria
    scenes = find_best_of(patch_id, manifest)

    # Loads all pixels in a patch across the 24 scenes and 12 bands
    x = []
    for scene in scenes:
        x.append(stack_bands(patch_id, scene))

    # Returns a reordered np.ndarray holding all x for the given patch
    return np.moveaxis(np.array(x), 0, 2)


def load_patch_df(patch_id: str, manifest) -> pd.DataFrame:
    """Loads a patch using patch ID from disk into a Pandas.DataFrame and returns.

    Args:
        patch_id (str): ID for patch to be loaded.
        manifest (pd.DataFrame): DataFrame outlining cloud cover percentages for all scenes in the patches desired.

    Returns:
        df (pd.DataFrame): Patch loaded into a DataFrame.
    """
    # Initialise DataFrame object
    df = pd.DataFrame()

    # Load patch from disk and create time-series pixel stacks
    patch = make_time_series(patch_id, manifest)

    # Reshape patch
    patch = patch.reshape((patch.shape[0] * patch.shape[1], patch.shape[2] * patch.shape[3]))

    # Loads accompanying labels from file and flattens
    labels = lc_load(patch_id).flatten()

    # Wraps each pixel stack in an numpy.array, appends to a list and adds as a column to df
    df['PATCH'] = [np.array(pixel) for pixel in patch]

    # Adds labels as a column to df
    df['LABELS'] = labels

    return df


def timestamp_now(fmt: str = '%d-%m-%Y_%H%M') -> str:
    """Gets the timestamp of the datetime now.

    Args:
        fmt (str): Format of the returned timestamp.

    Returns:
        Timestamp of the datetime now.
    """
    return datetime.now().strftime(fmt)


def find_subpopulations(labels: Union[list, tuple, np.ndarray], plot: bool = False) -> list:
    """Loads all LC labels for the given patches using lc_load() then finds the number of samples for each class.

    Args:
        labels (list or np.ndarray): Class labels describing the data to be analysed.
        plot (bool): Plots distribution of subpopulations if True.

    Returns:
        class_dist (list): Modal distribution of classes in the dataset provided.
    """
    # Finds the distribution of the classes within the data
    class_dist = Counter(np.array(labels).flatten()).most_common()

    if plot:
        # Plots a pie chart of the distribution of the classes within the given list of patches
        visutils.plot_subpopulations(class_dist, class_names=classes, cmap_dict=cmap_dict, save=False, show=True)

    return class_dist


def subpopulations_from_manifest(manifest: pd.DataFrame, func: callable = lc_load, plot: bool = False) -> list:
    """Uses the dataset manifest to calculate the size of classes within the dataset without
        loading the label files.

    Args:
        manifest (pd.DataFrame): DataFrame containing the fractional sizes of classes and centre pixel labels
            of all samples of the dataset to be used.
        func (function): Optional;
        plot (bool): Optional; Whether or not to plot the class distribution pie chart.

    Returns:
        class_dist (list): Modal distribution of classes in the dataset provided.
    """
    class_dist = Counter()
    if func is lc_load:
        for classification in classes.keys():
            try:
                count = manifest['%d' % classification].sum() * image_size[0] * image_size[1]
                if count == 0.0 or count == 0:
                    continue
                else:
                    class_dist[classification] = count
            except KeyError:
                continue
        class_dist = class_dist.most_common()

    elif func is find_centre_label:
        class_dist = Counter(manifest['CPL']).most_common()

    if plot:
        # Plots a pie chart of the distribution of the classes within the given list of patches
        visutils.plot_subpopulations(class_dist, class_names=classes, cmap_dict=cmap_dict, save=False, show=True)

    return class_dist


def num_batches(num_ids: int) -> int:
    """Determines the number of batches needed to cover the dataset across ids.

    Args:
        num_ids (int): Number of patch IDs in the dataset to be loaded in by batches.

    Returns:
        num_batches (int): Number of batches needed to cover the whole dataset.
    """
    return int((num_ids * image_size[0] * image_size[1]) / params['batch_size'])


def func_by_str(module: str, func: str) -> Any:
    """Gets the constructor or callable within a module defined by the names supplied.

    Args:
        module (str): Name (and path to) of module desired function or class is within.
        func (str): Name of function or class desired.

    Returns:
        Constructor or callable request by string.
    """
    # Gets module found from the path/ name supplied.
    module = importlib.import_module(module)

    # Returns the constructor/ callable within the module.
    return getattr(module, func)


def check_len(param: Any, comparator: Any) -> Union[Any, list]:
    """Checks the length of one object against a comparator object.

    Args:
        param: Object to have length checked.
        comparator: Object to compare length of param to.

    Returns:
        param if length of param == comparator.
        list with param[0] elements of length comparator if param =! comparator.
        list with param elements of length comparator if param does not have __len__.
    """
    if hasattr(param, '__len__'):
        if len(param) == len(comparator):
            return param
        else:
            return [param[0]] * len(comparator)
    else:
        return [param] * len(comparator)


def calc_grad(model: torch.nn.Module) -> Union[float, None]:
    """Calculates and prints to standout the 2D grad norm of the model parameters.

    Args:
        model (torch.nn.Module): Torch model to calculate grad norms from.

    Returns:
        total_norm (float): Total 2D grad norm of the model.

    Raises:
        AttributeError: If model has no attribute 'parameters'.
    """
    total_norm = 0.0

    try:
        # Iterate through all model parameters.
        for p in model.parameters():
            # Calculate 2D grad norm
            param_norm = p.grad.data.norm(2)
            print(param_norm.item())

            # Converts norm to float, squares and adds to total_norm.
            total_norm += param_norm.item() ** 2

        # Square-root to give final total_norm.
        total_norm **= 0.5
        print('Total Norm:', total_norm)

        return total_norm
    except AttributeError:
        print('Model has no attribute \'parameters\'. Cannot calculate grad norms')

        return


def unzip_pairs(pairs: Union[list, tuple, np.ndarray]) -> Iterator:
    """Splits the patch ID and scene date strings from scene pairs into separate lists.

    Args:
        pairs (list[str] or tuple[str] or np.ndarray[str]):

    Returns:
        Iterator that creates a list of patch IDs and matching scene dates.
    """
    return map(list, zip(*pairs))


def select_df_by_patch(df: pd.DataFrame, patch_ids: Union[list, tuple, np.ndarray]) -> pd.DataFrame:
    """Selects the section of the DataFrame for the patch IDs provided with no duplicate IDs.
        i.e not by patch and not by scene.

    Args:
        df (pd.DataFrame): DataFrame mapping patch IDs, scene dates and their properties.
        patch_ids (list[str] or tuple[str] or np.ndarray[str]): List of unique patch IDs to cut DataFrame to.

    Returns:
        new_df (pd.DataFrame): DataFrame for the patches specified.
    """
    # Drops the patch IDs for each scene so there are all unique IDs for each patch.
    new_df = df.drop_duplicates('PATCH')
    return new_df[new_df['PATCH'].isin(patch_ids)]


def select_df_by_scenes(df: pd.DataFrame, scenes: Union[list, tuple, np.ndarray]) -> pd.DataFrame:
    """Selects the section of the DataFrame for the scenes specified.

    Args:
        df (pd.DataFrame): DataFrame mapping patch IDs, scene dates and their properties.
        scenes (list[list[str]]): List of patch ID - scene date tuple pairs to use as selection criteria.

    Returns:
        new_df (pd.DataFrame): DataFrame for the scenes specified.
    """
    # Converts scene pairs into scene tags to search the DataFrame for.
    new_df = df[df['SCENE'].isin(scene_tag(scenes))]

    return new_df


def print_class_dist(class_dist: Union[list, tuple, np.ndarray], class_labels: dict = classes) -> None:
    """Prints the supplied class_dist in a pretty table format using tabulate.

    Args:
        class_dist (list[list[int]]): 2D iterable which should be of the form as that
            created from Counter.most_common().
        class_labels (dict): Mapping of class labels to class names.

    Returns:
        None
    """

    def calc_frac(count: float, total: float) -> str:
        """Calculates the percentage size of the class from the number of counts and
            supplied total counts across the dataset.

        Args:
            count (float): Number of samples in dataset belonging to this class.
            total (float): Total number of samples across dataset.

        Returns:
            Formatted string of the percentage size to 2 decimal places.
        """
        return '{:.2f}%'.format(count * 100.0 / total)

    # Convert class_dist to dict with class labels.
    rows = [{'#': mode[0], 'LABEL': class_labels[mode[0]], 'COUNT': mode[1]} for mode in class_dist]

    # Create pandas DataFrame from dict.
    df = pd.DataFrame(rows)

    # Add percentage size of classes.
    df['SIZE'] = df['COUNT'].apply(calc_frac, total=float(df['COUNT'].sum()))

    # Convert dtype of COUNT from float to int64.
    df = df.astype({'COUNT': 'int64'})

    # Set the index to class numbers and sort into ascending order.
    df.set_index('#', drop=True, inplace=True)
    df.sort_values(by='#', inplace=True)

    # Use tabulate to print the DataFrame in a pretty plain text format to stdout.
    print(tabulate(df, headers='keys', tablefmt='psql'))


def scene_tag(scenes: Union[list, tuple, np.ndarray]) -> list:
    """Creates a list of patch ID - date tags that uniquely identify each scene in a single string.

    Args:
        scenes (list[list[str]]): List of patch ID - scene date tuple pairs create tags from.

    Returns:
        List of scene tag strings that uniquely identify each scene.
    """
    return ['{}-{}'.format(patch_id, date) for patch_id, date in scenes]


def extract_from_tag(tag: str) -> Tuple[str, str]:
    """Extracts the patch ID and date for a scene from a scene tag.

    Args:
        tag (str): Scene tag string that uniquely defines a scene. Should be of form: {patch_id}-{date}

    Returns:
        patch_id (str): Unique patch ID for scene.
        date (str): Date of the scene.
    """
    patch_id, _, date = tag.partition('-')
    return patch_id, date


def model_output_flatten(x: Any) -> Union[np.ndarray, list]:
    """Attempts to flatten the supplied array. If not ragged, should be flattened with numpy.
    If ragged, the first 2 dimensions will be flattened using list appending.

    Args:
        x: Array to be flattened.

    Returns:
        x: Either a flattened ndarray or if this failed, a list that has it's first 2 dimensions flattened.
    """
    try:
        x = x.flatten()

    except AttributeError:
        x = np.array(x).flatten()

    except ValueError:
        for i in range(len(x)):
            for j in range(len(x[i])):
                x.append(x[i][j])

    return x


def make_classification_report(pred: Union[list, np.ndarray], labels: Union[list, np.ndarray],
                               class_labels: dict, print_cr: bool = True, p_dist: bool = False) -> pd.DataFrame:
    """Generates a DataFrame of the precision, recall, f-1 score and support of the supplied predictions
    and ground truth labels.

    Uses scikit-learn's classification_report to calculate the metrics:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

    Args:
        pred (list[int] or np.ndarray[int]): List of predicted labels.
        labels (list[int] or np.ndarray[int]): List of corresponding ground truth labels.
        class_labels (dict): Dictionary mapping class labels to class names.
        print_cr (bool): Optional; Whether to print a copy of the classification report DataFrame put through tabulate.
        p_dist (bool): Optional; Whether to print to screen the distribution of classes within each dataset.

    Returns:
        cr_df (pd.DataFrame): Classification report with the precision, recall, f-1 score and support
            for each class in a DataFrame.
    """
    # Checks if any of the classes in the dataset were not present in both the predictions and ground truth labels.
    # Returns corrected and re-ordered predictions, labels and class_labels.
    pred, labels, class_labels = check_test_empty(pred, labels, class_labels, p_dist=p_dist)

    # Gets the list of class names from the dict.
    class_names = [class_labels[i] for i in range(len(class_labels))]

    # Uses Sci-kit Learn's classification_report to generate the report as a nested dict.
    cr = classification_report(y_true=labels, y_pred=pred, labels=[i for i in range(len(class_labels))],
                               zero_division=0, output_dict=True)

    # Constructs DataFrame from classification report dict.
    cr_df = pd.DataFrame(cr)

    # Delete unneeded columns.
    del cr_df['accuracy']
    del cr_df['macro avg']
    del cr_df['weighted avg']

    # Transpose DataFrame so rows are classes and columns are metrics.
    cr_df = cr_df.T

    # Add column for the class names.
    cr_df['LABEL'] = class_names

    # Re-order the columns so the class names are on the left-hand side.
    cr_df = cr_df[['LABEL', 'precision', 'recall', 'f1-score', 'support']]

    # Prints the DataFrame put through tabulate into a pretty text format to stdout.
    if print_cr:
        print(tabulate(cr_df, headers='keys', tablefmt='psql'))

    return cr_df


def run_tensorboard(path: Optional[Union[str, list, tuple]] = None, env_name: str = 'env2',
                    exp_name: Optional[str] = None, host_num: Optional[Union[str, int]] = 6006) -> None:
    """Runs the TensorBoard logs and hosts on a local webpage.

    Args:
        path (str or list[str] or tuple[str]): Path to the directory holding the log.
            Can be a string or a list of strings for each sub-directory.
        env_name (str): Name of the Conda environment to run TensorBoard in.
        exp_name (str): Unique name of the experiment to run the logs of.
        host_num (str or int): Local host number TensorBoard will be hosted on.

    Raises:
        KeyError: If exp_name is None but the default cannot be found in config, return None.
        KeyError: If path is None but the default cannot be found in config, return None.

    Returns:
        None
    """
    if not exp_name:
        try:
            exp_name = config['exp_name']
            if not path:
                try:
                    path = config['dir']['results'][:-1]
                except KeyError:
                    print('KeyError: Path not specified and default cannot be found.')
                    print('ABORT OPERATION')
                    return
        except KeyError:
            print('KeyError: Experiment name not specified and cannot be found in config.')
            print('ABORT OPERATION')
            return

    # Changes working directory to that containing the TensorBoard log.
    if isinstance(path, (list, tuple)):
        os.chdir(os.path.join(*path))

    elif isinstance(path, str):
        os.chdir(path)

    # Activates the correct Conda environment.
    os.system('conda activate {}'.format(env_name))

    # Runs TensorBoard log.
    os.system('tensorboard --logdir={}'.format(exp_name))

    # Opens the TensorBoard log in a locally hosted webpage of the default system browser.
    webbrowser.open('localhost:{}'.format(host_num))


def compute_roc_curves(probs: np.ndarray, labels: Union[list, np.ndarray],
                       class_labels: list, micro: bool = True, macro: bool = True) -> Tuple[dict, dict, dict]:
    """Computes the false-positive rate, true-positive rate and AUCs for each class using a one-vs-all approach.
    The micro and macro averages are for each of these variables is also computed.

    Adapted from scikit-learn's example at:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Args:
        probs (np.ndarray): Array of probabilistic predicted classes from model where each sample
            should have a list of the predicted probability for each class.
        labels (list[int] or np.ndarray[int]): List of corresponding ground truth labels.
        class_labels (list): List of class label numbers.
        micro (bool): Optional; Whether or not to compute the micro average ROC curves.
        macro (bool): Optional; Whether or not to compute the macro average ROC curves.

    Returns:
        fpr (dict): Dictionary of false-positive rates for each class and micro and macro averages.
        tpr (dict): Dictionary of true-positive rates for each class and micro and macro averages.
        roc_auc (dict): Dictionary of AUCs for each class and micro and macro averages.
    """

    # One-hot-encoders the class labels to match binarised input expected by roc_curve.
    targets = label_binarize(labels, classes=class_labels)

    # Dicts to hold the false-positive rate, true-positive rate and Area Under Curves
    # of each class and micro, macro averages.
    fpr = {}
    tpr = {}
    roc_auc = {}

    # Initialises a progress bar.
    with alive_bar(len(class_labels), bar='blocks') as bar:
        # Compute ROC curve and ROC AUC for each class.
        print('Computing class ROC curves')
        for key in class_labels:
            try:
                fpr[key], tpr[key], _ = roc_curve(targets[:, key], probs[:, key], pos_label=1)
                roc_auc[key] = auc(fpr[key], tpr[key])
                bar(f'Class {key}')
            except UndefinedMetricWarning:
                bar(f'Class {key} empty!')

    if micro:
        # Get the current memory utilisation of the system.
        sysvmem = psutil.virtual_memory()

        if sys.getsizeof(probs) < 0.25 * sysvmem.free:
            try:
                # Compute micro-average ROC curve and ROC AUC.
                print('Calculating micro average ROC curve')
                fpr['micro'], tpr['micro'], _ = roc_curve(targets.ravel(), probs.ravel())
                roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
            except MemoryError as err:
                print(err)
                pass
        else:
            try:
                raise MemoryError
            except MemoryError:
                print('WARNING: Size of predicted probabilities may exceed free system memory.')
                print('Aborting micro averaging.')
                pass

    if macro:
        # Aggregate all false positive rates.
        all_fpr = np.unique(np.concatenate([fpr[key] for key in class_labels]))

        # Then interpolate all ROC curves at these points.
        print('Interpolating macro average ROC curve')
        mean_tpr = np.zeros_like(all_fpr)

        # Initialises a progress bar.
        with alive_bar(len(class_labels), bar='blocks') as bar:
            for key in class_labels:
                mean_tpr += np.interp(all_fpr, fpr[key], tpr[key])
                bar('Interpolating about class {}'.format(key))

        # Finally average it and compute AUC
        mean_tpr /= len(class_labels)

        # Add macro FPR, TPR and AUCs to dicts.
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def return_updated_kwargs(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        kwargs.update(results[-1])
        return (*results[:-1], kwargs)
    return wrapper
