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

Created under a project funded by the Ordnance Survey Ltd

TODO:
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.utils import visutils
import yaml
import os
import glob
import math
import importlib
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, OrderedDict
import rasterio as rt
import rasterio.mask as rtmask
import fiona
from osgeo import gdal, osr
import torch
from sklearn.model_selection import train_test_split
from torch.backends import cudnn

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'


with open(config_path) as file:
    config = yaml.safe_load(file)

imagery_config_path = config['dir']['imagery_config']
data_config_path = config['dir']['data_config']

with open(imagery_config_path) as file:
    imagery_config = yaml.safe_load(file)

with open(data_config_path) as file:
    data_config = yaml.safe_load(file)

# Path to directory holding dataset
data_dir = os.sep.join(config['dir']['data'])

# Path to directory to output plots to
results_dir = os.path.join(*config['dir']['results'])

# Model Name
model_name = config['model_name']

# Prefix to every patch ID in every patch directory name
patch_dir_prefix = imagery_config['patch_dir_prefix']

label_suffix = data_config['label_suffix']

# Band IDs of SENTINEL-2 images contained in the LandCoverNet dataset
band_ids = imagery_config['data_specs']['band_ids']

# Defines size of the images to determine the number of batches
image_size = imagery_config['data_specs']['image_size']

flattened_image_size = image_size[0] * image_size[1]

classes = data_config['classes']

cmap_dict = data_config['colours']

# Parameters
params = config['hyperparams']['params']


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_cuda_device():
    """Finds and returns the CUDA device, if one is available. Else, returns CPU as device.
    Assumes there is at most only one CUDA device.

    Returns:
        CUDA device, if found. Else, CPU device.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    return device


def exist_delete_check(fn: str):
    """Checks if given file exists then deletes if true.

    Args:
        fn (str): Path to file to have existence checked then deleted.
    """
    # Checks if file exists. Deletes if True. No action taken if False
    if os.path.exists(fn):
        os.remove(fn)
    else:
        pass


def mkexpdir(name):
    """Makes a new directory below the results directory with name provided. If directory already exists,
    no action is taken.

    Args:
        name (str): Name of new directory.
    """
    try:
        os.mkdir(os.path.join(results_dir, name))
    except FileExistsError:
        pass


def datetime_reformat(timestamp: str, fmt1: str, fmt2: str):
    """Takes a str representing a time stamp in one format and returns it reformatted into a second.

    Args:
        timestamp (str): Datetime string to be reformatted.
        fmt1 (str): Format of original datetime.
        fmt2 (str): New format for datetime.

    Returns:
        (str): Datetime reformatted to fmt2.
    """
    return datetime.strptime(timestamp, fmt1).strftime(fmt2)


def prefix_format(patch_id: str, scene):
    """Formats a string representing the prefix of a path to any file in a scene.

    Args:
        patch_id (str): Unique patch ID.
        scene (str): Date of scene in YY_MM_DD format.

    Returns:
        prefix (str): Prefix of path to any file in a given scene.
    """
    return os.sep.join([data_dir, patch_dir_prefix + patch_id, scene, patch_id + '_' +
                       datetime_reformat(scene, '%Y_%m_%d', '%Y%m%d')])


def scene_grab(patch_id: str):
    """Finds all scenes for a given patch.

        Args:
            patch_id (str): Unique patch ID.

        Returns:
            scenes (list): List of CLD masks for each scene.
            scene_names (list): List of scene dates in YY_MM_DD.
        """
    path = '{}{}{}{}'.format(data_dir, os.sep, patch_dir_prefix, patch_id)

    # Get the name of all the directories for this patch
    scene_dirs = glob.glob('{}{}*{}'.format(path, os.sep, os.sep))

    # Extract the scene names (i.e the dates) from the paths
    return [scene.partition(path)[2].replace(os.sep, '') for scene in scene_dirs]


def cloud_grab(patch_id: str):
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


def patch_grab():
    """Fetches the patch IDs from the directory holding the whole dataset.

    Returns:
        (list): List of unique patch IDs.
    """
    # Fetches the names of the all the patch directories in the dataset
    patch_dirs = glob.glob('%s/%s*/' % (data_dir, patch_dir_prefix))

    # Extracts the patch ID from the directory names and returns the list
    return [(patch.partition(patch_dir_prefix)[2])[:-1] for patch in patch_dirs]


def get_patches_in_tile(tile_id, patch_ids=None):
    if patch_ids is None:
        patch_ids = patch_grab()

    return [patch_id for patch_id in patch_ids if tile_id in patch_id]


def date_grab(patch_id: str):
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


def load_array(path: str, band: int):
    """Extracts an array from opening a specific band of a .tif file.

    Args:
        path (str): Path to file.
        band (int): Band number of .tif file.

    Returns:
        data ([[float]]): 2D array representing the image from the .tif band requested.
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
    return load_array(os.sep.join([data_dir, patch_dir_prefix + patch_id, patch_id + '{}.tif'.format(label_suffix)]), 1)


def dataset_lc_load(ids: list, func=lc_load):
    """Loads Labels for the given patch IDs using the provided loading function.

    Args:
        ids (list): List of patch IDs.
        func (function): Optional; Function that loads label(s) via a patch ID input.

    Returns:
        Labels for the given patch IDs.
    """
    return [func(patch_id) for patch_id in ids]


def find_centre_label(patch_id: str):
    """Gets the annual land cover label for the central pixel of the given patch.

    Args:
        patch_id (str): Unique ID for the patch to find the centre label from.

    Returns:
        The land cover label of the central pixel of the patch.
    """
    labels = np.array(lc_load(patch_id))

    return labels[int(labels.shape[0] / 2)][int(labels.shape[1] / 2)]


def centre_pixel_only(image):
    new_image = np.zeros((*image_size, len(band_ids)))

    new_image[int(image_size[0]/2.0)][int(image_size[1]/2.0)] = image[int(image_size[0]/2.0)][int(image_size[1]/2.0)]

    return new_image


def cut_to_extents(patch_id, tile_id, tile_dir):
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


def transform_coordinates(path: str, new_cs: osr.SpatialReference):
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


def deg_to_dms(deg: float, axis: str = 'lat'):
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


def dec2deg(dec_co, axis='lat'):
    """Wrapper for deg_to_dms.

    Args:
        dec_co ([float]): Array of either latitude or longitude co-ordinates in decimal degrees.
        axis (str): Identifier between latitude ('lat') or longitude ('lon') for N-S, E-W direction identifier.

    Returns:
        deg_co ([str]): List of formatted strings in degrees, minutes and seconds.
    """
    deg_co = []
    for co in dec_co:
        deg_co.append(deg_to_dms(co, axis=axis))

    return deg_co


def labels_to_ohe(labels, n_classes):
    """Convert an iterable of indices to one-hot encoded labels.

    Args:
        labels (list[int]): List of class number labels to be converted to OHE
        n_classes (int): Number of classes to determine length of OHE label

    Returns:
        Labels in OHE form
    """
    targets = np.array(labels).reshape(-1)
    return np.eye(n_classes)[targets]


def split_data(patch_ids=None, split=(0.7, 0.15, 0.15), func: callable = lc_load, seed: int = 42, shuffle: bool = True,
               balance: bool = True, p_dist: bool = False, plot: bool = False):
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

    # Prints the class sub-populations of each dataset to screen.
    if p_dist or plot:
        print('\nTrain: \n', find_subpopulations(dataset_lc_load(train_ids, func), plot=plot))
        print('\nValidation: \n', find_subpopulations(dataset_lc_load(val_ids, func), plot=plot))
        print('\nTest: \n', find_subpopulations(dataset_lc_load(test_ids, func), plot=plot))

    ids = {'train': train_ids,
           'val': val_ids,
           'test': test_ids}

    return ids


def class_weighting(class_dist, normalise: bool = False):
    """Constructs weights for each class defined by the distribution provided. Each class weight is the inverse
    of the number of samples of that class. Note: This will most likely mean that the weights will not sum to unity.

    Args:
        class_dist (Any[Any[float]]): 2D iterable which should be of the form created from Counter.most_common().
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


def weight_samples(scenes, func=find_centre_label, class_weights=None, normalise: bool = False):
    """Produces a weight for each sample in the dataset defined by the scene - patch ID tuple pairs provided.

    Args:
        scenes (list[Tuple[str]): List of patch ID - scene date tuple pairs that define the dataset.
        func (callable): Optional; Function to load the labels from the dataset with.
        class_weights (dict): Optional; Dictionary mapping class number to its weight.

    Returns:
        sample_weights (list[float]): List of weights for every sample in the dataset defined by patch IDs.
    """
    # Uses class_weighting to generate the class weights given the patch IDs and function provided.
    if class_weights is None:
        patch_ids = [scene[0] for scene in scenes]
        class_weights = class_weighting(find_subpopulations(dataset_lc_load(patch_ids, func)), normalise=normalise)

    sample_weights = []

    # Computes the sample weights
    for scene in scenes:
        # Adds a sample weight per scene based on the corresponding patch class weight of the scene.
        sample_weights.append(class_weights[find_centre_label(scene[0])])

    return sample_weights


def find_empty_classes(patch_ids: list = None, func: callable = find_centre_label, class_dist=None):
    """Finds which classes defined by config files are not present in the dataset defined by supplied patch IDs.

    Args:
        patch_ids (list[str]): Optional; List of patch IDs that outline the whole dataset to be used. If not provided,
            the patch IDs are inferred from the directory using patch_grab.
        func (callable): Optional; Function to load the labels from the dataset with.
        class_dist (Any[Any[float]]): Optional; 2D iterable which should be of the form created
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


def eliminate_classes(empty_classes, old_classes: dict = None, old_cmap: dict = None):
    """Eliminates empty classes from the class text label and class colour dictionaries and re-normalise.
    This should ensure that the remaining list of classes is still a linearly spaced list of numbers.

    Args:
        empty_classes (list[int]): List of classes not found in class_dist and are thus empty/ not present in dataset.

    Returns:
        reordered_classes (dict): Mapping of remaining class labels to text description.
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


def class_transform(label, matrix):
    return matrix[label]


def mask_transform(array, matrix):
    for key in matrix.keys():
        array[array == key] = matrix[key]

    return array


def class_dist_transform(class_dist, matrix):
    new_class_dist = []
    for mode in class_dist:
        new_class_dist.append((class_transform(mode[0], matrix), mode[1]))

    return new_class_dist


def find_patch_modes(patch_id):
    """Finds the distribution of the classes within this patch

    Args:
        patch_id (str): Unique patch ID

    Returns:
        (Counter): Modal distribution of classes in the patch provided
    """
    return Counter(np.array(lc_load(patch_id)).flatten()).most_common()


def class_frac(patch):
    """Computes the fractional sizes of the classes of the given patch and returns a dict of the results

    Args:
        patch (pandas.Series): Row of DataFrame representing the entry for a patch

    Returns:
        new_columns (dict): Dictionary with keys as class numbers and associated values of fractional size of class
                            plus a key-value pair for the patch ID
    """
    new_columns = {'PATCH': patch['PATCH']}
    for mode in patch['MODES']:
        new_columns[mode[0]] = mode[1] / (image_size[0] * image_size[1])

    return new_columns


def extract_patch_ids(scene):
    return scene[0]


def make_sorted_streams(patch_ids: list = None, scenes: list = None, func: callable = lc_load):
    """Creates a DataFrame with columns of patch IDs sorted for each class by class size in those patches.

    Args:
        patch_ids (list[str]): List of patch IDs defining the dataset to be sorted.

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


def cloud_cover(scene: np.ndarray):
    """Calculates percentage cloud cover for a given scene based on its scene CLD.

    Args:
        scene (np.ndarray): Cloud cover mask for a particular scene.

    Returns:
        (float): Percentage cloud cover of scene.
    """
    return np.sum(scene) / scene.size


def month_sort(df: pd.DataFrame, month: str):
    """Finds the the scene with the lowest cloud cover in a given month.

    Args:
        df (pd.DataFrame): Dataframe containing all scenes and their cloud cover percentages.
        month (str): Month of a year to sort.

    Returns:
        (str): Date of the scene with the lowest cloud cover percentage for the given month.
    """
    return df.loc[month].sort_values(by='COVER')['DATE'][0]


def ref_scene_select(df: pd.DataFrame, n_scenes: int = 12):
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


def threshold_scene_select(df: pd.DataFrame, thres: float = 0.3):
    """Selects all scenes in a patch with a cloud cover less than the threshold provided.

    Args:
        df (pd.DataFrame): Dataframe containing all scenes and their cloud cover percentages.
        thres (float): Optional; Fractional limit of cloud cover below which scenes shall be selected.

    Returns:
        List of strings representing dates of the selected scenes in YY_MM_DD format.
    """
    return df.loc[df['COVER'] < thres]['DATE'].tolist()


def find_best_of(patch_id: str, selector: callable = ref_scene_select, **kwargs):
    """Finds the 24 scenes sorted by cloud cover according to REF's 2-step criteria using scene_selection().

    Args:
        patch_id (str): Unique patch ID.
        selector (callable): Optional; Function to use to select scenes.
            Must take an appropriately constructed pd.DataFrame.
        **kwargs: Kwargs for func.

    Returns:
        scene_names (list): List of 24 strings representing dates of the 24 selected scenes in YY_MM_DD format.
    """
    # Creates a DataFrame
    patch = pd.DataFrame()

    # Using cloud_grab(), gets all the scene CLDs and dates for the given patch and adds to DataFrame
    patch['SCENE'], patch['DATE'] = cloud_grab(patch_id)

    # Calculates the cloud cover percentage for every scene and adds to DataFrame
    patch['COVER'] = patch['SCENE'].apply(cloud_cover)

    # Removes unneeded scene column
    del patch['SCENE']

    # Re-indexes the DataFrame to datetime
    patch.set_index(pd.to_datetime(patch['DATE'], format='%Y_%m_%d'), drop=True, inplace=True)

    # Sends DataFrame to scene_selection() and returns the 24 selected scenes
    return selector(patch, **kwargs)


def pair_production(patch_id: str, func: callable, **kwargs) -> list:
    """Creates pairs of patch ID and date of scene to define the scenes to load from a patch.

    Args:
        patch_id (str): Unique ID of the patch.
        func (callable): Optional; Function to use to select scenes.
            Must take an appropriately constructed pd.DataFrame.
        **kwargs: Kwargs for func.

    Returns:
        A list of tuples of pairs of patch ID and date of scene as strings.
    """
    scenes = find_best_of(patch_id, func, **kwargs)

    return [(patch_id, scene) for scene in scenes]


def scene_extract(patch_ids: list, *args, **kwargs):
    """Uses pair_production to produce patch ID - scene pairs for the whole dataset outlined by patch_ids.

    Args:
        patch_ids (list[str]):
        *args: Args for pair_production
        **kwargs: Kwargs for pair_production

    Returns:
        pairs (list[tuple[str, str]]): List of patch ID - scene pairs defining the dataset.
    """
    pairs = []
    for patch_id in patch_ids:
        # Loads pairs for given patch ID
        patch_pairs = pair_production(patch_id, *args, **kwargs)

        # Appends pairs to list one-by-one
        for pair in patch_pairs:
            pairs.append(pair)

    return pairs


def stack_bands(patch_id: str, scene: str):
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


def make_time_series(patch_id: str) -> np.ndarray:
    """Makes a time-series of each pixel of a patch across 24 scenes selected by REF's criteria using scene_selection().
     All the bands in the chosen scene are stacked using stack_bands().

    Args:
        patch_id (str): Unique patch ID.

    Returns:
        (np.ndarray): Array of shape(rows, columns, 24, 12) holding all x for a patch.
    """
    # List of scene dates found by REF's selection criteria
    scenes = find_best_of(patch_id)

    # Loads all pixels in a patch across the 24 scenes and 12 bands
    x = []
    for scene in scenes:
        x.append(stack_bands(patch_id, scene))

    # Returns a reordered np.ndarray holding all x for the given patch
    return np.moveaxis(np.array(x), 0, 2)


def load_patch_df(patch_id: str) -> pd.DataFrame:
    """Loads a patch using patch ID from disk into a Pandas.DataFrame and returns

    Args:
        patch_id (str): ID for patch to be loaded

    Returns:
        df (pandas.DataFrame): Patch loaded into a DataFrame
    """
    # Initialise DataFrame object
    df = pd.DataFrame()

    # Load patch from disk and create time-series pixel stacks
    patch = make_time_series(patch_id)

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


def find_subpopulations(labels, plot: bool = False):
    """Loads all LC labels for the given patches using lc_load() then finds the number of samples for each class.

    Args:
        labels (list or np.ndarray): Class labels describing the data to be analysed.
        plot (bool): Plots distribution of subpopulations if True.

    Returns:
        Modal distribution of classes in the dataset provided.
    """
    # Finds the distribution of the classes within the data
    class_dist = Counter(np.array(labels).flatten()).most_common()

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


def func_by_str(module: str, func: str):
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


def check_len(param, comparator):
    if hasattr(param, '__len__'):
        if len(param) == len(comparator):
            return param
        else:
            return [param[0]] * len(comparator)
    else:
        return [param] * len(comparator)
