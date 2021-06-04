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
    * Fully document

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.utils import visutils
import yaml
import os
import glob
import math
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, OrderedDict
import rasterio as rt
from osgeo import gdal, osr
import torch
from sklearn.model_selection import train_test_split
from torch.backends import cudnn

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'
dataset_config_path = '../../config/landcovernet.yml'

with open(config_path) as file:
    config = yaml.safe_load(file)

with open(dataset_config_path) as file:
    dataset_config = yaml.safe_load(file)

# Path to directory holding dataset
data_dir = config['dir']['data']

# Path to directory to output plots to
results_dir = os.path.join(*config['dir']['results'])

# Model Name
model_name = config['model_name']

# Prefix to every patch ID in every patch directory name
patch_dir_prefix = dataset_config['patch_dir_prefix']

# Band IDs of SENTINEL-2 images contained in the LandCoverNet dataset
band_ids = dataset_config['data_specs']['band_ids']

# Defines size of the images to determine the number of batches
image_size = dataset_config['data_specs']['image_size']

flattened_image_size = image_size[0] * image_size[1]

classes = dataset_config['classes']

cmap_dict = dataset_config['colours']

# Parameters
params = config['hyperparams']['params']


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_cuda_device():
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
    """Finds and loads all CLDs for a given patch.

    Args:
        patch_id (str): Unique patch ID.

    Returns:
        scenes (list): List of CLD masks for each scene.
        scene_names (list): List of scene dates in YY_MM_DD.
    """
    # Get the name of all the directories for this patch
    scene_dirs = glob.glob('{}{}{}{}{}*{}'.format(data_dir, os.sep, patch_dir_prefix, patch_id, os.sep, os.sep))

    # Extract the scene names (i.e the dates) from the paths
    scene_names = [(scene.partition(os.sep)[2].partition(os.sep)[2])[:-1] for scene in scene_dirs]

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


def date_grab(patch_id: str):
    """Finds all the name of all the scene directories for a patch and returns a list of the dates reformatted.

    Args:
        patch_id (str): Unique patch ID.

    Returns:
        (list): List of the dates of the scenes in DD.MM.YYYY format for this patch_ID.
    """
    # Get the name of all the directories for this patch
    scene_dirs = glob.glob('%s/%s%s/*/' % (data_dir, patch_dir_prefix, patch_id))

    # Extract the scene names (i.e the dates) from the paths
    scene_names = [(scene.partition('\\')[2])[:-1] for scene in scene_dirs]

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
    return load_array(os.sep.join([data_dir, patch_dir_prefix + patch_id, patch_id + '_2018_LC_10m.tif']), 1)


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


def split_data(patch_ids=None, split=(0.7, 0.15, 0.15), seed: int = 42, shuffle: bool = True,
               p_dist: bool = False, ctr_lbl: bool = False, plot: bool = False):
    """Splits the patch IDs into train, validation and test id sets.

    Args:
        patch_ids (list[str]): Optional; List of patch IDs that outline the whole dataset to be used. If not provided,
            the patch IDs are inferred from the directory using patch_grab.
        split (list[float] or tuple[float]): Optional; Three values giving the fractional sizes of the datasets, in the
            order (train, validation, test).
        seed (int): Optional; Random seed number to fix the shuffling of the data split.
        shuffle (bool): Optional; Whether to shuffle the patch IDs in the splitting of the IDs.
        p_dist (bool): Optional; Whether to print to screen the distribution of classes within each dataset.
        ctr_lbl (bool): Optional; Loads only centre labels for class distribution analysis.
        plot (bool): Optional; Whether or not to plot pie charts of the class distributions within each dataset.

    Returns:
        ids (dict): Dictionary of patch IDs representing train, validation and test datasets.
    """
    # Fetches all patch IDs in the dataset
    if patch_ids is None:
        patch_ids = patch_grab()

    # Splits the dataset into train and val-test
    train_ids, val_test_ids = train_test_split(patch_ids, train_size=split[0], test_size=(split[1] + split[2]),
                                               shuffle=shuffle, random_state=seed)

    # Splits the val-test dataset into validation and test
    val_ids, test_ids = train_test_split(val_test_ids, train_size=(split[1] / (split[1] + split[2])),
                                         test_size=(split[2] / (split[1] + split[2])), shuffle=shuffle,
                                         random_state=seed)

    # Prints the class sub-populations of each dataset to screen.
    if p_dist:
        if ctr_lbl:
            print('\nTrain: \n', find_subpopulations([find_centre_label(patch_id) for patch_id in train_ids],
                                                     plot=plot))
            print('\nValidation: \n', find_subpopulations([find_centre_label(patch_id) for patch_id in val_ids],
                                                          plot=plot))
            print('\nTest: \n', find_subpopulations([find_centre_label(patch_id) for patch_id in test_ids], plot=plot))

        else:
            print('\nTrain: \n', find_subpopulations(dataset_lc_load(train_ids), plot=plot))
            print('\nValidation: \n', find_subpopulations(dataset_lc_load(val_ids), plot=plot))
            print('\nTest: \n', find_subpopulations(dataset_lc_load(test_ids), plot=plot))

    ids = {'train': train_ids,
           'val': val_ids,
           'test': test_ids}

    return ids


def class_weighting(class_dist):
    # Finds total number of samples to normalise data
    n_samples = 0
    for mode in class_dist:
        n_samples += mode[1]

    class_weights = {}
    for mode in class_dist:
        class_weights[mode[0]] = 1.0 / mode[1]

    return class_weights


def weight_samples(patch_ids, func=find_centre_label, class_weights=None):
    if class_weights is None:
        class_weights = class_weighting(find_subpopulations(dataset_lc_load(patch_ids, func), plot=False))

    sample_weights = []

    for patch in patch_ids:
        for i in range(24):
            sample_weights.append(class_weights[find_centre_label(patch)])

    return sample_weights


def find_empty_classes(patch_ids: list, func=find_centre_label, class_dist=None):
    if class_dist is None:
        class_dist = find_subpopulations(dataset_lc_load(patch_ids, func), plot=False)

    empty = []
    for label in classes.keys():
        if label not in [mode[0] for mode in class_dist]:
            empty.append(label)

    return empty


def eliminate_classes(empty_classes):
    # Makes deep copies of the class and cmap dicts.
    new_classes = {key: value[:] for key, value in classes.items()}
    new_colours = {key: value[:] for key, value in cmap_dict.items()}

    # Deletes empty classes from copied dicts.
    for label in empty_classes:
        del new_classes[label]
        del new_colours[label]

    over_keys = [key for key in new_classes.keys() if key >= len(new_classes.keys())]

    over_classes = OrderedDict({key: new_classes[key] for key in over_keys})
    over_colours = OrderedDict({key: new_colours[key] for key in over_keys})

    reordered_classes = {}
    reordered_colours = {}
    conversion = {}

    for i in range(len(new_classes.keys())):
        if i in new_classes:
            reordered_classes[i] = new_classes[i]
            reordered_colours[i] = new_colours[i]
            conversion[i] = i

        if i not in new_classes:
            class_key, class_value = over_classes.popitem()
            colour_key, colour_value = over_colours.popitem()

            reordered_classes[i] = class_value
            reordered_colours[i] = colour_value

            conversion[class_key] = i

    return reordered_classes, conversion, reordered_colours


def class_transform(label, matrix):
    return matrix[label]


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


def make_sorted_streams(patch_ids):
    """Creates a DataFrame with columns of patch IDs sorted for each class by class size in those patches.

    Args:
        patch_ids (list[str]): List of patch IDs defining the dataset to be sorted.

    Returns:
        streams_df (pd.DataFrame): Database of list of patch IDs sorted by fractional sizes of class labels.
    """
    df = pd.DataFrame()
    df['PATCH'] = patch_ids

    # Calculates the class modes of each patch.
    df['MODES'] = df['PATCH'].apply(find_patch_modes)

    # Calculates the fractional size of each class in each patch.
    df = pd.DataFrame([row for row in df.apply(class_frac, axis=1)])

    df.fillna(0, inplace=True)

    class_dist = find_subpopulations(dataset_lc_load(df['PATCH']), plot=False)

    stream_size = int(len(df['PATCH']) / len(classes))

    streams = {}

    for mode in reversed(class_dist):
        stream = df.sort_values(by=mode[0], ascending=False)['PATCH'][:stream_size]
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


def scene_selection(df: pd.DataFrame):
    """Selects the 24 best scenes of a patch based on REF's 2-step selection criteria.

    Args:
        df (pd.DataFrame): Dataframe containing all scenes and their cloud cover percentages.

    Returns:
        scene_names (list): List of 24 strings representing dates of the 24 selected scenes in YY_MM_DD format.
    """
    # Step 1: Find scene with lowest cloud cover percentage in each month
    step1 = []
    for month in range(1, 13):
        step1.append(month_sort(df, '%d-2018' % month))

    # Step 2: Find the 12 scenes with the lowest cloud cover percentage of the remaining scenes
    df.drop(index=pd.to_datetime(step1, format='%Y_%m_%d'), inplace=True)
    step2 = df.sort_values(by='COVER')['DATE'][:12].tolist()

    # Return 24 scenes selected by the 2-step REF criteria
    return step1 + step2


def find_best_of(patch_id: str):
    """Finds the 24 scenes sorted by cloud cover according to REF's 2-step criteria using scene_selection().

    Args:
        patch_id (str): Unique patch ID.

    Returns:
        scene_names (list): List of 24 strings representing dates of the 24 selected scenes in YY_MM_DD format.
    """
    # Creates a DataFrame
    patch = pd.DataFrame()

    # Using scene_grab(), gets all the scene CLDs and dates for the given patch and adds to DataFrame
    patch['SCENE'], patch['DATE'] = scene_grab(patch_id)

    # Calculates the cloud cover percentage for every scene and adds to DataFrame
    patch['COVER'] = patch['SCENE'].apply(cloud_cover)

    # Removes unneeded scene column
    del patch['SCENE']

    # Re-indexes the DataFrame to datetime
    patch.set_index(pd.to_datetime(patch['DATE'], format='%Y_%m_%d'), drop=True, inplace=True)

    # Sends DataFrame to scene_selection() and returns the 24 selected scenes
    return scene_selection(patch)


def pair_production(patch_id: str, func) -> list:
    scenes = func(patch_id)

    return [(patch_id, scene) for scene in scenes]


def scene_extract(patch_ids: list, *args, **kwargs):
    pairs = []
    for patch_id in patch_ids:
        patch_pairs = pair_production(patch_id, *args, **kwargs)
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
    """Loads all LC labels for the given patches using lc_load() then finds the number of samples for each class

    Args:
        labels (list or np.ndarray): Class labels describing the data to be analysed
        plot (bool): Plots distribution of subpopulations if True

    Returns:
        Modal distribution of classes in the dataset provided
    """
    # Finds the distribution of the classes within the data
    class_dist = Counter(np.array(labels).flatten()).most_common()

    if plot:
        # Plots a pie chart of the distribution of the classes within the given list of patches
        visutils.plot_subpopulations(class_dist, class_names=classes, cmap_dict=cmap_dict, save=False, show=True)

    return class_dist


def num_batches(num_ids: int) -> int:
    """Determines the number of batches needed to cover the dataset across ids

    Args:
        num_ids (int): Number of patch IDs in the dataset to be loaded in by batches

    Returns:
        num_batches (int): Number of batches needed to cover the whole dataset
    """
    return int((num_ids * image_size[0] * image_size[1]) / params['batch_size'])
