"""utils

Module to handle all utility functions for training, testing and evaluation of a model

TODO:
    * Fully document

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import yaml
import os
import glob
import math
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from collections import Counter
import seaborn as sns
import tensorflow as tf
import rasterio as rt
from osgeo import gdal, osr

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = 'config.yml'
dataset_config_path = 'landcovernet.yml'

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
def exist_delete_check(fn):
    """Checks if given file exists then deletes if true

    Args:
        fn (str): Path to file to have existence checked then deleted

    Returns:
        None

    """
    # Checks if file exists. Deletes if True. No action taken if False
    if os.path.exists(fn):
        os.remove(fn)
    else:
        pass


def datetime_reformat(timestamp, fmt1, fmt2):
    """Takes a str representing a time stamp in one format and returns it reformatted into a second

    Args:
        timestamp (str): Datetime string to be reformatted
        fmt1 (str): Format of original datetime
        fmt2 (str): New format for datetime

    Returns:
        (str): Datetime reformatted to fmt2
    """
    return datetime.strptime(timestamp, fmt1).strftime(fmt2)


def prefix_format(patch_id, scene):
    """Formats a string representing the prefix of a path to any file in a scene

    Args:
        patch_id (str): Unique patch ID
        scene (str): Date of scene in YY_MM_DD format

    Returns:
        prefix (str): Prefix of path to any file in a given scene
    """
    return os.sep.join([data_dir, patch_dir_prefix + patch_id, scene, patch_id + '_' +
                       datetime_reformat(scene, '%Y_%m_%d', '%Y%m%d')])


def scene_grab(patch_id):
    """Finds and loads all CLDs for a given patch

    Args:
        patch_id (str): Unique patch ID

    Returns:
        scenes (list): List of CLD masks for each scene
        scene_names (list): List of scene dates in YY_MM_DD

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
    """Fetches the patch IDs from the directory holding the whole dataset

    Returns:
        (list): List of unique patch IDs

    """
    # Fetches the names of the all the patch directories in the dataset
    patch_dirs = glob.glob('%s/%s*/' % (data_dir, patch_dir_prefix))

    # Extracts the patch ID from the directory names and returns the list
    return [(patch.partition(patch_dir_prefix)[2])[:-1] for patch in patch_dirs]


def date_grab(patch_id):
    """Finds all the name of all the scene directories for a patch and returns a list of the dates reformatted

    Args:
        patch_id (str): Unique patch ID

    Returns:
        (list): List of the dates of the scenes in DD.MM.YYYY format for this patch_ID
    """
    # Get the name of all the directories for this patch
    scene_dirs = glob.glob('%s/%s%s/*/' % (data_dir, patch_dir_prefix, patch_id))

    # Extract the scene names (i.e the dates) from the paths
    scene_names = [(scene.partition('\\')[2])[:-1] for scene in scene_dirs]

    # Format the dates from US YYYY_MM_DD format into UK DD.MM.YYYY format and return list
    return [datetime_reformat(date, '%Y_%m_%d', '%d.%m.%Y') for date in scene_names]


def load_array(path, band):
    """Extracts an array from opening a specific band of a .tif file

    Args:
        path (str): Path to file
        band (int): Band number of .tif file

    Returns:
        data ([[float]]): 2D array representing the image from the .tif band requested

    """
    raster = rt.open(path)

    data = raster.read(band)

    return data


def lc_load(patch_id):
    """Loads the LC labels for a given patch

    Args:
        patch_id (str): Unique patch ID

    Returns:
        LC_label (list): 2D array containing LC labels for each pixel of a patch

    """
    return load_array(os.sep.join([data_dir, patch_dir_prefix + patch_id, patch_id + '_2018_LC_10m.tif']), 1)


def transform_coordinates(path, new_cs):
    """Extracts the co-ordinates of a GeoTiff file from path and returns the co-ordinates of the corners of that file
    in the new co-ordinates system provided

    Args:
        path (str): Path to GeoTiff to extract and transform co-ordinates from
        new_cs(SpatialReference): Co-ordinate system to convert GeoTiff co-ordinates from

    Returns:
        ([[tuple]]): The corners of the image in the new co-ordinate system
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


def deg_to_dms(deg, axis='lat'):
    """Credit to Gustavo Gonçalves on Stack Overflow
    https://stackoverflow.com/questions/2579535/convert-dd-decimal-degrees-to-dms-degrees-minutes-seconds-in-python

    Args:
        deg (float): Decimal degrees of latitude or longitude
        axis (str): Identifier between latitude ('lat') or longitude ('lon') for N-S, E-W direction identifier

    Returns:
        str of inputted deg in degrees, minutes and seconds in the form DegreesºMinutes Seconds Hemisphere
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
    """Wrapper for deg_to_dms

    Args:
        dec_co ([float]): Array of either latitude or longitude co-ordinates in decimal degrees
        axis (str): Identifier between latitude ('lat') or longitude ('lon') for N-S, E-W direction identifier

    Returns:
        deg_co ([str]): List of formatted strings in degrees, minutes and seconds
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
    """Creates a DataFrame with columns of patch IDs sorted for each class by class size in those patches

    Args:
        patch_ids (list[str]):

    Returns:
        streams_df (pandas.DataFrame): Database of list of patch IDs sorted by fractional sizes of class labels

    """
    df = pd.DataFrame()
    df['PATCH'] = patch_ids

    df['MODES'] = df['PATCH'].apply(find_patch_modes)

    df = pd.DataFrame([row for row in df.apply(class_frac, axis=1)])

    df.fillna(0, inplace=True)

    class_dist = find_subpopulations(df['PATCH'], plot=False)

    stream_size = int(len(df['PATCH']) / len(classes))

    streams = {}

    for mode in reversed(class_dist):
        stream = df.sort_values(by=mode[0], ascending=False)['PATCH'][:stream_size]
        streams[mode[0]] = stream.tolist()
        df.drop(stream.index, inplace=True)

    streams_df = pd.DataFrame(streams)

    return streams_df


def cloud_cover(scene):
    """Calculates percentage cloud cover for a given scene based on its scene CLD

    Args:
        scene (numpy.ndarray):

    Returns:
        (float): Percentage cloud cover
    """
    return np.sum(scene) / scene.size


def month_sort(df, month):
    """Finds the the scene with the lowest cloud cover in a given month

    Args:
        df (pandas.DataFrame): Dataframe containing all scenes and their cloud cover percentages
        month (str): Month of a year to sort

    Returns:
        (str): Date of the scene with the lowest cloud cover percentage for the given month
    """
    return df.loc[month].sort_values(by='COVER')['DATE'][0]


def scene_selection(df):
    """Selects the 24 best scenes of a patch based on REF's 2-step selection criteria

    Args:
        df (pandas.DataFrame): Dataframe containing all scenes and their cloud cover percentages

    Returns:
        scene_names (list): List of 24 strings representing dates of the 24 selected scenes in YY_MM_DD format
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


def find_best_of(patch_id):
    """Finds the 24 scenes sorted by cloud cover according to REF's 2-step criteria using scene_selection()

    Args:
        patch_id (str): Unique patch ID

    Returns:
        scene_names (list): List of 24 strings representing dates of the 24 selected scenes in YY_MM_DD format
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


def stack_bands(patch_id, scene):
    """Stacks together all the bands of the SENTINEL-2 images in a given scene of a patch

    Args:
        patch_id (str): Unique patch ID
        scene (str): Date of scene in YY_MM_DD format to stack bands in

    Returns:
        Normalised and stacked red, green, blue arrays into RGB array
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


def make_time_series(patch_id):
    """Makes a time-series of each pixel of a patch across 24 scenes selected by REF's criteria using scene_selection().
     All the bands in the chosen scene are stacked using stack_bands()

    Args:
        patch_id (str): Unique patch ID

    Returns:
        (numpy.ndarray): Array of shape(rows, columns, 24, 12) holding all x for a patch
    """
    # List of scene dates found by REF's selection criteria
    scenes = find_best_of(patch_id)

    # Loads all pixels in a patch across the 24 scenes and 12 bands
    x = []
    for scene in scenes:
        x.append(stack_bands(patch_id, scene))

    # Returns a reordered numpy.ndarray holding all x for the given patch
    return np.moveaxis(np.array(x), 0, 2)


def find_subpopulations(ids, plot=False):
    """Loads all LC labels for the given patches using lc_load() then finds the number of samples for each class

    Args:
        ids (list): List of patch IDs to analyse
        plot (bool): Plots distribution of subpopulations if True

    Returns:
        class_dist (Counter): Modal distribution of classes in the dataset provided
    """
    # Loads all LC label masks for the given patch IDs
    labels = []
    for patch_id in ids:
        labels.append(lc_load(patch_id))

    if plot:
        # Plots a pie chart of the distribution of the classes within the given list of patches
        plot_subpopulations(np.array(labels).flatten(), class_names=classes, cmap=cmap_dict)

    # Finds the distribution of the classes within the data
    return Counter(np.array(labels).flatten()).most_common()


def plot_subpopulations(class_labels, class_names=None, cmap=None, filename=None, save=True, show=False):
    """Creates a pie chart of the distribution of the classes within the data

    Args:
        class_labels (np.array[int]): List of class labels
        class_names (dict): Dictionary mapping class labels to class names
        cmap (dict): Dictionary mapping class labels to class colours
        filename (str): Name of file to save plot to
        show (bool): Whether to show plot
        save (bool): Whether to save plot to file

    Returns:
        None
    """

    # Finds the distribution of the classes within the data
    class_dist = Counter(class_labels).most_common()

    # List to hold the name and percentage distribution of each class in the data as str
    class_data = []

    # List to hold the total counts of each class
    counts = []

    # List to hold colours of classes in the correct order
    colours = []

    # Finds total number of samples to normalise data
    n_samples = 0
    for mode in class_dist:
        n_samples += mode[1]

    # For each class, find the percentage of data that is that class and the total counts for that class
    for label in class_dist:
        # Sets percentage label to <0.01% for classes matching that equality
        if (label[1] * 100.0 / n_samples) > 0.01:
            class_data.append('{} \n{:.2f}%'.format(class_names[label[0]], (label[1] * 100.0 / n_samples)))
        else:
            class_data.append('{} \n<0.01%'.format(class_names[label[0]]))
        counts.append(label[1])
        colours.append(cmap[label[0]])

    # Locks figure size
    plt.figure(figsize=(6, 5))

    # Plot a pie chart of the data distribution amongst the classes
    patches, text = plt.pie(counts, colors=colours, explode=[i * 0.05 for i in range(len(class_data))])

    # Adds legend
    plt.legend(patches, class_data, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    # Shows and/or saves plot
    if show:
        plt.show()
    if save:
        plt.savefig(filename)
        plt.close()


def num_batches(num_ids):
    """Determines the number of batches needed to cover the dataset across ids

    Args:
        num_ids (int): Number of patch IDs in the dataset to be loaded in by batches

    Returns:
        num_batches (int): Number of batches needed to cover the whole dataset
    """
    return int((num_ids * image_size[0] * image_size[1]) / params['batch_size'])


def plot_history(metrics, filename=None, save=True, show=False):
    """Plots model history based on metrics supplied

    Args:
        metrics (dict): Dictionary containing the names and results of the metrics by which model was assessed
        filename (str): Name of file to save plot to
        show (bool): Whether to show plot
        save (bool): Whether to save plot to file

    Returns:
        None
    """""
    # Initialise figure
    plt.figure()

    # Plots each metric in metrics, appending their artist handles
    handles = []
    for metric in metrics.values():
        handles.append(plt.plot(metric)[0])

    # Creates legend from plot artist handles and names of metrics
    plt.legend(handles=handles, labels=metrics.keys())

    # Adds axis labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')

    # Shows and/or saves plot
    if show:
        plt.show()
    if save:
        plt.savefig(filename)
        plt.close()


def make_confusion_matrix(test_pred, test_labels, filename=None, show=True, save=False):
    """Creates a heat-map of the confusion matrix of the given model

    Args:
        test_pred([[int]]): Predictions made by model on test images
        test_labels ([[int]]): Accompanying labels for testing images
        filename (str): Name of file to save plot to
        show (bool): Whether to show plot
        save (bool): Whether to save plot to file

    Returns:
        None
    """
    # Creates the confusion matrix based on these predictions and the corresponding ground truth labels
    cm = tf.math.confusion_matrix(labels=test_labels, predictions=test_pred).numpy()

    # Normalises confusion matrix
    cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    np.nan_to_num(cm_norm, copy=False)

    # Extract class names from dict in numeric order to ensure labels match matrix
    class_names = [classes[key] for key in range(len(classes.keys()))]

    # Converts confusion matrix to Pandas.DataFrame
    cm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)

    # Plots figure
    plt.figure()
    sns.heatmap(cm_df, annot=True, square=True, cmap=plt.cm.get_cmap('Blues'), vmin=0.0, vmax=1.0)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')

    # Shows and/or saves plot
    if show:
        plt.show()
    if save:
        plt.savefig(filename)
        plt.close()


def timestamp_now(fmt='%d-%m-%Y_%H%M'):
    return datetime.now().strftime(fmt)


def format_plot_names():
    def standard_format(plot_type, file_ext):
        filename = '{}_{}_{}.{}'.format(model_name, plot_type, timestamp, file_ext)
        return os.path.join(results_dir, filename)

    timestamp = timestamp_now(fmt='%d-%m-%Y_%H%M')

    filenames = {'History': standard_format('MH', 'png'),
                 'Pred': standard_format('TP', 'png'),
                 'CM': standard_format('CM', 'png')}

    return filenames


def plot_results(metrics, z, y, save=True, show=False):
    filenames = format_plot_names()

    plot_history(metrics, filename=filenames['History'], save=save, show=show)

    plot_subpopulations(z, class_names=classes, cmap=cmap_dict, filename=filenames['Pred'], save=save, show=show)

    make_confusion_matrix(test_labels=y, test_pred=z, filename=filenames['CM'], save=save, show=show)

