"""utils

Script to handle all utility functions for training, testing and evaluation of a model

TODO:

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import yaml
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import Radiant_MLHub_DataVis as rdv
from matplotlib import pyplot as plt
from collections import Counter
import seaborn as sns
import tensorflow as tf

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = 'config.yml'

with open(config_path) as file:
    config = yaml.safe_load(file)

# Path to directory holding dataset
data_dir = config['dir']['data']

# Path to directory to output plots to
results_dir = os.path.join(*config['dir']['results'])

# Model Name
model_name = config['model_name']

# Prefix to every patch ID in every patch directory name
patch_dir_prefix = config['patch_dir_prefix']

# Band IDs of SENTINEL-2 images contained in the LandCoverNet dataset
band_ids = config['data_specs']['band_ids']

# Defines size of the images to determine the number of batches
image_size = config['data_specs']['image_size']

flattened_image_size = image_size[0] * image_size[1]

classes = rdv.RE_classes

# Parameters
params = config['hyperparams']['params']


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def prefix_format(patch_id, scene):
    """Formats a string representing the prefix of a path to any file in a scene

    Args:
        patch_id (str): Unique patch ID
        scene (str): Date of scene in YY_MM_DD format

    Returns:
        prefix (str): Prefix of path to any file in a given scene
    """
    return os.sep.join([data_dir, patch_dir_prefix + patch_id, scene, patch_id + '_' +
                       rdv.datetime_reformat(scene, '%Y_%m_%d', '%Y%m%d')])


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
        scenes.append(rdv.load_array('%s_CLD_10m.tif' % prefix_format(patch_id, date), 1))

    return scenes, scene_names


def lc_load(patch_id):
    """Loads the LC labels for a given patch

    Args:
        patch_id (str): Unique patch ID

    Returns:
        LC_label (list): 2D array containing LC labels for each pixel of a patch

    """
    return rdv.load_array(os.sep.join([data_dir, patch_dir_prefix + patch_id, patch_id + '_2018_LC_10m.tif']), 1)


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
        image = rdv.load_array('%s_%s_10m.tif' % (prefix_format(patch_id, scene), band), 1).astype('float')
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
        plot_subpopulations(np.array(labels).flatten(), class_names=rdv.RE_classes, cmap=rdv.RE_cmap_dict)

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


def format_plot_names():
    def standard_format(plot_type, file_ext):
        filename = '{}_{}_{}.{}'.format(model_name, plot_type, timestamp, file_ext)
        return os.path.join(results_dir, filename)

    timestamp = datetime.now().strftime('%d-%m-%Y_%H%M')

    filenames = {'History': standard_format('MH', 'png'),
                 'Pred': standard_format('TP', 'png'),
                 'CM': standard_format('CM', 'png')}

    return filenames


def plot_results(metrics, z, y, save=True, show=False):
    filenames = format_plot_names()

    plot_history(metrics, filename=filenames['History'], save=save, show=show)

    plot_subpopulations(z, class_names=classes, cmap=rdv.RE_cmap_dict, filename=filenames['Pred'], save=save, show=show)

    make_confusion_matrix(test_labels=y, test_pred=z, filename=filenames['CM'], save=save, show=show)

