"""visutils

Module to visualise .tiff images and associated label masks downloaded from the Radiant MLHub API.

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
    * Add ability to plot labelled RGB images using the annual land cover labels
    * Add option to append annual land cover mask to patch GIFs

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.utils import utils
import os
import yaml
import imageio
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
import cv2
from alive_progress import alive_bar

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'
lcn_config_path = '../../config/landcovernet.yml'

with open(config_path) as file:
    config = yaml.safe_load(file)

with open(lcn_config_path) as file:
    lcn_config = yaml.safe_load(file)

# Path to directory holding dataset
data_dir = config['dir']['data']

# Prefix to every patch ID in every patch directory name
patch_dir_prefix = lcn_config['patch_dir_prefix']

# Automatically fixes the layout of the figures to accommodate the colour bar legends
plt.rcParams['figure.constrained_layout.use'] = True

# Increases DPI to avoid strange plotting errors for class heatmaps.
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Downloads required plugin for imageio if not already present
imageio.plugins.freeimage.download()

manifest = pd.read_csv(utils.get_manifest())


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def path_format(names):
    """Takes a dictionary of unique IDs to format the paths and names of files associated with the desired scene

    Args:
        names (dict): Dictionary of IDs to uniquely identify the scene and selected bands

    Returns:
        scene_path (str): Path to directory holding images from desired scene
        rgb (dict): Dictionary of filenames of R, G & B band images
        data_name (str): Name of the file containing the label mask
    """
    # Format the two required date formats used by REF MLHub
    date1 = utils.datetime_reformat(names['date'], '%d.%m.%Y', '%Y_%m_%d')
    date2 = utils.datetime_reformat(names['date'], '%d.%m.%Y', '%Y%m%d')

    # Format path to the directory holding all the scene files
    scene_path = "{}{}".format(os.sep.join((*data_dir, patch_dir_prefix + names['patch_ID'], date1)), os.sep)

    # Format the name of the file containing the label mask data
    scene_data_name = '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['band_ID'])
    patch_data_name = utils.get_label_path(names['patch_ID'])

    # Create a dictionary of the names of the requested red, green, blue images
    rgb = {'R': '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['R_band']),
           'G': '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['G_band']),
           'B': '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['B_band'])}

    return rgb, scene_path, scene_data_name, patch_data_name


def deinterlace(x, f):
    new_x = []
    for i in range(f):
        x_i = []
        for j in np.arange(start=i, stop=len(x), step=f):
            x_i.append(x[j])
        new_x.append(np.array(x_i).flatten())

    return np.array(new_x).flatten()


def discrete_heatmap(data, classes=None, cmap_style=None):
    """Plots a heatmap with a discrete colour bar. Designed for Radiant Earth MLHub 256x256 SENTINEL images

    Args:
        data (array_like): 2D Array of data to be plotted as a heat map
        classes ([str]): List of all possible class labels
        cmap_style (str, ListedColormap): Name or object for colour map style

    Returns:
        None

    """
    # Initialises a figure
    plt.figure()

    # Creates a cmap from query
    cmap = plt.get_cmap(cmap_style, len(classes))

    # Plots heatmap onto figure
    heatmap = plt.imshow(data, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5)

    # Sets tick intervals to standard 32x32 block size
    plt.xticks(np.arange(0, data.shape[0] + 1, 32))
    plt.yticks(np.arange(0, data.shape[1] + 1, 32))

    # Add grid overlay
    plt.grid(which='both', color='#CCCCCC', linestyle=':')

    # Plots colour bar onto figure
    clb = plt.colorbar(heatmap, ticks=np.arange(0, len(classes)), shrink=0.77)

    # Sets colour bar ticks to class labels
    clb.ax.set_yticklabels(classes)

    # Display figure
    plt.show()

    # Close figure
    plt.close()


def stack_rgb(scene_path, rgb):
    """Stacks together red, green and blue image arrays from file to create a RGB array

    Args:
        scene_path (str): Path to directory holding images from desired scene
        rgb (dict): Dictionary of filenames of R, G & B band images

    Returns:
        Normalised and stacked red, green, blue arrays into RGB array
    """

    # Load R, G, B images from file and normalise
    bands = []
    for band in ['R', 'G', 'B']:
        img = utils.load_array(scene_path + rgb[band], 1)
        norm = np.zeros((img.shape[0], img.shape[1]))
        bands.append(cv2.normalize(img, norm, 0, 255, cv2.NORM_MINMAX))

    # Stack together RGB bands
    # Note that it has to be order BGR not RGB due to the order numpy stacks arrays
    return np.dstack((bands[2], bands[1], bands[0]))


def make_rgb_image(scene_path, rgb):
    """Creates an RGB image from a composition of red, green and blue band .tif images

    Args:
        scene_path (str): Path to directory holding images from desired scene
        rgb ([str]): List of filenames of R, G & B band images

    Returns:
        rgb_image (AxesImage): Plotted RGB image object
    """
    # Stack RGB image data together
    rgb_image_array = stack_rgb(scene_path, rgb)

    # Create RGB image
    rgb_image = plt.imshow(rgb_image_array)

    # Sets tick intervals to standard 32x32 block size
    plt.xticks(np.arange(0, rgb_image_array.shape[0] + 1, 32))
    plt.yticks(np.arange(0, rgb_image_array.shape[1] + 1, 32))

    # Add grid overlay
    plt.grid(which='both', color='#CCCCCC', linestyle=':')

    plt.show()

    return rgb_image


def labelled_rgb_image(names, mode: str = 'patch', data_band=1, classes=None, block_size=32, cmap_style=None, alpha=0.5, new_cs=None,
                       show=True, save=True, figdim=(8.02, 10.32)):
    """Produces a layered image of an RGB image and it's associated label mask heat map alpha blended on top

    Args:
        names (dict): Dictionary of IDs to uniquely identify the scene and selected bands
        data_band (int): Band number of data .tif file
        classes ([str]): List of all possible class labels
        block_size (int): Size of block image sub-division in pixels
        cmap_style (str, ListedColormap): Name or object for colour map style
        alpha (float): Fraction determining alpha blending of label mask
        new_cs(SpatialReference): Co-ordinate system to convert image to and use for labelling
        show (bool): True for show figure when plotted. False if not
        save (bool): True to save figure to file. False if not
        figdim (tuple): Figure (height, width) in inches

    Returns:
        fn (str): Path to figure save location

    """
    # Get required formatted paths and names
    rgb, scene_path, scene_data_name, patch_data_name = path_format(names)

    data_name = scene_path + scene_data_name
    if mode == 'patch':
        data_name = patch_data_name

    # Stacks together the R, G, & B bands to form an array of the RGB image
    rgb_image = stack_rgb(scene_path, rgb)

    # Loads data to plotted as heatmap from file
    data = utils.load_array(data_name, band=data_band)

    # Defines the 'extent' of the composite image based on the size of the mask.
    # Assumes mask and RGB image have same 2D shape
    extent = 0, data.shape[0], 0, data.shape[1]

    # Initialises a figure
    fig, ax1 = plt.subplots()

    # Create RGB image
    ax1.imshow(rgb_image, extent=extent)

    # Creates a cmap from query
    cmap = plt.get_cmap(cmap_style, len(classes))

    # Plots heatmap onto figure
    heatmap = ax1.imshow(data, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5, extent=extent, alpha=alpha)

    # Sets tick intervals to standard 32x32 block size
    ax1.set_xticks(np.arange(0, data.shape[0] + 1, block_size))
    ax1.set_yticks(np.arange(0, data.shape[1] + 1, block_size))

    # Creates a secondary x and y axis to hold lat-lon
    ax2 = ax1.twiny().twinx()

    # Gets the co-ordinates of the corners of the image in decimal lat-lon
    corners = utils.transform_coordinates(data_name, new_cs)

    # Creates a discrete mapping of the block size ticks to latitude longitude extent of the image
    lat_extent = np.linspace(start=corners[1][1][0], stop=corners[0][1][0],
                             num=int(data.shape[0]/block_size) + 1, endpoint=True)
    lon_extent = np.linspace(start=corners[0][0][1], stop=corners[0][1][1],
                             num=int(data.shape[0]/block_size) + 1, endpoint=True)

    # Plots an invisible line across the diagonal of the image to create the secondary axis for lat-lon
    ax2.plot(lon_extent, lat_extent, ' ',
             clip_box=Bbox.from_extents(lon_extent[0], lat_extent[0], lon_extent[-1], lat_extent[-1]))

    # Sets ticks for lat-lon
    ax2.set_xticks(lon_extent)
    ax2.set_yticks(lat_extent)

    # Sets the limits of the secondary axis so they should align with the primary
    ax2.set_xlim(left=lon_extent[0], right=lon_extent[-1])
    ax2.set_ylim(top=lat_extent[-1], bottom=lat_extent[0])

    # Converts the decimal lat-lon into degrees, minutes, seconds to label the axis
    lat_labels = utils.dec2deg(lat_extent, axis='lat')
    lon_labels = utils.dec2deg(lon_extent, axis='lon')

    # Sets the secondary axis tick labels
    ax2.set_xticklabels(lon_labels, fontsize=11)
    ax2.set_yticklabels(lat_labels, fontsize=10, rotation=-30, ha='left')

    # Add grid overlay
    ax1.grid(which='both', color='#CCCCCC', linestyle=':')

    # Plots colour bar onto figure
    clb = plt.colorbar(heatmap, ticks=np.arange(0, len(classes)), shrink=0.9, aspect=75, drawedges=True)

    # Sets colour bar ticks to class labels
    clb.ax.set_yticklabels(classes, fontsize=11)

    # Bodge to get a figure title by using the colour bar title.
    clb.ax.set_title('%s\n%s\nLand Cover' % (names['patch_ID'], names['date']), loc='left', fontsize=15)

    # Set axis labels
    ax1.set_xlabel('(x) - Pixel Position', fontsize=14)
    ax1.set_ylabel('(y) - Pixel Position', fontsize=14)
    ax2.set_ylabel('Latitude', fontsize=14, rotation=270, labelpad=12)
    ax2.set_title('Longitude')  # Bodge

    # Manual trial and error fig size which fixes aspect ratio issue
    fig.set_figheight(figdim[0])
    fig.set_figwidth(figdim[1])

    # Display figure
    if show:
        plt.show()

    # Path and file name of figure
    fn = '%s/%s_%s_RGBHM.png' % (scene_path, names['patch_ID'], utils.datetime_reformat(names['date'],
                                                                                        '%d.%m.%Y', '%Y%m%d'))

    # If true, save file to fn
    if save:
        # Checks if file already exists. Deletes if true
        utils.exist_delete_check(fn)

        # Save figure to fn
        fig.savefig(fn)

    # Close figure
    plt.close()

    return fn


def make_gif(names, gif_name, frame_length=1.0, data_band=1, classes=None, cmap_style=None, new_cs=None, alpha=0.5,
             save=False, figdim=(8.02, 10.32)):
    """Wrapper to labelled_rgb_image() to make a GIF for a patch out of scenes

    Args:
        names (dict): Dictionary of IDs to uniquely identify the patch dir and selected bands
        gif_name (str): Path to and name of GIF to be made
        frame_length (float): Length of each GIF frame in seconds
        data_band (int): Band number of data .tif file
        classes ([str]): List of all possible class labels
        cmap_style (str, ListedColormap): Name or object for colour map style
        new_cs(SpatialReference): Co-ordinate system to convert image to and use for labelling
        alpha (float): Fraction determining alpha blending of label mask
        save (bool): True to save figure to file. False if not
        figdim (tuple): Figure (height, width) in inches

    Returns:
        None

    """
    # Fetch all the scene dates for this patch in DD.MM.YYYY format
    dates = utils.date_grab(names['patch_ID'])

    # Initialise progress bar
    with alive_bar(len(dates), bar='blocks') as bar:

        # List to hold filenames and paths of images created
        frames = []
        for date in dates:
            # Update progress bar with current scene
            bar.text('SCENE ON %s' % date)

            # Update names date field
            names['date'] = date

            # Create a frame of the GIF for a scene of the patch
            frame = labelled_rgb_image(names, data_band=data_band, classes=classes, cmap_style=cmap_style,
                                       new_cs=new_cs, alpha=alpha, save=save, show=False, figdim=figdim)

            # Read in frame just created and add to list of frames
            frames.append(imageio.imread(frame))

            # Update bar with step completion
            bar()

    # Create a 'unknown' bar to 'spin' while the GIF is created
    with alive_bar(unknown='waves') as bar:
        # Add current operation to spinner bar
        bar.text('MAKING PATCH %s GIF' % names['patch_ID'])

        # Create GIF
        imageio.mimsave(gif_name, frames, 'GIF-FI', duration=frame_length, quantizer='nq')


def make_all_the_gifs(names, frame_length=1.0, data_band=1, classes=None, cmap_style=None, new_cs=None, alpha=0.5,
                      figdim=(8.02, 10.32)):
    """Wrapper to make_gifs() to iterate through all patches in dataset

    Args:
        names (dict): Dictionary holding the band IDs. Patch ID and date added per iteration
        frame_length (float): Length of each GIF frame in seconds
        data_band (int): Band number of data .tif file
        classes ([str]): List of all possible class labels
        cmap_style (str, ListedColormap): Name or object for colour map style
        new_cs(SpatialReference): Co-ordinate system to convert image to and use for labelling
        alpha (float): Fraction determining alpha blending of label mask
        figdim (tuple): Figure (height, width) in inches

    Returns:
        None

    """
    # Gets all the patch IDs from the dataset directory
    patches = utils.patch_grab()

    # Iterator for progress counter
    i = 0

    # Iterate through all patches
    for patch in patches:
        # Count this iteration for the progress counter
        i = i + 1

        # Print status update
        print('\r\nNOW SERVING PATCH %s (%s/%s): ' % (patch, i, len(patches)))

        # Update dictionary for this patch
        names['patch_ID'] = patch

        # Define name of GIF for this patch
        gif_name = '%s/%s%s/%s.gif' % (data_dir, patch_dir_prefix, names['patch_ID'], names['patch_ID'])

        # Call make_gif() for this patch
        make_gif(names, gif_name, frame_length=frame_length, data_band=data_band, classes=classes,
                 cmap_style=cmap_style, new_cs=new_cs, alpha=alpha, save=True, figdim=figdim)

    print('\r\nOPERATION COMPLETE')


def plot_all_pvl(predictions, labels, patch_ids, exp_id, new_cs, classes, cmap):
    def chunks(x, n):
        """Yield successive n-sized chunks from x."""
        for i in range(0, len(x), n):
            yield x[i:i + n]

    flat_z = np.array(list(chunks(predictions, int(len(predictions) / len(patch_ids)))))
    flat_y = np.array(list(chunks(labels, int(len(labels) / len(patch_ids)))))

    z_shape = flat_z.shape
    y_shape = flat_y.shape

    z = np.array([z_i.reshape((int(np.sqrt(z_shape[1])), int(np.sqrt(z_shape[1])))) for z_i in flat_z])
    y = np.array([y_i.reshape((int(np.sqrt(y_shape[1])), int(np.sqrt(y_shape[1])))) for y_i in flat_y])

    figdim = (9.3, 10.5)

    for j in range(len(patch_ids)):
        prediction_plot(z[j], y[j], patch_ids[j], exp_id, new_cs, classes=classes, cmap_style=cmap, show=False,
                        figdim=figdim)


def prediction_plot(z, y, patch_id, exp_id, new_cs, classes=None, block_size=32, cmap_style=None,
                    show=True, save=True, figdim=None):
    names = config['rgb_params']
    names['patch_ID'] = patch_id
    names['date'] = utils.datetime_reformat(utils.find_best_of(patch_id, manifest)[-1], '%Y_%m_%d', '%d.%m.%Y')

    # Get required formatted paths and names
    rgb, scene_path, data_name, _ = path_format(names)

    # Stacks together the R, G, & B bands to form an array of the RGB image
    rgb_image = stack_rgb(scene_path, rgb)

    # Defines the 'extent' of the image based on the size of the mask.
    extent = 0, y.shape[0], 0, y.shape[1]

    # Initialises a figure
    fig = plt.figure(figsize=figdim)

    gs = GridSpec(nrows=2, ncols=2, figure=fig)

    axes = np.array([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])])

    # Creates a cmap from query
    cmap = plt.get_cmap(cmap_style, len(classes))

    # Plots heatmap onto figure
    z_heatmap = axes[0].imshow(z, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5)
    y_heatmap = axes[1].imshow(y, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5)

    # Create RGB image
    axes[2].imshow(rgb_image, extent=extent)

    # Sets tick intervals to standard 32x32 block size
    axes[0].set_xticks(np.arange(0, z.shape[0] + 1, block_size))
    axes[0].set_yticks(np.arange(0, z.shape[1] + 1, block_size))

    axes[1].set_xticks(np.arange(0, y.shape[0] + 1, block_size))
    axes[1].set_yticks(np.arange(0, y.shape[1] + 1, block_size))

    axes[2].set_xticks(np.arange(0, rgb_image.shape[0] + 1, block_size))
    axes[2].set_yticks(np.arange(0, rgb_image.shape[1] + 1, block_size))

    # Add grid overlay
    axes[0].grid(which='both', color='#CCCCCC', linestyle=':')
    axes[1].grid(which='both', color='#CCCCCC', linestyle=':')
    axes[2].grid(which='both', color='#CCCCCC', linestyle=':')

    # Gets the co-ordinates of the corners of the image in decimal lat-lon
    corners = utils.transform_coordinates(scene_path + data_name, new_cs)

    # Creates a discrete mapping of the block size ticks to latitude longitude extent of the image
    lat_extent = np.linspace(start=corners[1][1][0], stop=corners[0][1][0],
                             num=int(y.shape[0] / block_size) + 1, endpoint=True)
    lon_extent = np.linspace(start=corners[0][0][1], stop=corners[0][1][1],
                             num=int(y.shape[0] / block_size) + 1, endpoint=True)

    # Converts the decimal lat-lon into degrees, minutes, seconds to label the axis
    lat_labels = utils.dec2deg(lat_extent, axis='lat')
    lon_labels = utils.dec2deg(lon_extent, axis='lon')

    # Sets the secondary axis tick labels
    axes[2].set_xticklabels(lon_labels, fontsize=9, rotation=30)
    axes[2].set_yticklabels(lat_labels, fontsize=9)

    # Plots colour bar onto figure
    clb = fig.colorbar(z_heatmap, ax=axes.ravel().tolist(), location='top',
                       ticks=np.arange(0, len(classes)), aspect=75, drawedges=True)

    # Sets colour bar ticks to class labels
    clb.ax.set_xticklabels(classes.values(), fontsize=9)

    # Set figure title and subplot titles
    fig.suptitle('{}'.format(patch_id), fontsize=15)
    axes[0].set_title('Predicted', fontsize=13)
    axes[1].set_title('Ground Truth', fontsize=13)
    axes[2].set_title('Reference Imagery From {}'.format(names['date']), fontsize=13)

    # Set axis labels
    axes[0].set_xlabel('(x) - Pixel Position', fontsize=10)
    axes[0].set_ylabel('(y) - Pixel Position', fontsize=10)
    axes[1].set_xlabel('(x) - Pixel Position', fontsize=10)
    axes[1].set_ylabel('(y) - Pixel Position', fontsize=10)
    axes[2].set_xlabel('Longitude', fontsize=10)
    axes[2].set_ylabel('Latitude', fontsize=10)

    # Display figure
    if show:
        plt.show()

    # Path and file name of figure
    fn = '{}/{}_{}_PvL_{}.png'.format(os.path.join(*config['dir']['results']), exp_id, patch_id, utils.timestamp_now())

    # If true, save file to fn
    if save:
        # Checks if file already exists. Deletes if true
        utils.exist_delete_check(fn)

        # Save figure to fn
        fig.savefig(fn)

    # Close figure
    plt.close()

    return fn


def plot_subpopulations(class_dist, class_names=None, cmap_dict=None, filename=None, save=True, show=False):
    """Creates a pie chart of the distribution of the classes within the data

    Args:
        class_dist (Counter): Modal distribution of classes in the dataset provided
        class_names (dict): Dictionary mapping class labels to class names
        cmap_dict (dict): Dictionary mapping class labels to class colours
        filename (str): Name of file to save plot to
        show (bool): Whether to show plot
        save (bool): Whether to save plot to file

    Returns:
        None
    """
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
        colours.append(cmap_dict[label[0]])

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


def make_confusion_matrix(test_pred, test_labels, classes, filename=None, show=True, save=False):
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
    # Finds the distribution of the classes within the data
    labels_dist = utils.find_subpopulations(test_labels)
    pred_dist = utils.find_subpopulations(test_pred)

    print(classes)
    print('labels:', labels_dist)
    print('pred:', pred_dist)

    empty = []

    # Checks which classes are not present in labels and predictions and adds to empty.
    for label in classes.keys():
        if label not in [mode[0] for mode in labels_dist] and label not in [mode[0] for mode in pred_dist]:
            empty.append(label)

    # Eliminates and reorganises classes based on those not present during testing.
    classes, transform, _ = utils.eliminate_classes(empty, old_classes=classes)

    test_labels = utils.mask_transform(test_labels, transform)
    test_pred = utils.mask_transform(test_pred, transform)

    # Creates the confusion matrix based on these predictions and the corresponding ground truth labels.
    cm = tf.math.confusion_matrix(labels=test_labels, predictions=test_pred).numpy()

    # Normalises confusion matrix.
    cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    np.nan_to_num(cm_norm, copy=False)

    # Extract class names from dict in numeric order to ensure labels match matrix.
    class_names = [classes[key] for key in range(len(classes.keys()))]

    # Converts confusion matrix to Pandas.DataFrame.
    cm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)

    # Plots figure.
    plt.figure()
    sns.heatmap(cm_df, annot=True, square=True, cmap=plt.cm.get_cmap('Blues'), vmin=0.0, vmax=1.0)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')

    # Shows and/or saves plot.
    if show:
        plt.show()
    if save:
        plt.savefig(filename)
        plt.close()


def format_plot_names(model_name, timestamp, path):
    def standard_format(plot_type, file_ext):
        filename = '{}_{}_{}.{}'.format(model_name, plot_type, timestamp, file_ext)
        return os.path.join(os.path.join(*path), filename)

    filenames = {'History': standard_format('MH', 'png'),
                 'Pred': standard_format('TP', 'png'),
                 'CM': standard_format('CM', 'png')}

    return filenames


def plot_results(metrics, plots, z, y, class_names, colours, save=True, show=False, model_name='',
                 timestamp=None, results_dir: list = ('')):
    if timestamp is None:
        timestamp = utils.timestamp_now(fmt='%d-%m-%Y_%H%M')

    filenames = format_plot_names(model_name, timestamp, results_dir)

    if plots['History']:
        plot_history(metrics, filename=filenames['History'], save=save, show=show)
    if plots['Pred']:
        plot_subpopulations(utils.find_subpopulations(z), class_names=class_names,
                            cmap_dict=colours, filename=filenames['Pred'], save=save, show=show)
    if plots['CM']:
        make_confusion_matrix(test_labels=y, test_pred=z, classes=class_names, filename=filenames['CM'],
                              save=save, show=show)
