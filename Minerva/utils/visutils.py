"""Module to visualise .tiff images and associated label masks.

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
        Must contain paths to auxiliary configs.
    imagery_config (dict): Config defining the properties of the imagery used in the experiment.
    data_config (dict): Config defining the properties of the data used in the experiment.
    data_dir (list): Path to directory holding dataset.
    patch_dir_prefix (str): Prefix to every patch ID in every patch directory name.
    n_pixels (int): Total number of pixels in each sample (per band).

TODO:
    * Add ability to plot labelled RGB images using the annual land cover labels
    * Add option to append annual land cover mask to patch GIFs
    * Reduce boilerplate

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Union, Optional, Tuple, Dict
from typing_extensions import Literal
from Minerva.utils import utils
import os
import yaml
import imageio
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import cv2
import osr
from alive_progress import alive_bar

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'

with open(config_path) as file:
    config = yaml.safe_load(file)

with open(config['dir']['configs']['imagery_config']) as file:
    imagery_config = yaml.safe_load(file)

with open(config['dir']['configs']['data_config']) as file:
    data_config = yaml.safe_load(file)

# Path to directory holding dataset.
data_dir = config['dir']['data']

# Prefix to every patch ID in every patch directory name.
patch_dir_prefix = imagery_config['patch_dir_prefix']

n_pixels = imagery_config['data_specs']['image_size'][0] * imagery_config['data_specs']['image_size'][1]

# Automatically fixes the layout of the figures to accommodate the colour bar legends.
plt.rcParams['figure.constrained_layout.use'] = True

# Increases DPI to avoid strange plotting errors for class heatmaps.
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Removes margin in x-axis of plots.
plt.rcParams['axes.xmargin'] = 0

# Downloads required plugin for imageio if not already present.
imageio.plugins.freeimage.download()

# Filters out all TensorFlow messages other than errors.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_max_samples = 25


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def path_format(names: dict) -> Tuple[Dict[str, str], str, str, str]:
    """Takes a dictionary of unique IDs to format the paths and names of files associated with the desired scene.

    Args:
        names (dict): Dictionary of IDs to uniquely identify the scene and selected bands.

    Returns:
        scene_path (str): Path to directory holding images from desired scene.
        rgb (dict): Dictionary of filenames of R, G & B band images.
        scene_data_name (str): Name of the file containing the scene classification label mask.
        patch_data_name (str): Path to the file containing the annual classification label mask.
    """
    # Format the two required date formats used by REF MLHub.
    date1 = utils.datetime_reformat(names['date'], '%d.%m.%Y', '%Y_%m_%d')
    date2 = utils.datetime_reformat(names['date'], '%d.%m.%Y', '%Y%m%d')

    # Format path to the directory holding all the scene files.
    scene_path = "{}{}".format(os.sep.join((*data_dir, patch_dir_prefix + names['patch_ID'], date1)), os.sep)

    # Format the name of the file containing the label mask data.
    scene_data_name = '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['band_ID'])
    patch_data_name = utils.get_label_path(names['patch_ID'])

    # Create a dictionary of the names of the requested red, green, blue images.
    rgb = {'R': '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['R_band']),
           'G': '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['G_band']),
           'B': '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['B_band'])}

    return rgb, scene_path, scene_data_name, patch_data_name


def de_interlace(x: Union[list, np.ndarray], f: int) -> np.ndarray:
    """Separates interlaced arrays, `x' at a frequency of `f' from each other.

    Args:
        x (list or np.ndarray): Array of data to be de-interlaced.
        f (int): Frequency at which interlacing occurs. Equivalent to number of sources interlaced together.

    Returns:
        De-interlaced array. Each source array is now sequentially connected.
    """
    new_x = []
    for i in range(f):
        x_i = []
        for j in np.arange(start=i, stop=len(x), step=f):
            x_i.append(x[j])
        new_x.append(np.array(x_i).flatten())

    return np.array(new_x).flatten()


def get_extent(shape: Tuple[int, int], data_fn: str, new_cs: osr.SpatialReference,
               spacing: int = 32) -> Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """Gets the extent of the image with 'shape' and at data_fn in latitude, longitude of system new_cs.

    Args:
        shape (tuple[int, int]): 2D shape of image to be used to define the extents of the composite image.
        data_fn (str): Path and filename of the TIF file whose geospatial meta data will be used
            to get the corners of the image in latitude and longitude.
        new_cs(osr.SpatialReference): Co-ordinate system to convert co-ordinates found in data_fn TIF file to.
        spacing (int): Spacing of the lat - lon ticks.

    Returns:
        extent (tuple[int, int, int, int]): The corners of the image in pixel co-ordinates e.g. (0, 256, 0, 256).
        lat_extent (np.ndarray): The latitude extent of the image with ticks at intervals defined by 'spacing'.
        lon_extent (np.ndarray): The longitude extent of the image with ticks at intervals defined by 'spacing'.
    """
    # Defines the 'extent' for a composite image based on the size of shape.
    extent = 0, shape[0], 0, shape[1]

    # Gets the co-ordinates of the corners of the image in decimal lat-lon.
    corners = utils.transform_coordinates(data_fn, new_cs)

    # Creates a discrete mapping of the spaced ticks to latitude longitude extent of the image.
    lat_extent = np.linspace(start=corners[1][1][0], stop=corners[0][1][0],
                             num=int(shape[0] / spacing) + 1, endpoint=True)
    lon_extent = np.linspace(start=corners[0][0][1], stop=corners[0][1][1],
                             num=int(shape[0] / spacing) + 1, endpoint=True)

    return extent, lat_extent, lon_extent


def discrete_heatmap(data, classes: Optional[Union[list, tuple, np.ndarray]] = None,
                     cmap_style: Optional[Union[str, ListedColormap]] = None, block_size: int = 32) -> None:
    """Plots a heatmap with a discrete colour bar. Designed for Radiant Earth MLHub 256x256 SENTINEL images.

    Args:
        data (list or np.ndarray): 2D Array of data to be plotted as a heat map.
        classes (list[str]): Optional; List of all possible class labels.
        cmap_style (str, ListedColormap): Optional; Name or object for colour map style.
        block_size (int): Optional; Size of block image sub-division in pixels.

    Returns:
        None
    """
    # Initialises a figure.
    plt.figure()

    # Creates a cmap from query.
    cmap = plt.get_cmap(cmap_style, len(classes))

    # Plots heatmap onto figure.
    heatmap = plt.imshow(data, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5)

    # Sets tick intervals to block size. Default 32 x 32.
    plt.xticks(np.arange(0, data.shape[0] + 1, block_size))
    plt.yticks(np.arange(0, data.shape[1] + 1, block_size))

    # Add grid overlay.
    plt.grid(which='both', color='#CCCCCC', linestyle=':')

    # Plots colour bar onto figure.
    clb = plt.colorbar(heatmap, ticks=np.arange(0, len(classes)), shrink=0.77)

    # Sets colour bar ticks to class labels.
    clb.ax.set_yticklabels(classes)

    # Display figure.
    plt.show()

    # Close figure.
    plt.close()


def stack_rgb(scene_path: str, rgb: dict) -> np.ndarray:
    """Stacks together red, green and blue image arrays from file to create a RGB array.

    Args:
        scene_path (str): Path to directory holding images from desired scene.
        rgb (dict): Dictionary of filenames of R, G & B band images.

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


def make_rgb_image(scene_path: str, rgb: dict, block_size: int = 32):
    """Creates an RGB image from a composition of red, green and blue band .tif images

    Args:
        scene_path (str): Path to directory holding images from desired scene
        rgb (dict): Dictionary of filenames of R, G & B band images.
        block_size (int): Optional; Size of block image sub-division in pixels.

    Returns:
        rgb_image (AxesImage): Plotted RGB image object
    """
    # Stack RGB image data together.
    rgb_image_array = stack_rgb(scene_path, rgb)

    # Create RGB image.
    rgb_image = plt.imshow(rgb_image_array)

    # Sets tick intervals to block size. Default 32 x 32.
    plt.xticks(np.arange(0, rgb_image_array.shape[0] + 1, block_size))
    plt.yticks(np.arange(0, rgb_image_array.shape[1] + 1, block_size))

    # Add grid overlay.
    plt.grid(which='both', color='#CCCCCC', linestyle=':')

    plt.show()

    return rgb_image


def labelled_rgb_image(names: dict, mode: str = 'patch', data_band: int = 1,
                       classes: Optional[Union[list, tuple, np.ndarray]] = None, block_size: int = 32,
                       cmap_style: Optional[Union[str, ListedColormap]] = None, alpha: float = 0.5,
                       new_cs: Optional[osr.SpatialReference] = None,
                       show: bool = True, save: bool = True, figdim: tuple = (8.02, 10.32)) -> str:
    """Produces a layered image of an RGB image and it's associated label mask heat map alpha blended on top.

    Args:
        names (dict): Dictionary of IDs to uniquely identify the scene and selected bands.
        mode (str): Optional; Whether to plot the `patch' level labels or the scene. Default `scene'.
        data_band (int): Optional; Band number of data .tif file.
        classes (list[str]): Optional; List of all possible class labels.
        block_size (int): Optional; Size of block image sub-division in pixels.
        cmap_style (str or ListedColormap): Optional; Name or object for colour map style.
        alpha (float): Optional; Fraction determining alpha blending of label mask.
        new_cs(osr.SpatialReference): Optional; Co-ordinate system to convert image to and use for labelling.
        show (bool): Optional; True for show figure when plotted. False if not.
        save (bool): Optional; True to save figure to file. False if not.
        figdim (tuple): Optional; Figure (height, width) in inches.

    Returns:
        fn (str): Path to figure save location
    """
    # Get required formatted paths and names.
    rgb, scene_path, scene_data_name, patch_data_name = path_format(names)

    data_name = scene_path + scene_data_name
    if mode == 'patch':
        data_name = patch_data_name

    # Stacks together the R, G, & B bands to form an array of the RGB image.
    rgb_image = stack_rgb(scene_path, rgb)

    # Loads data to plotted as heatmap from file.
    data = utils.load_array(data_name, band=data_band)

    extent, lat_extent, lon_extent = get_extent(data.shape, data_name, new_cs, spacing=block_size)

    # Initialises a figure.
    fig, ax1 = plt.subplots()

    # Create RGB image.
    ax1.imshow(rgb_image, extent=extent)

    # Creates a cmap from query.
    cmap = plt.get_cmap(cmap_style, len(classes))

    # Plots heatmap onto figure.
    heatmap = ax1.imshow(data, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5, extent=extent, alpha=alpha)

    # Sets tick intervals to standard 32x32 block size.
    ax1.set_xticks(np.arange(0, data.shape[0] + 1, block_size))
    ax1.set_yticks(np.arange(0, data.shape[1] + 1, block_size))

    # Creates a secondary x and y axis to hold lat-lon.
    ax2 = ax1.twiny().twinx()

    # Plots an invisible line across the diagonal of the image to create the secondary axis for lat-lon.
    ax2.plot(lon_extent, lat_extent, ' ',
             clip_box=Bbox.from_extents(lon_extent[0], lat_extent[0], lon_extent[-1], lat_extent[-1]))

    # Sets ticks for lat-lon.
    ax2.set_xticks(lon_extent)
    ax2.set_yticks(lat_extent)

    # Sets the limits of the secondary axis so they should align with the primary.
    ax2.set_xlim(left=lon_extent[0], right=lon_extent[-1])
    ax2.set_ylim(top=lat_extent[-1], bottom=lat_extent[0])

    # Converts the decimal lat-lon into degrees, minutes, seconds to label the axis.
    lat_labels = utils.dec2deg(lat_extent, axis='lat')
    lon_labels = utils.dec2deg(lon_extent, axis='lon')

    # Sets the secondary axis tick labels.
    ax2.set_xticklabels(lon_labels, fontsize=11)
    ax2.set_yticklabels(lat_labels, fontsize=10, rotation=-30, ha='left')

    # Add grid overlay.
    ax1.grid(which='both', color='#CCCCCC', linestyle=':')

    # Plots colour bar onto figure.
    clb = plt.colorbar(heatmap, ticks=np.arange(0, len(classes)), shrink=0.9, aspect=75, drawedges=True)

    # Sets colour bar ticks to class labels.
    clb.ax.set_yticklabels(classes, fontsize=11)

    # Bodge to get a figure title by using the colour bar title.
    clb.ax.set_title('%s\n%s\nLand Cover' % (names['patch_ID'], names['date']), loc='left', fontsize=15)

    # Set axis labels.
    ax1.set_xlabel('(x) - Pixel Position', fontsize=14)
    ax1.set_ylabel('(y) - Pixel Position', fontsize=14)
    ax2.set_ylabel('Latitude', fontsize=14, rotation=270, labelpad=12)
    ax2.set_title('Longitude')  # Bodge

    # Manual trial and error fig size which fixes aspect ratio issue.
    fig.set_figheight(figdim[0])
    fig.set_figwidth(figdim[1])

    # Display figure.
    if show:
        plt.show()

    # Path and file name of figure.
    fn = '%s/%s_%s_RGBHM.png' % (scene_path, names['patch_ID'],
                                 utils.datetime_reformat(names['date'], '%d.%m.%Y', '%Y%m%d'))

    # If true, save file to fn.
    if save:
        # Checks if file already exists. Deletes if true.
        utils.exist_delete_check(fn)

        # Save figure to fn.
        fig.savefig(fn)

    # Close figure.
    plt.close()

    return fn


def make_gif(names: dict, gif_name: str, frame_length: float = 1.0, data_band: int = 1,
             classes: Optional[Union[list, tuple, np.ndarray]] = None,
             cmap_style: Optional[Union[str, ListedColormap]] = None,
             new_cs: Optional[osr.SpatialReference] = None, alpha: float = 0.5, save: bool = False,
             figdim: tuple = (8.02, 10.32)) -> None:
    """Wrapper to labelled_rgb_image() to make a GIF for a patch out of scenes.

    Args:
        names (dict): Dictionary of IDs to uniquely identify the patch dir and selected bands.
        gif_name (str): Path to and name of GIF to be made.
        frame_length (float): Optional; Length of each GIF frame in seconds.
        data_band (int): Optional; Band number of data .tif file.
        classes (list[str]): Optional; List of all possible class labels.
        cmap_style (str or ListedColormap): Optional; Name or object for colour map style.
        new_cs(osr.SpatialReference): Optional; Co-ordinate system to convert image to and use for labelling.
        alpha (float): Optional; Fraction determining alpha blending of label mask.
        save (bool): Optional; True to save figure to file. False if not.
        figdim (tuple): Optional; Figure (height, width) in inches.

    Returns:
        None
    """
    # Fetch all the scene dates for this patch in DD.MM.YYYY format.
    dates = utils.date_grab(names['patch_ID'])

    # Initialise progress bar.
    with alive_bar(len(dates), bar='blocks') as bar:

        # List to hold filenames and paths of images created.
        frames = []
        for date in dates:
            # Update progress bar with current scene.
            bar.text('SCENE ON %s' % date)

            # Update names date field.
            names['date'] = date

            # Create a frame of the GIF for a scene of the patch.
            frame = labelled_rgb_image(names, data_band=data_band, classes=classes, cmap_style=cmap_style,
                                       new_cs=new_cs, alpha=alpha, save=save, show=False, figdim=figdim)

            # Read in frame just created and add to list of frames.
            frames.append(imageio.imread(frame))

            # Update bar with step completion.
            bar()

    # Create a 'unknown' bar to 'spin' while the GIF is created.
    with alive_bar(unknown='waves') as bar:
        # Add current operation to spinner bar.
        bar.text('MAKING PATCH %s GIF' % names['patch_ID'])

        # Create GIF.
        imageio.mimsave(gif_name, frames, 'GIF-FI', duration=frame_length, quantizer='nq')


def make_all_the_gifs(names: dict, frame_length: float = 1.0, data_band: int = 1,
                      classes: Optional[Union[list, tuple, np.ndarray]] = None,
                      cmap_style: Optional[Union[str, ListedColormap]] = None,
                      new_cs: Optional[osr.SpatialReference] = None,
                      alpha: float = 0.5, figdim: tuple = (8.02, 10.32)) -> None:
    """Wrapper to make_gifs() to iterate through all patches in dataset.

    Args:
        names (dict): Dictionary holding the band IDs. Patch ID and date added per iteration.
        frame_length (float): Optional; Length of each GIF frame in seconds.
        data_band (int): Optional; Band number of data .tif file.
        classes (list[str]): Optional; List of all possible class labels.
        cmap_style (str, ListedColormap): Optional; Name or object for colour map style.
        new_cs(SpatialReference): Optional; Co-ordinate system to convert image to and use for labelling.
        alpha (float): Optional; Fraction determining alpha blending of label mask.
        figdim (tuple): Optional; Figure (height, width) in inches.

    Returns:
        None
    """
    # Gets all the patch IDs from the dataset directory.
    patches = utils.patch_grab()

    # Iterator for progress counter.
    i = 0

    # Iterate through all patches.
    for patch in patches:
        # Count this iteration for the progress counter.
        i += 1

        # Print status update.
        print('\r\nNOW SERVING PATCH %s (%s/%s): ' % (patch, i, len(patches)))

        # Update dictionary for this patch.
        names['patch_ID'] = patch

        # Define name of GIF for this patch.
        gif_name = '%s/%s%s/%s.gif' % (data_dir, patch_dir_prefix, names['patch_ID'], names['patch_ID'])

        # Call make_gif() for this patch.
        make_gif(names, gif_name, frame_length=frame_length, data_band=data_band, classes=classes,
                 cmap_style=cmap_style, new_cs=new_cs, alpha=alpha, save=True, figdim=figdim)

    print('\r\nOPERATION COMPLETE')


def plot_all_pvl(z: Union[list, np.ndarray], y: Union[list, np.ndarray], patch_ids: Union[list, tuple, np.ndarray],
                 classes: dict, colours: dict, fn_prefix: str, frac: float = 0.05,
                 fig_dim: Tuple[float, float] = (9.3, 10.5)) -> None:
    """Uses prediction_plot to plot all predicted versus ground truth comparison plots from MLP testing.

    Args:
        z (list[list[int]] or np.ndarray[np.ndarray[int]]): List of predicted label masks.
        y (list[list[int]] or np.ndarray[np.ndarray[int]]): List of corresponding ground truth label masks.
        patch_ids (list[str]): List of IDs identifying the patches from which predictions and labels came from.
        classes (dict[str]): Dictionary mapping class labels to class names.
        colours (dict[str]): Dictionary mapping class labels to colours.
        fn_prefix (str): Common filename prefix (including path to file) for all plots of this type
            from this experiment to use.
        frac (float): Optional; Fraction of patch samples to plot.
        fig_dim (Tuple[float, float]): Optional; Figure (height, width) in inches.

    Returns:
        None
    """
    def chunks(x, n: int):
        """Yield successive n-sized chunks from x.
        Args:
            x (list or np.ndarray): Array to be split into chunks.
            n (int): Length of yielded array.
        Yields:
            n-sized chunks from x.
        """
        for i in range(0, len(x), n):
            yield x[i:i + n]

    # Gets the number of workers to use as the frequency for the de-interlacing operation.
    num_workers = config['hyperparams']['params']['num_workers']

    cmap = ListedColormap(colours.values(), N=len(colours.values()))

    # `De-interlaces' the outputs to account for the effects of multi-threaded workloads.
    z = de_interlace(z, num_workers)
    y = de_interlace(y, num_workers)
    patch_ids = de_interlace(patch_ids, num_workers)

    # Extracts just a patch ID for each test patch supplied.
    patch_ids = [patch_ids[i] for i in np.arange(start=0, stop=len(patch_ids), step=n_pixels)]

    # Create a new projection system in lat-lon
    new_cs = osr.SpatialReference()
    new_cs.ImportFromEPSG(data_config['co_sys']['id'])

    flat_z = np.array(list(chunks(z, int(len(z) / len(patch_ids)))))
    flat_y = np.array(list(chunks(y, int(len(y) / len(patch_ids)))))

    z_shape = flat_z.shape
    y_shape = flat_y.shape

    z = np.array([z_i.reshape((int(np.sqrt(z_shape[1])), int(np.sqrt(z_shape[1])))) for z_i in flat_z])
    y = np.array([y_i.reshape((int(np.sqrt(y_shape[1])), int(np.sqrt(y_shape[1])))) for y_i in flat_y])

    print('PRODUCING PREDICTED VERSUS GROUND TRUTH PLOTS')
    # Limits number of masks to produce to a fractional number of total and no more than _max_samples.
    n_samples = int(frac * len(patch_ids))
    if n_samples > _max_samples:
        n_samples = _max_samples

    # Initialises a progress bar for the epoch.
    with alive_bar(n_samples, bar='blocks') as bar:

        # Plots the predicted versus ground truth labels for all test patches supplied.
        for i in random.sample(range(len(patch_ids)), n_samples):
            prediction_plot(z[i], y[i], patch_ids[i], 'patch', new_cs=new_cs, classes=classes, cmap_style=cmap,
                            fn_prefix=fn_prefix, show=False, fig_dim=fig_dim)

            bar()


def prediction_plot(z: np.ndarray, y: np.ndarray, sample_id: str, sample_type: Literal['scene', 'patch'],
                    new_cs: osr.SpatialReference, exp_id: Optional[str] = None, classes: Optional[dict] = None,
                    block_size: int = 32, cmap_style: Optional[Union[str, ListedColormap]] = None, show: bool = True,
                    save: bool = True, fig_dim: Optional[Tuple[float, float]] = None,
                    fn_prefix: Optional[str] = None) -> None:
    """Produces a figure containing subplots of the predicted label mask, the ground truth label mask
        and a reference RGB image of the same patch.

    Args:
        z (np.ndarray[np.ndarray[int]]): 2D array of the predicted label mask.
        y (np.ndarray[np.ndarray[int]]): 2D array of the corresponding ground truth label mask.
        sample_id (str): Unique ID of the patch.
        sample_type (str): Denotes what sort of sample is to be plotted. Must be either 'scene' or 'patch'.
        new_cs(osr.SpatialReference): Optional; Co-ordinate system to convert image to and use for labelling.
        exp_id (str): Optional; Unique ID for the experiment run that predictions and labels come from.
        classes (dict[str]): Optional; Dictionary mapping class labels to class names.
        block_size (int): Optional; Size of block image sub-division in pixels.
        cmap_style (str, ListedColormap): Optional; Name or object for colour map style.
        show (bool): Optional; True for show figure when plotted. False if not.
        save (bool): Optional; True to save figure to file. False if not.
        fig_dim (Tuple[float, float]): Optional; Figure (height, width) in inches.
        fn_prefix (str): Optional; Common filename prefix (including path to file) for all plots of this type
            from this experiment. Appended with the sample ID to give the filename to save the plot to.

    Returns:
        None
    """
    manifest = pd.read_csv(utils.get_manifest())

    names = config['rgb_params']
    patch_id = sample_id

    if sample_type == 'scene':
        patch_id, date = utils.extract_from_tag(sample_id)
        names['date'] = utils.datetime_reformat(date, '%Y_%m_%d', '%d.%m.%Y')
    if sample_type == 'patch':
        names['date'] = utils.datetime_reformat(utils.find_best_of(patch_id, manifest)[-1], '%Y_%m_%d', '%d.%m.%Y')

    names['patch_ID'] = patch_id

    # Get required formatted paths and names.
    rgb, scene_path, data_name, _ = path_format(names)

    # Stacks together the R, G, & B bands to form an array of the RGB image.
    rgb_image = stack_rgb(scene_path, rgb)

    extent, lat_extent, lon_extent = get_extent(y.shape, scene_path + data_name, new_cs, spacing=block_size)

    # Initialises a figure.
    fig = plt.figure(figsize=fig_dim)

    gs = GridSpec(nrows=2, ncols=2, figure=fig)

    axes = np.array([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])])

    # Creates a cmap from query.
    cmap = plt.get_cmap(cmap_style, len(classes))

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
    axes[0].grid(which='both', color='#CCCCCC', linestyle=':')
    axes[1].grid(which='both', color='#CCCCCC', linestyle=':')
    axes[2].grid(which='both', color='#CCCCCC', linestyle=':')

    # Converts the decimal lat-lon into degrees, minutes, seconds to label the axis.
    lat_labels = utils.dec2deg(lat_extent, axis='lat')
    lon_labels = utils.dec2deg(lon_extent, axis='lon')

    # Sets the secondary axis tick labels.
    axes[2].set_xticklabels(lon_labels, fontsize=9, rotation=30)
    axes[2].set_yticklabels(lat_labels, fontsize=9)

    # Plots colour bar onto figure.
    clb = fig.colorbar(z_heatmap, ax=axes.ravel().tolist(), location='top',
                       ticks=np.arange(0, len(classes)), aspect=75, drawedges=True)

    # Sets colour bar ticks to class labels.
    clb.ax.set_xticklabels(classes.values(), fontsize=9)

    # Set figure title and subplot titles.
    fig.suptitle('{}'.format(patch_id), fontsize=15)
    axes[0].set_title('Predicted', fontsize=13)
    axes[1].set_title('Ground Truth', fontsize=13)
    axes[2].set_title('Reference Imagery From {}'.format(names['date']), fontsize=13)

    # Set axis labels.
    axes[0].set_xlabel('(x) - Pixel Position', fontsize=10)
    axes[0].set_ylabel('(y) - Pixel Position', fontsize=10)
    axes[1].set_xlabel('(x) - Pixel Position', fontsize=10)
    axes[1].set_ylabel('(y) - Pixel Position', fontsize=10)
    axes[2].set_xlabel('Longitude', fontsize=10)
    axes[2].set_ylabel('Latitude', fontsize=10)

    # Display figure.
    if show:
        plt.show()

    if fn_prefix is None:
        path = os.path.join(*config['dir']['results'])
        fn_prefix = os.sep.join([path, '{}_{}_Mask'.format(exp_id, utils.timestamp_now())])

    # Path and file name of figure.
    fn = '{}_{}.png'.format(fn_prefix, sample_id)

    # If true, save file to fn.
    if save:
        # Checks if file already exists. Deletes if true.
        utils.exist_delete_check(fn)

        # Save figure to fn.
        fig.savefig(fn)

    # Close figure.
    plt.close()


def seg_plot(z: list, y: list, ids: list, classes: dict, colours: dict, fn_prefix: str,
             frac: float = 0.05, fig_dim: Tuple[float, float] = (9.3, 10.5)) -> None:
    """Custom function for pre-processing the outputs from image segmentation testing for data visualisation.

    Args:
        z (list[float]): Predicted segmentation masks by the network.
        y (list[float]): Corresponding ground truth masks.
        ids (list[str]): Corresponding patch IDs for the test data supplied to the network.
        classes (dict): Dictionary mapping class labels to class names.
        colours (dict): Dictionary mapping class labels to colours.
        fn_prefix (str): Common filename prefix (including path to file) for all plots of this type
            from this experiment to use.
        frac (float): Optional; Fraction of patch samples to plot.
        fig_dim (tuple[float, float]): Optional; Figure (height, width) in inches.

    Returns:
        None
    """
    z = np.array(z)
    y = np.array(y)

    z = np.reshape(z, (z.shape[0] * z.shape[1], z.shape[2], z.shape[3]))
    y = np.reshape(y, (y.shape[0] * y.shape[1], y.shape[2], y.shape[3]))
    ids = np.array(ids).flatten()

    # Create a new projection system in lat-lon.
    new_cs = osr.SpatialReference()
    new_cs.ImportFromEPSG(data_config['co_sys']['id'])

    print('PRODUCING PREDICTED MASKS')

    # Limits number of masks to produce to a fractional number of total and no more than _max_samples.
    n_samples = int(frac * len(ids))
    if n_samples > _max_samples:
        n_samples = _max_samples

    # Initialises a progress bar for the epoch.
    with alive_bar(n_samples, bar='blocks') as bar:

        # Plots the predicted versus ground truth labels for all test patches supplied.
        for i in random.sample(range(len(ids)), n_samples):
            prediction_plot(z[i], y[i], ids[i], 'scene', exp_id=config['model_name'], new_cs=new_cs,
                            classes=classes, fig_dim=fig_dim, show=False, fn_prefix=fn_prefix,
                            cmap_style=ListedColormap(colours.values(), N=len(colours)))

            bar()


def plot_subpopulations(class_dist: Union[list, tuple, np.ndarray], class_names: Optional[dict] = None,
                        cmap_dict: Optional[dict] = None, filename: Optional[str] = None, save: bool = True,
                        show: bool = False) -> None:
    """Creates a pie chart of the distribution of the classes within the data.

    Args:
        class_dist (list[list]): Modal distribution of classes in the dataset provided.
        class_names (dict): Optional; Dictionary mapping class labels to class names.
        cmap_dict (dict): Optional; Dictionary mapping class labels to class colours.
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
            class_data.append('{} \n{:.2f}%'.format(class_names[label[0]], (label[1] * 100.0 / n_samples)))
        else:
            class_data.append('{} \n<0.01%'.format(class_names[label[0]]))
        counts.append(label[1])
        colours.append(cmap_dict[label[0]])

    # Locks figure size.
    plt.figure(figsize=(6, 5))

    # Plot a pie chart of the data distribution amongst the classes.
    patches, text = plt.pie(counts, colors=colours, explode=[i * 0.05 for i in range(len(class_data))])

    # Adds legend.
    plt.legend(patches, class_data, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    # Shows and/or saves plot.
    if show:
        plt.show()
    if save:
        plt.savefig(filename)
        plt.close()


def plot_history(metrics: dict, filename: Optional[str] = None, save: bool = True, show: bool = False) -> None:
    """Plots model history based on metrics supplied.

    Args:
        metrics (dict): Dictionary containing the names and results of the metrics by which model was assessed.
        filename (str): Optional; Name of file to save plot to.
        show (bool): Optional; Whether to show plot.
        save (bool): Optional; Whether to save plot to file.

    Returns:
        None
    """
    # Initialise figure.
    plt.figure()

    # Plots each metric in metrics, appending their artist handles.
    handles = []
    for metric in metrics.values():
        handles.append(plt.plot(metric['x'], metric['y'])[0])

    # Creates legend from plot artist handles and names of metrics.
    plt.legend(handles=handles, labels=metrics.keys())

    # Forces x-axis ticks to be integers.
    plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adds a grid overlay with green dashed lines.
    plt.grid(color='green', linestyle='--', linewidth=0.5)  # For some funky gridlines

    # Adds axis labels.
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')

    # Shows and/or saves plot.
    if show:
        plt.show()
    if save:
        plt.savefig(filename)
        plt.close()


def make_confusion_matrix(test_pred: Union[list, np.ndarray], test_labels: Union[list, np.ndarray],
                          classes: dict, filename: Optional[str] = None, show: bool = True,
                          save: bool = False) -> None:
    """Creates a heat-map of the confusion matrix of the given model.

    Args:
        test_pred(list[int]): Predictions made by model on test images.
        test_labels (list[int]): Accompanying ground truth labels for testing images.
        classes (dict): Dictionary mapping class labels to class names.
        filename (str): Optional; Name of file to save plot to.
        show (bool): Optional; Whether to show plot.
        save (bool): Optional; Whether to save plot to file.

    Returns:
        None
    """
    test_pred, test_labels, classes = utils.check_test_empty(test_pred, test_labels, classes)

    test_pred = np.array(test_pred, dtype=np.uint8)
    test_labels = np.array(test_labels, dtype=np.uint8)

    # Creates the confusion matrix based on these predictions and the corresponding ground truth labels.
    cm = []
    try:
        cm = tf.math.confusion_matrix(labels=test_labels, predictions=test_pred, dtype=np.uint8).numpy()
    except RuntimeWarning as err:
        print('\n', err)
        print('At least one class had no ground truth or no predicted labels!')

    # Normalises confusion matrix.
    cm_norm = np.around(cm.astype(np.float16) / cm.sum(axis=1)[:, np.newaxis], decimals=2)
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


def make_roc_curves(probs: Union[list, np.ndarray], labels: Union[list, np.ndarray], class_names: dict, colours: dict,
                    micro: bool = True, macro: bool = True, filename: Optional[str] = None, show: bool = False,
                    save: bool = True) -> None:
    """Plots ROC curves for each class, the micro and macro average ROC curves and accompanying AUCs.

    Adapted from Scikit-learn's example at:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Args:
        probs (list or np.ndarray): Array of probabilistic predicted classes from model where each sample
            should have a list of the predicted probability for each class.
        labels (list or np.ndarray): List of corresponding ground truth labels.
        class_names (dict): Dictionary mapping class labels to class names.
        colours (dict): Dictionary mapping class labels to colours.
        micro (bool): Optional; Whether or not to compute and plot the micro average ROC curves.
        macro (bool): Optional; Whether or not to compute and plot the macro average ROC curves.
        filename (str): Optional; Name of file to save plot to.
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
    fpr, tpr, roc_auc = utils.compute_roc_curves(probs, labels, class_labels, micro=micro, macro=macro)

    # Plot all ROC curves
    print('\nPlotting ROC Curves')
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
        plt.plot(
            fpr[key],
            tpr[key],
            color=colours[key],
            label=f'{class_names[key]} ' + '(AUC = {:.2f})'.format(roc_auc[key]),
        )

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
        plt.show()
    if save:
        plt.savefig(filename)
        print('ROC Curves plot SAVED')
        plt.close()


def format_plot_names(model_name: str, timestamp: str, path: Union[list, tuple]) -> dict:
    """Creates unique filenames of plots in a standardised format.

    Args:
        model_name (str): Name of model. e.g. MLP-MkVI.
        timestamp (str): Time and date to be used to identify experiment.
        path (list[str]): Path to the directory for storing plots as a list of strings for each level.

    Returns:
        filenames (dict): Formatted filenames for plots.
    """
    def standard_format(plot_type: str, *sub_dir) -> str:
        """Creates a unique filename for a plot in a standardised format.

        Args:
            plot_type (str): Plot type to use in filename.
            sub_dir (str): Additional sub-directories to add to path to filename.

        Returns:
            String of path to filename of the form "{model_name}_{timestamp}_{plot_type}.{file_ext}"
        """
        filename = '{}_{}_{}'.format(model_name, timestamp, plot_type)
        return os.path.join(*path, *sub_dir, filename)

    filenames = {'History': standard_format('MH') + '.png',
                 'Pred': standard_format('TP') + '.png',
                 'CM': standard_format('CM') + '.png',
                 'ROC': standard_format('ROC' + '.png'),
                 'Mask': standard_format('Mask', 'Masks'),
                 'PvT': standard_format('PvT', 'PvTs')}

    return filenames


def plot_results(plots: dict, z: Union[list, np.ndarray], y: Union[list, np.ndarray], metrics: Optional[dict] = None,
                 ids: Optional[list] = None, probs: Optional[Union[list, np.ndarray]] = None,
                 class_names: Optional[dict] = None, colours: Optional[dict] = None, save: bool = True,
                 show: bool = False, model_name: Optional[str] = None, timestamp: Optional[str] = None,
                 results_dir: Optional[Union[list, tuple]] = None) -> None:
    """Orchestrates the creation of various plots from the results of a model fitting.

    Args:
        plots (dict): Dictionary defining which plots to make.
        z (list[list[int]] or np.ndarray[np.ndarray[int]]): List of predicted label masks.
        y (list[list[int]] or np.ndarray[np.ndarray[int]]): List of corresponding ground truth label masks.
        metrics (dict): Optional; Dictionary containing a log of various metrics used to assess
            the performance of a model.
        ids (list[str]): Optional; List of IDs defining the origin of samples to the model.
            May be either patch IDs or scene tags.
        probs (list or np.ndarray): Optional; Array of probabilistic predicted classes from model where each sample
            should have a list of the predicted probability for each class.
        class_names (dict): Optional; Dictionary mapping class labels to class names.
        colours (dict): Optional; Dictionary mapping class labels to colours.
        save (bool): Optional; Whether to save the plots to file.
        show (bool): Optional; Whether to show the plots.
        model_name (str): Optional; Name of model. e.g. MLP-MkVI.
        timestamp (str): Optional; Time and date to be used to identify experiment.
            If not specified, the current date-time is used.
        results_dir (list): Optional; Path to the directory for storing plots as a list of strings for each level.

    Notes:
        save = True, show = False regardless of input for plots made for each sample such as PvT or Mask plots.

    Returns:
        None
    """
    flat_z = utils.model_output_flatten(z)
    flat_y = utils.model_output_flatten(y)

    if timestamp is None:
        timestamp = utils.timestamp_now(fmt='%d-%m-%Y_%H%M')

    filenames = format_plot_names(model_name, timestamp, results_dir)

    try:
        os.mkdir(os.sep.join(results_dir))
    except FileExistsError as err:
        print(err)

    if plots['History']:
        print('\nPLOTTING MODEL HISTORY')
        plot_history(metrics, filename=filenames['History'], save=save, show=show)
    if plots['Pred']:
        print('\nPLOTTING CLASS DISTRIBUTION OF PREDICTIONS')
        plot_subpopulations(utils.find_subpopulations(flat_z), class_names=class_names,
                            cmap_dict=colours, filename=filenames['Pred'], save=save, show=show)
    if plots['CM']:
        print('\nPLOTTING CONFUSION MATRIX')
        make_confusion_matrix(test_labels=flat_y, test_pred=flat_z, classes=class_names, filename=filenames['CM'],
                              save=save, show=show)

    if plots['ROC']:
        print('\nPLOTTING ROC CURVES')
        make_roc_curves(probs, flat_y, class_names=class_names, colours=colours, filename=filenames['ROC'],
                        micro=plots['micro'], macro=plots['macro'], save=save, show=show)

    if plots['PvT']:
        os.mkdir(os.path.join(*results_dir, 'PvTs'))
        plot_all_pvl(z, y, ids, classes=class_names, colours=colours, fn_prefix=filenames['PvT'])

    if plots['Mask']:
        os.mkdir(os.path.join(*results_dir, 'Masks'))
        seg_plot(z, y, ids, fn_prefix=filenames['Mask'], classes=class_names, colours=colours)
