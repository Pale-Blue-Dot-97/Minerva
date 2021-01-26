"""Radiant_MLHub_DataVis

Script to locate, open, read and visualise .tiff images and associated label masks downloaded from the
Radiant MLHub API.

Author: Harry Baker


TODO:
    * Add ability to plot labelled RGB images using the annual land cover labels
    * Add option to append annual land cover mask to patch GIFs

Requires:
    * API Key.txt containing your Radiant MLHub API key
    * Dataset downloaded via Landcovernet_Download_API.py in this directory

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
import glob
import math
import imageio
import rasterio as rt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.transforms import Bbox
import datetime as dt
from osgeo import gdal, osr
from sklearn.preprocessing import normalize
from alive_progress import alive_bar

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# Path to directory holding dataset
data_dir = 'landcovernet'

# Prefix to every patch ID in every patch directory name
patch_dir_prefix = 'ref_landcovernet_v1_labels_'

# Automatically fixes the layout of the figures to accommodate the colour bar legends
plt.rcParams['figure.constrained_layout.use'] = True

# Create a new projection system in lat-lon
WGS84_4326 = osr.SpatialReference()
WGS84_4326.ImportFromEPSG(4326)

# Downloads required plugin for imageio if not already present
imageio.plugins.freeimage.download()

# ======= RADIANT MLHUB PRESETS =======================================================================================
# Radiant Earth land cover classes reformatted to split across two lines for neater plots
RE_classes = ['No Data',
              'Water',
              'Artificial\nBareground',
              'Natural\nBareground',
              'Permanent\nSnow/Ice',
              'Woody\nVegetation',
              'Cultivated\nVegetation',
              '(Semi) Natural\nVegetation']

# Custom colour mapping specified by Radiant Earth Foundation
RE_cmap_dict = {0: '#FF0000',  # Red
                1: '#0000ff',
                2: '#888888',
                3: '#d1a46d',
                4: '#f5f5ff',
                5: '#d64c2b',
                6: '#186818',
                7: '#00ff00'}

# Custom cmap matching the Radiant Earth Foundation specifications
RE_cmap = ListedColormap(RE_cmap_dict.values(), N=len(RE_classes))

# Pre-set RE figure height and width (in inches)
RE_figdim = (8.02, 10.32)

# ======= SENTINEL-2 L2A SCL PRESETS ==================================================================================
# SCL land cover classes reformatted to split across two lines for neater plots
S2_SCL_classes = ['No Data',
                  'Saturated OR\nDefective',
                  'Dark Area Pixels',
                  'Cloud Shadows',
                  'Vegetation',
                  'Not Vegetated',
                  'Water',
                  'Unclassified',
                  'Cloud Medium\nProbability',
                  'Cloud High\nProbability',
                  'Thin Cirrus',
                  'Snow']

# Custom colour mapping from class definitions in the SENTINEL-2 L2A MSI
S2_SCL_cmap_dict = {0: '#000000',
                    1: '#f71910',
                    2: '#404040',
                    3: '#7f3f0f',
                    4: '#31ff00',
                    5: '#faff10',
                    6: '#2c00cc',
                    7: '#757171',
                    8: '#aeaaaa',
                    9: '#d0cece',
                    10: '#45c8fe',
                    11: '#fc58ff'}

# Custom cmap matching the SENTINEL-2 L2A SCL classes
S2_SCL_cmap = ListedColormap(S2_SCL_cmap_dict.values(), N=len(S2_SCL_classes))

# Preset SCL figure height and width (in inches)
S2_SCL_figdim = (8, 10.44)


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


def datetime_reformat(datetime, fmt1, fmt2):
    """Takes a str representing a time stamp in one format and returns it reformatted into a second

    Args:
        datetime (str): Datetime string to be reformatted
        fmt1 (str): Format of original datetime
        fmt2 (str): New format for datetime

    Returns:
        (str): Datetime reformatted to fmt2
    """
    return dt.datetime.strptime(datetime, fmt1).strftime(fmt2)


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
    date1 = datetime_reformat(names['date'], '%d.%m.%Y', '%Y_%m_%d')
    date2 = datetime_reformat(names['date'], '%d.%m.%Y', '%Y%m%d')

    # Format path to the directory holding all the scene files
    scene_path = '%s/%s%s/%s/' % (data_dir, patch_dir_prefix, names['patch_ID'], date1)

    # Format the name of the file containing the label mask data
    data_name = '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['band_ID'])

    # Create a dictionary of the names of the requested red, green, blue images
    rgb = {'R': '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['R_band']),
           'G': '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['G_band']),
           'B': '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['B_band'])}

    return rgb, scene_path, data_name


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
        bands.append(normalize(load_array(scene_path + rgb[band], 1)))

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


def labelled_rgb_image(names, data_band=1, classes=None, block_size=32, cmap_style=None, alpha=0.5, new_cs=None,
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
    rgb, scene_path, data_name = path_format(names)

    # Stacks together the R, G, & B bands to form an array of the RGB image
    rgb_image = stack_rgb(scene_path, rgb)

    # Loads data to plotted as heatmap from file
    data = load_array(scene_path + data_name, band=data_band)

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
    corners = transform_coordinates(scene_path + data_name, new_cs)

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
    lat_labels = dec2deg(lat_extent, axis='lat')
    lon_labels = dec2deg(lon_extent, axis='lon')

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
    fn = '%s/%s_%s_RGBHM.png' % (scene_path, names['patch_ID'], datetime_reformat(names['date'], '%d.%m.%Y', '%Y%m%d'))

    # If true, save file to fn
    if save:
        # Checks if file already exists. Deletes if true
        exist_delete_check(fn)

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
    dates = date_grab(names['patch_ID'])

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
    patches = patch_grab()

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


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
if __name__ == '__main__':
    # Additional options for names dictionary:
    #           'patch_ID': '31PGS_15',     Five char alpha-numeric SENTINEL tile ID and
    #                                       2 digit int REF MLHub chip (patch) ID ranging from 0-29
    #           'date': '16.04.2018',       Date of scene in DD.MM.YYYY format
    my_names = {'band_ID': 'SCL',           # 3 char alpha-numeric Band ID
                'R_band': 'B02',            # Red, Green, Blue band IDs for RGB images
                'G_band': 'B03',
                'B_band': 'B04'}

    make_all_the_gifs(my_names, frame_length=0.5, data_band=1, classes=S2_SCL_classes, cmap_style=S2_SCL_cmap,
                      new_cs=WGS84_4326, alpha=0.3, figdim=S2_SCL_figdim)
