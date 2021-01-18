"""Tiff_Read

Script to locate, open, read and manipulate .tiff images and datasets downloaded from the Radiant MLHub API using
rasterio.

TODO:
    * Fully document

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import rasterio as rt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.transforms import Bbox
import datetime as dt
from osgeo import gdal, osr
import math
import glob
import os
import imageio
from tqdm import tqdm
# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
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

# Automatically fixes the layout of the figures to accommodate the colour bar legends
plt.rcParams['figure.constrained_layout.use'] = True

# Create a new projection system in lat-lon
WGS84_4326 = osr.SpatialReference()
WGS84_4326.ImportFromEPSG(4326)

imageio.plugins.freeimage.download()


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def exist_delete_check(fn):
    if os.path.exists(fn):
        os.remove(fn)
    else:
        pass


def date_format(date, fmt1, fmt2):
    return dt.datetime.strptime(date, fmt1).strftime(fmt2)


def patch_grab():
    patch_dirs = glob.glob('landcovernet/ref_landcovernet_v1_labels_*/')

    return [(patch.partition('ref_landcovernet_v1_labels_')[2])[:-1] for patch in patch_dirs]


def date_grab(names):
    scene_dirs = glob.glob('landcovernet/ref_landcovernet_v1_labels_%s/*/' % names['patch_ID'])

    scene_names = [(scene.partition('\\')[2])[:-1] for scene in scene_dirs]

    return [date_format(date, '%Y_%m_%d', '%d.%m.%Y') for date in scene_names]


def path_format(names):
    date1 = date_format(names['date'], '%d.%m.%Y', '%Y_%m_%d')
    date2 = date_format(names['date'], '%d.%m.%Y', '%Y%m%d')

    scene_path = 'landcovernet/ref_landcovernet_v1_labels_%s/%s/' % (names['patch_ID'], date1)
    data_name = '%s_%s_%s_10m.tif' % (names['patch_ID'], date2, names['band_ID'])

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
    # Open GeoTiff in GDAL
    ds = gdal.Open(path)
    w = ds.RasterXSize
    h = ds.RasterYSize

    # Create SpatialReference object
    old_cs = osr.SpatialReference()

    # Fetch projection system from GeoTiff and set to old_cs
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    trans = osr.CoordinateTransformation(old_cs, new_cs)

    gt_data = ds.GetGeoTransform()
    min_x = gt_data[0]
    min_y = gt_data[3] + w * gt_data[4] + h * gt_data[5]
    max_x = gt_data[0] + w * gt_data[1] + h * gt_data[2]
    max_y = gt_data[3]

    return [[trans.TransformPoint(min_x, max_y)[:2], trans.TransformPoint(max_x, max_y)[:2]],
            [trans.TransformPoint(min_x, min_y)[:2], trans.TransformPoint(max_x, min_y)[:2]]]


def deg_to_dms(deg, axis='lat'):
    """Credit to Gustavo Gonçalves on Stack Overflow
    https://stackoverflow.com/questions/2579535/convert-dd-decimal-degrees-to-dms-degrees-minutes-seconds-in-python

    Args:
        deg:
        axis:

    Returns:

    """
    decimals, number = math.modf(deg)
    d = int(number)
    m = int(decimals * 60)
    s = (deg - d - m / 60) * 3600.00
    compass = {
        'lat': ('N', 'S'),
        'lon': ('E', 'W')
    }
    compass_str = compass[axis][0 if d >= 0 else 1]
    return '{}º{}\'{:.0f}"{}'.format(abs(d), abs(m), abs(s), compass_str)


def dec2deg(dec_co, axis='lat'):
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
    def normalise(array):
        """Normalise bands into 0.0 - 1.0 scale

        Args:
            array ([float]): Array to be normalised

        Returns:
            Normalised array
        """
        array_min, array_max = array.min(), array.max()
        return (array - array_min) / (array_max - array_min)

    # Load R, G, B images from file and normalise
    bands = []
    for band in ['R', 'G', 'B']:
        bands.append(normalise(load_array(scene_path + rgb[band], 1)))

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
                       show=True, save=True):
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
    fig.set_figheight(8.02)
    fig.set_figwidth(10.32)

    # Display figure
    if show:
        plt.show()

    # Path and file name of figure
    fn = '%s/%s_%s_RGBHM.png' % (scene_path, names['patch_ID'], date_format(names['date'], '%d.%m.%Y', '%Y%m%d'))

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
             save=False):
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

    Returns:
        None

    """

    dates = date_grab(names)

    pb = tqdm(total=100)

    frames = []
    for date in dates:
        pb.set_description('Scene on %s' % date)
        names['date'] = date
        frame = labelled_rgb_image(names, data_band=data_band, classes=classes, cmap_style=cmap_style, new_cs=new_cs,
                                   alpha=alpha, save=save, show=False)

        frames.append(imageio.imread(frame))
        pb.update(1)

    pb.set_description('MAKING GIF')
    imageio.mimsave(gif_name, frames, 'GIF-FI', duration=frame_length, quantizer='nq')
    pb.update(100 - len(dates))
    pb.close()


def make_all_the_gifs(names, frame_length=1.0, data_band=1, classes=None, cmap_style=None, new_cs=None, alpha=0.5):
    """Wrapper to make_gifs() to iterate through all patches in dataset

    Args:
        names (dict): Dictionary holding the band IDs. Patch ID and date added per iteration
        frame_length (float): Length of each GIF frame in seconds
        data_band (int): Band number of data .tif file
        classes ([str]): List of all possible class labels
        cmap_style (str, ListedColormap): Name or object for colour map style
        new_cs(SpatialReference): Co-ordinate system to convert image to and use for labelling
        alpha (float): Fraction determining alpha blending of label mask

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
        gif_name = 'landcovernet/ref_landcovernet_v1_labels_%s/%s.gif' % (names['patch_ID'], names['patch_ID'])

        # Call make_gif() for this patch
        make_gif(names, gif_name, frame_length=frame_length, data_band=data_band, classes=classes,
                 cmap_style=cmap_style, new_cs=new_cs, alpha=alpha, save=True)

    print('\r\nOPERATION COMPLETE')


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
if __name__ == '__main__':
    # Additional options for names dictionary:
    # 'patch_ID': '31PGS_15',     Five char alpha-numeric SENTINEL tile ID and
    #                             2 digit int REF MLHub chip (patch) ID ranging from 0-29
    # 'date': '16.04.2018',       Date of scene in DD.MM.YYYY format
    my_names = {'band_ID': 'SCL',           # 3 char alpha-numeric Band ID
                'R_band': 'B02',            # Red, Green, Blue band IDs for RGB images
                'G_band': 'B03',
                'B_band': 'B04'}

    make_all_the_gifs(my_names, frame_length=0.5, data_band=1, classes=RE_classes, cmap_style=RE_cmap,
                      new_cs=WGS84_4326, alpha=0.3)
