"""Tiff_Read

Script to locate, open, read and manipulate .tiff images and datasets downloaded from the Radiant MLHub API using
rasterio.

TODO:
    * Generalise to plot heatmaps of all label masks
    * Fully document

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import Landcovernet_Download_API as ap
import rasterio as rt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime as dt
from osgeo import gdal, osr
# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
classes = ap.get_classes()

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
RE_cmap = ListedColormap(RE_cmap_dict.values(), N=len(classes))

# Automatically fixes the layout of the figures to accommodate the colour bar legends
plt.rcParams['figure.constrained_layout.use'] = True


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
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


def stack_RGB(scene_path, rgb):
    """Stacks together red, green and blue image arrays from file to create a RGB array

    Args:
        scene_path (str): Path to directory holding images from desired scene
        rgb ([str]): List of filenames of R, G & B band images

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
    for band in rgb:
        bands.append(normalise(load_array(scene_path + band, 1)))

    # Stack together RGB bands
    # Note that it has to be order BGR not RGB due to the order numpy stacks arrays
    return np.dstack((bands[2], bands[1], bands[0]))


def RGB_image(scene_path, rgb):
    """Creates an RGB image from a composition of red, green and blue band .tif images

    Args:
        scene_path (str): Path to directory holding images from desired scene
        rgb ([str]): List of filenames of R, G & B band images

    Returns:
        rgb_image (AxesImage): Plotted RGB image object
    """
    # Stack RGB image data together
    rgb_image_array = stack_RGB(scene_path, rgb)

    # Create RGB image
    rgb_image = plt.imshow(rgb_image_array)

    # Sets tick intervals to standard 32x32 block size
    plt.xticks(np.arange(0, rgb_image_array.shape[0] + 1, 32))
    plt.yticks(np.arange(0, rgb_image_array.shape[1] + 1, 32))

    # Add grid overlay
    plt.grid(which='both', color='#CCCCCC', linestyle=':')

    plt.show()

    return rgb_image


def labelled_RGB_image(scene_path, rgb, data_path, data_band=1, classes=None, cmap_style=None, alpha=0.5, new_cs=None):
    """Produces a layered image of an RGB image and it's associated label mask heat map alpha blended on top

    Args:
        scene_path (str): Path to directory holding images from desired scene
        rgb ([str]): List of filenames of R, G & B band images
        data_path (str): Path to tif data file to be plotted as a heat map
        data_band (int): Band number of data .tif file
        classes ([str]): List of all possible class labels
        cmap_style (str, ListedColormap): Name or object for colour map style
        alpha (float): Fraction determining alpha blending of label mask

    Returns:
        None

    """
    # Stacks together the R, G, & B bands to form an array of the RGB image
    rgb_image = stack_RGB(scene_path, rgb)

    # Loads data to plotted as heatmap from file
    data = load_array(data_path, band=data_band)

    # Defines the 'extent' of the composite image based on the size of the mask.
    # Assumes mask and RGB image have same 2D shape
    extent = 0, data.shape[0], 0, data.shape[1]

    # Initialises a figure
    fig, ax1 = plt.figure()

    # Create RGB image
    ax1.imshow(rgb_image, extent=extent)

    ax2 = ax1.twinx().twiny()

    # Creates a cmap from query
    cmap = plt.get_cmap(cmap_style, len(classes))

    # Plots heatmap onto figure
    heatmap = ax2.imshow(data, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5, extent=extent, alpha=alpha)

    # Sets tick intervals to standard 32x32 block size
    ax1.set_xticks(np.arange(0, data.shape[0] + 1, 32))
    ax1.set_yticks(np.arange(0, data.shape[1] + 1, 32))

    lat_lon_corners = transform_coordinates((path+rgb[0]), new_cs)
    
    ax2.set_xticks()

    # Add grid overlay
    ax1.grid(which='both', color='#CCCCCC', linestyle=':')

    # Plots colour bar onto figure
    clb = ax2.colorbar(heatmap, ticks=np.arange(0, len(classes)), shrink=0.77)

    # Sets colour bar ticks to class labels
    clb.ax.set_yticklabels(classes)

    # Display figure
    plt.show()

    # Close figure
    plt.close()


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
# Five char alpha-numeric SENTINEL tile ID
tile_ID = '38PKT'

# 2 digit int SENTINEL chip ID ranging from 0-29
chip_ID = '22'

# Date of scene in DD.MM.YYYY format
date = '16.04.2018'

# 3 char alpha-numeric Band ID
band_ID = 'B01'

# Red, Green, Blue band IDs for RGB images
R_band = 'B02'
G_band = 'B03'
B_band = 'B04'

stamp = dt.datetime.strptime(date, '%d.%m.%Y')

date1 = stamp.strftime('%Y_%m_%d')
date2 = stamp.strftime('%Y%m%d')

fp = 'landcovernet/ref_landcovernet_v1_labels_%s_%s/%s/' % (tile_ID, chip_ID, date1)
fn = '%s_%s_%s_%s_10m.tif' % (tile_ID, chip_ID, date2, band_ID)

r_name = '%s_%s_%s_%s_10m.tif' % (tile_ID, chip_ID, date2, R_band)
g_name = '%s_%s_%s_%s_10m.tif' % (tile_ID, chip_ID, date2, G_band)
b_name = '%s_%s_%s_%s_10m.tif' % (tile_ID, chip_ID, date2, B_band)

path = fp + fn

# Create a new projection system in lat-lon
new_cs = osr.SpatialReference()
new_cs.ImportFromEPSG(4326)

transform_coordinates(path, new_cs)

"""
RGB_image(fp, (r_name, g_name, b_name))

discrete_heatmap(load_array(path, band=1), classes=classes, cmap_style=RE_cmap)

labelled_RGB_image(fp, (r_name, g_name, b_name), scl_name, band=1, classes=classes, cmap_style=RE_cmap)
"""
