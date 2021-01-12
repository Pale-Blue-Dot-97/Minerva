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
from osgeo import gdal
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

plt.rcParams['figure.constrained_layout.use'] = True


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_array(path, band):
    raster = rt.open(path)

    data = raster.read(band)

    return data


def discrete_heatmap(array, classes=None, cmap_style=None):
    """Plots a heatmap with a discrete colour bar. Designed for Radiant Earth MLHub 256x256 SENTINEL images

    Args:
        array (array_like): 2D Array of data to be plotted as a heat map
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
    heatmap = plt.imshow(array, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5)

    # Sets tick intervals to standard 32x32 block size
    plt.xticks(np.arange(0, array.shape[0] + 1, 32))
    plt.yticks(np.arange(0, array.shape[1] + 1, 32))

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


def stack_RGB(scene_path, r_name, g_name, b_name):
    def normalise(array):
        """Normalise bands into 0.0 - 1.0 scale

        Args:
            array:

        Returns:

        """
        array_min, array_max = array.min(), array.max()
        return (array - array_min) / (array_max - array_min)

    # Load R, G, B images from file
    r_image = load_array(scene_path + r_name, 1)
    g_image = load_array(scene_path + g_name, 1)
    b_image = load_array(scene_path + b_name, 1)

    # Normalise all arrays and stack together.
    # Note that it has to be order BGR not RGB due to the order numpy stacks arrays
    return np.dstack((normalise(b_image), normalise(g_image), normalise(r_image)))


def RGB_image(scene_path, r_name, g_name, b_name):
    # Stack RGB image data together
    rgb_image_array = stack_RGB(scene_path, r_name, g_name, b_name)

    # Create RGB image
    rgb_image = plt.imshow(rgb_image_array)

    # Sets tick intervals to standard 32x32 block size
    plt.xticks(np.arange(0, rgb_image_array.shape[0] + 1, 32))
    plt.yticks(np.arange(0, rgb_image_array.shape[1] + 1, 32))

    # Add grid overlay
    plt.grid(which='both', color='#CCCCCC', linestyle=':')

    plt.show()

    return rgb_image


def masked_RGB_image(scene_path, r_name, g_name, b_name, data, classes=None, cmap_style=None, alpha=0.5):
    rgb_image = stack_RGB(scene_path, r_name, g_name, b_name)

    extent = 0, data.shape[0], 0, data.shape[1]

    # Initialises a figure
    plt.figure()

    # Create RGB image
    plt.imshow(rgb_image, extent=extent)

    # Creates a cmap from query
    cmap = plt.get_cmap(cmap_style, len(classes))

    # Plots heatmap onto figure
    heatmap = plt.imshow(data, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5, extent=extent, alpha=alpha)

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


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
# Five char alpha-numeric SENTINEL tile ID
tile_ID = '38PKT'

# 2 digit int SENTINEL chip ID ranging from 0-29
chip_ID = '22'

# Date of scene in DD.MM.YYYY format
date = '08.10.2018'

# 3 char alpha-numeric Band ID
band_ID = 'SCL'

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


RGB_image(fp, r_name, g_name, b_name)

path = fp + fn

discrete_heatmap(load_array(path, band=1), classes=classes, cmap_style=RE_cmap)

masked_RGB_image(fp, r_name, g_name, b_name, load_array(path, band=1), classes=classes, cmap_style=RE_cmap)
