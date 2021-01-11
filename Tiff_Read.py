"""Tiff_Read

Script to locate, open, read and manipulate .tiff images and datasets downloaded from the Radiant MLHub API using
rasterio.

TODO:
    *

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
import rasterio as rt
import numpy as np
from scipy.stats import mode
from matplotlib import pyplot
# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_array(path, band):
    raster = rt.open(path)

    data = raster.read(band)

    return data


def plot_heatmap(array, classes=None, cmap=None):
    pyplot.imshow(array, cmap)
    pyplot.show()
    pyplot.close()


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
fp = 'landcovernet/ref_landcovernet_v1_labels_31PGS_01/2018_01_01/'
fn = '31PGS_01_20180101_SCL_10m.tif'

path = fp + fn

plot_heatmap(load_array(path, 1))




