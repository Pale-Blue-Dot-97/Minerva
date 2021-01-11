"""Tiff_Read

Script to locate, open, read and manipulate .tiff images and datasets downloaded from the Radiant MLHub API using
rasterio.

TODO:
    *

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import Landcovernet_Download_API as ap
import os
import rasterio as rt
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
classes = ap.get_classes()


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_array(path, band):
    raster = rt.open(path)

    data = raster.read(band)

    return data


def plot_heatmap(array, classes=None, cmap=None):
    heatmap = plt.imshow(array, cmap, vmin=0, vmax=len(classes))
    clb = plt.colorbar(heatmap, ticks=range(len(classes)), boundaries=range(len(classes)))
    clb.ax.set_yticklabels(classes)
    plt.show()
    plt.close()


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
fp = 'landcovernet/ref_landcovernet_v1_labels_31PGS_01/2018_01_01/'
fn = '31PGS_01_20180101_SCL_10m.tif'

path = fp + fn

print(classes)
print(type(classes))

plot_heatmap(load_array(path, band=1), classes=classes)




