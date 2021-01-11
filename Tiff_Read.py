"""Tiff_Read

Script to locate, open, read and manipulate .tiff images and datasets downloaded from the Radiant MLHub API using
rasterio.

TODO:
    * Generalise to plot heatmaps of all label masks
    * Add ability to plot label masks over image data
    * Fully document

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


def discrete_heatmap(array, classes=None, cmap_style=None):
    cmap = plt.get_cmap(cmap_style, len(classes))
    heatmap = plt.matshow(array, cmap=cmap, vmin=0.5, vmax=len(classes) + 0.5)
    clb = plt.colorbar(heatmap, ticks=np.arange(1, len(classes) + 1))
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

discrete_heatmap(load_array(path, band=1), classes=classes)




