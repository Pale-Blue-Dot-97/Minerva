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
import rasterio as rt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

RE_cmap_list = ['#FF0000', '#0000ff', '#888888', '#d1a46d', '#f5f5ff', '#d64c2b', '#186818', '#00ff00']

# Custom cmap matching the Radiant Earth Foundation specifications
RE_cmap = ListedColormap(RE_cmap_list, N=len(classes))


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_array(path, band):
    raster = rt.open(path)

    data = raster.read(band)

    return data


def discrete_heatmap(array, classes=None, cmap_style=None):
    cmap = plt.get_cmap(cmap_style, len(classes))
    heatmap = plt.matshow(array, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5)
    clb = plt.colorbar(heatmap, ticks=np.arange(0, len(classes)))
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

discrete_heatmap(load_array(path, band=1), classes=classes, cmap_style=RE_cmap)




