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
def plot_heatmap(array):
    cmap = 'rainbow'
    heatmap = pyplot.imshow(array, cmap)
    heatmap.show()


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
fp = 'landcovernet/ref_landcovernet_v1_labels_31PGS_01'

os.chdir(fp)

fn = '31PGS_01_2018_LC_10m.tif'

raster = rt.open(fn)

data = raster.read()
print(data.shape)

data1 = raster.read(1).tolist()
print(data1)
print(mode(data1, axis=None)[0])

data2 = raster.read(2).tolist()
print(data2)
print(mode(data2, axis=None)[0])

os.chdir('2018_01_01')



