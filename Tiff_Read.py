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
# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================

# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================

fp = 'landcovernet/ref_landcovernet_v1_labels_29NMG_27'

os.chdir(fp)

fn = '29NMG_27_2018_LC_10m.tif'

raster = rt.open(fn)

print('Shape:', raster.shape)
print('Type:', type(raster))
print('CRS:', raster.crs)
#print('Meta:', data.attrs)

data = raster.read()
print(data.shape)

data1 = raster.read(1)
print(data1)
print(mode(data1, axis=None)[0])

data2 = raster.read(2)
print(data2)
print(mode(data2, axis=None)[0])
