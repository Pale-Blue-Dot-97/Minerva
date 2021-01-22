"""MinervaPercep

Script to create a simple MLP to classify land cover of the images in the LandCoverNet V1 dataset

TODO:
    * Add methods to locate, load, pre-process and arrange data and labels from LandCoverNet
    * Add methods to split the data into train and test
    * Construct a simple MLP
    * Add methods to visualise and analyse results of training and testing
    * Add method to save models to file

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import rasterio as rt
import numpy as np
from osgeo import gdal, osr
import math
import glob
import os
from alive_progress import alive_bar
# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# Path to directory holding dataset
data_dir = 'landcovernet'

# Prefix to every patch ID in every patch directory name
patch_dir_prefix = 'ref_landcovernet_v1_labels_'


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================

# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
if __name__ == '__main__':
    pass
