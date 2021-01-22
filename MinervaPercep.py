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
#import os
#import glob
#import math
from abc import ABC
#import numpy as np
import torch as pt
#import rasterio as rt
#from osgeo import gdal, osr
#from alive_progress import alive_bar
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
class MLP(pt.nn.Module, ABC):
    def __init__(self):
        super(MLP, self).__init__()
        self.il = pt.nn.Linear(24, 48)
        self.relu1 = pt.nn.ReLU()
        self.hl = pt.nn.Linear(48, 48)
        self.relu2 = pt.nn.ReLU()
        self.cl = pt.nn.Linear(48, 12)
        self.sm = pt.nn.Softmax()

    def forward(self, x):
        hidden1 = self.il(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.hl(relu1)
        relu2 = self.relu2(hidden2)
        output = self.cl(relu2)
        output = self.sm(output)
        return output


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
if __name__ == '__main__':
    minervaPercep = MLP()
