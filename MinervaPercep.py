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
# import os
import glob
# import math
from abc import ABC
import numpy as np
import pandas as pd
import torch as pt
import Radiant_MLHub_DataVis as rdv
# import rasterio as rt
# from osgeo import gdal, osr
# from alive_progress import alive_bar
# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# Path to directory holding dataset
data_dir = 'landcovernet'

# Prefix to every patch ID in every patch directory name
patch_dir_prefix = 'ref_landcovernet_v1_labels_'


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MLP(pt.nn.Module, ABC):
    def __init__(self, input_size, n_classes):
        super(MLP, self).__init__()
        self.il = pt.nn.Linear(input_size, 2 * input_size)
        self.relu1 = pt.nn.ReLU()
        self.hl = pt.nn.Linear(2 * input_size, 2 * input_size)
        self.relu2 = pt.nn.ReLU()
        self.cl = pt.nn.Linear(2 * input_size, n_classes)
        self.sm = pt.nn.Softmax()

    def forward(self, x):
        hidden1 = self.il(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.hl(relu1)
        relu2 = self.relu2(hidden2)
        output = self.cl(relu2)
        output = self.sm(output)
        return output


class Dataset(pt.utils.data.Dataset):
    """Characterizes a dataset for PyTorch.
    Source: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, list_IDs, labels):
        """Initialization"""
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        x = pt.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return x, y


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def scene_grab(patch_id):
    # Get the name of all the directories for this patch
    scene_dirs = glob.glob('%s/%s%s/*/' % (data_dir, patch_dir_prefix, patch_id))

    # Extract the scene names (i.e the dates) from the paths
    scene_names = [(scene.partition('\\')[2])[:-1] for scene in scene_dirs]

    scenes = []

    for date in scene_names:
        scenes.append(rdv.load_array(
            '%s/%s%s/%s/%s_%s_CLD_10m.tif' % (data_dir, patch_dir_prefix, patch_id, date, patch_id,
                                              rdv.datetime_reformat(date, '%Y_%m_%d', '%Y%m%d')), 1))

    return scenes, scene_names


def cloud_cover(scene):
    return np.sum(scene) / scene.size


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
if __name__ == '__main__':
    #minervaPercep = MLP(24, 12)

    patch = pd.DataFrame()
    patch['SCENE'], patch['DATE'] = scene_grab('38PKT_22')
    patch['COVER'] = patch['SCENE'].apply(cloud_cover)

    patch.set_index(pd.to_datetime(patch['DATE'], format='%Y_%m_%d'), drop=True, inplace=True)
    print(patch)

    print(patch.sort_values(by='COVER'))


