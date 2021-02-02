"""MinervaPercep

Script to create a simple MLP to classify land cover of the images in the LandCoverNet V1 dataset

TODO:
    * Construct a simple MLP
    * Add methods to visualise and analyse results of training and testing
    * Add method to save models to file

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import glob
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, OneHotEncoder
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.backends import cudnn
from itertools import cycle, chain, islice
import Radiant_MLHub_DataVis as rdv
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from collections import Counter
# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# Path to directory holding dataset
data_dir = 'landcovernet'

# Prefix to every patch ID in every patch directory name
patch_dir_prefix = 'ref_landcovernet_v1_labels_'

# Band IDs of SENTINEL-2 images contained in the LandCoverNet dataset
band_ids = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

# Defines size of the images to determine the number of batches
image_size = (256, 256)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

# Parameters
params = {'batch_size': 32,
          #'shuffle': True,
          'num_workers': 2}

# Creates an One Hot Encoder (OHE) to convert labels
ohe = OneHotEncoder()


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MLP(torch.nn.Module):
    """
    Simple class to construct a Multi-Layer Perceptron (MLP)
    """
    def __init__(self, input_size, n_classes):
        super(MLP, self).__init__()
        self.il = torch.nn.Linear(input_size, 2 * input_size)
        self.relu1 = torch.nn.ReLU()
        self.hl = torch.nn.Linear(2 * input_size, 2 * input_size)
        self.relu2 = torch.nn.ReLU()
        self.cl = torch.nn.Linear(2 * input_size, n_classes)
        self.sm = torch.nn.Sigmoid()

    def forward(self, x):
        """Performs a forward pass of the network

        Args:
            self
            x (torch.Tensor):

        Returns:

        """
        hidden1 = self.il(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.hl(relu1)
        relu2 = self.relu2(hidden2)
        output = self.cl(relu2)
        return self.sm(output)


class BatchLoader(IterableDataset):
    """
    Source: https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    """
    def __init__(self, patch_ids, batch_size):
        self.patch_ids = patch_ids
        self.batch_size = batch_size

    @property
    def shuffled_data_list(self):
        return random.sample(self.patch_ids, len(self.patch_ids))

    def process_data(self, patch_id):
        patch = make_time_series(patch_id)
        flat_patch = patch.reshape(-1, *patch.shape[-2:])

        labels = np.int16(np.array(lc_load(patch_id)).flatten())

        for i in range(len(labels)):
            yield flat_patch[i].flatten(), labels[i]

    def get_stream(self, patch_ids):
        return chain.from_iterable(map(self.process_data, cycle(patch_ids)))

    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_data_list) for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_stream(self.patch_ids)


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def prefix_format(patch_id, scene):
    """Formats a string representing the prefix of a path to any file in a scene

    Args:
        patch_id (str): Unique patch ID
        scene (str): Date of scene in YY_MM_DD format

    Returns:
        prefix (str): Prefix of path to any file in a given scene
    """
    return '%s/%s%s/%s/%s_%s' % (data_dir, patch_dir_prefix, patch_id, scene, patch_id,
                                 rdv.datetime_reformat(scene, '%Y_%m_%d', '%Y%m%d'))


def scene_grab(patch_id):
    """Finds and loads all CLDs for a given patch

    Args:
        patch_id (str): Unique patch ID

    Returns:
        scenes (list): List of CLD masks for each scene
        scene_names (list): List of scene dates in YY_MM_DD

    """
    # Get the name of all the directories for this patch
    scene_dirs = glob.glob('%s/%s%s/*/' % (data_dir, patch_dir_prefix, patch_id))

    # Extract the scene names (i.e the dates) from the paths
    scene_names = [(scene.partition('\\')[2])[:-1] for scene in scene_dirs]

    # List to hold scenes
    scenes = []

    # Finds and appends each CLD of each scene of a patch to scenes
    for date in scene_names:
        scenes.append(rdv.load_array('%s_CLD_10m.tif' % prefix_format(patch_id, date), 1))

    return scenes, scene_names


def lc_load(patch_id):
    """Loads the LC labels for a given patch

    Args:
        patch_id (str): Unique patch ID

    Returns:
        LC_label (list): 2D array containing LC labels for each pixel of a patch

    """
    return rdv.load_array('%s/%s%s/%s_2018_LC_10m.tif' % (data_dir, patch_dir_prefix, patch_id, patch_id), 1)


def cloud_cover(scene):
    """Calculates percentage cloud cover for a given scene based on its scene CLD

    Args:
        scene (numpy.ndarray):

    Returns:
        (float): Percentage cloud cover
    """
    return np.sum(scene) / scene.size


def month_sort(df, month):
    """Finds the the scene with the lowest cloud cover in a given month

    Args:
        df (pandas.DataFrame): Dataframe containing all scenes and their cloud cover percentages
        month (str): Month of a year to sort

    Returns:
        (str): Date of the scene with the lowest cloud cover percentage for the given month
    """
    return df.loc[month].sort_values(by='COVER')['DATE'][0]


def scene_selection(df):
    """Selects the 24 best scenes of a patch based on REF's 2-step selection criteria

    Args:
        df (pandas.DataFrame): Dataframe containing all scenes and their cloud cover percentages

    Returns:
        scene_names (list): List of 24 strings representing dates of the 24 selected scenes in YY_MM_DD format
    """
    # Step 1: Find scene with lowest cloud cover percentage in each month
    step1 = []
    for month in range(1, 13):
        step1.append(month_sort(df, '%d-2018' % month))

    # Step 2: Find the 12 scenes with the lowest cloud cover percentage of the remaining scenes
    df.drop(index=pd.to_datetime(step1, format='%Y_%m_%d'), inplace=True)
    step2 = df.sort_values(by='COVER')['DATE'][:12].tolist()

    # Return 24 scenes selected by the 2-step REF criteria
    return step1 + step2


def find_best_of(patch_id):
    """Finds the 24 scenes sorted by cloud cover according to REF's 2-step criteria using scene_selection()

    Args:
        patch_id (str): Unique patch ID

    Returns:
        scene_names (list): List of 24 strings representing dates of the 24 selected scenes in YY_MM_DD format
    """
    # Creates a DataFrame
    patch = pd.DataFrame()

    # Using scene_grab(), gets all the scene CLDs and dates for the given patch and adds to DataFrame
    patch['SCENE'], patch['DATE'] = scene_grab(patch_id)

    # Calculates the cloud cover percentage for every scene and adds to DataFrame
    patch['COVER'] = patch['SCENE'].apply(cloud_cover)

    # Removes unneeded scene column
    del patch['SCENE']

    # Re-indexes the DataFrame to datetime
    patch.set_index(pd.to_datetime(patch['DATE'], format='%Y_%m_%d'), drop=True, inplace=True)

    # Sends DataFrame to scene_selection() and returns the 24 selected scenes
    return scene_selection(patch)


def stack_bands(patch_id, scene):
    """Stacks together all the bands of the SENTINEL-2 images in a given scene of a patch

    Args:
        patch_id (str): Unique patch ID
        scene (str): Date of scene in YY_MM_DD format to stack bands in

    Returns:
        Normalised and stacked red, green, blue arrays into RGB array
    """
    bands = []
    # Load R, G, B images from file and normalise
    for band in band_ids:
        bands.append(normalize(rdv.load_array('%s_%s_10m.tif' % (prefix_format(patch_id, scene), band), 1)))

    # Stack together RGB bands
    # Note that it has to be order BGR not RGB due to the order numpy stacks arrays
    return np.dstack(bands)


def make_time_series(patch_id):
    """Makes a time-series of each pixel of a patch across 24 scenes selected by REF's criteria using scene_selection().
     All the bands in the chosen scene are stacked using stack_bands()

    Args:
        patch_id (str): Unique patch ID

    Returns:
        (numpy.ndarray): Array of shape(rows, columns, 24, 12) holding all x for a patch
    """
    # List of scene dates found by REF's selection criteria
    scenes = find_best_of(patch_id)

    # Loads all pixels in a patch across the 24 scenes and 12 bands
    x = []
    for scene in scenes:
        x.append(stack_bands(patch_id, scene))

    # Returns a reordered numpy.ndarray holding all x for the given patch
    return np.moveaxis(np.array(x), 0, 2)


def class_balance(ids):
    """Loads all LC labels for the given patches using lc_load() then plots the class subpopulations using
    plot_subpopulations()

    Args:
        ids (list): List of patch IDs to analyse

    Returns:
        None
    """
    # Loads all LC label masks for the given patch IDs
    labels = []
    for patch_id in ids:
        labels.append(lc_load(patch_id))

    # Plots a pie chart of the distribution of the classes within the given list of patches
    plot_subpopulations(np.array(labels).flatten())


def num_batches(ids):
    return int((len(ids) * image_size[0] * image_size[1]) / params['batch_size'])


def plot_subpopulations(class_labels):
    """Creates a pie chart of the distribution of the classes within the data

    Args:
        class_labels ([int]): List of predicted classifications from model, in form of class numbers

    Returns:
        None
    """

    # Finds the distribution of the classes within the data
    modes = Counter(class_labels).most_common()

    # List to hold the name and percentage distribution of each class in the data as str
    classes = []

    # List to hold the total counts of each class
    counts = []

    # Finds total number of images to normalise data
    n_images = len(class_labels)

    # For each class, find the percentage of data that is that class and the total counts for that class
    for label in modes:
        classes.append('{} ({:.2f})'.format(label[0], (label[1] / n_images)))
        counts.append(label[1])

    # Plot a pie chart of the data distribution amongst the classes with labels of class name and percentage size
    plt.pie(counts, labels=classes)

    # Show plot for review
    plt.show()


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    max_epochs = 5

    patch_ids = rdv.patch_grab()
    train_ids, test_ids = train_test_split(patch_ids, train_size=0.1, test_size=0.9, shuffle=True, random_state=42)

    class_balance(patch_ids)

    train_dataset = BatchLoader(train_ids, batch_size=params['batch_size'])
    train_loader = DataLoader(train_dataset, **params)

    test_dataset = BatchLoader(test_ids, batch_size=params['batch_size'])
    test_loader = DataLoader(test_dataset, **params)

    model = MLP(288, 8)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    model.train()
    torch.set_grad_enabled(True)

    losses = []

    for epoch in range(max_epochs):
        # batch_num = 1
        with alive_bar(num_batches(train_ids), bar='blocks') as bar:
            for x_batch, y_batch in islice(train_loader, num_batches(train_ids)):
                # batch_num = batch_num + 1

                # Transfer to GPU
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimiser.zero_grad()

                # Forward pass
                y_pred = model(x_batch.float())

                # Compute Loss
                loss = criterion(y_pred.squeeze(), y_batch.long())

                # Backward pass
                loss.backward()
                optimiser.step()

                bar()

                losses.append(loss)

        print('\r\nEpoch {}: train loss: {}'.format(epoch, losses[-1].item()))

    plt.plot(np.array(losses))
    plt.show()
    # for x_batch, y_batch in islice(test_loader, num_batches(test_ids)):
    # print(x_batch)
    # print(y_batch)


if __name__ == '__main__':
    main()

