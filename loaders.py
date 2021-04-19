"""loaders

Module containing classes defining custom dataloaders for use in the fitting of neural networks

TODO:
    * Fully document
    * Add Loader for whole images

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from utils import utils
import random
from abc import ABC
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset
from itertools import cycle, chain
from collections import deque


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class BalancedBatchLoader(IterableDataset, ABC):
    """Adaptation of BatchLoader to load data with perfect class balance
    """

    def __init__(self, class_streams, batch_size=32, wheel_size=65536, patch_len=65536):
        """Initialisation

        Args:
            class_streams (pandas.DataFrame): DataFrame with a column of patch IDs for each class
            batch_size (int): Sets the number of samples in each batch
        """
        self.streams_df = class_streams
        self.batch_size = batch_size
        self.patch_len = patch_len

        # Dict to hold a `wheel' for each class
        self.wheels = {}

        # Initialise each wheel with a maximum length of wheel_size global parameter
        for cls in self.streams_df.columns.to_list():
            self.wheels[cls] = deque(maxlen=wheel_size)

        # Loads the patches from the row of IDs supplied into a pandas.Series of pandas.DataFrames
        patches = pd.Series([self.load_patch_df(patch_id) for patch_id in self.streams_df.sample(frac=1).iloc[0]])
        patches.apply(self.refresh_wheels)

        # Checks if wheel is empty after adding to wheel
        for cls in self.streams_df.columns.to_list():
            if not self.wheels[cls]:
                print('EMPTY WHEEL {}!'.format(cls))
                self.emergency_fill(cls)

    def load_patch_df(self, patch_id):
        """Loads a patch using patch ID from disk into a Pandas.DataFrame and returns

        Args:
            patch_id (str): ID for patch to be loaded

        Returns:
            df (pandas.DataFrame): Patch loaded into a DataFrame
        """
        # Initialise DataFrame object
        df = pd.DataFrame()

        # Load patch from disk and create time-series pixel stacks
        patch = utils.make_time_series(patch_id)

        # Reshape patch
        patch = patch.reshape((patch.shape[0] * patch.shape[1], patch.shape[2] * patch.shape[3]))

        # Loads accompanying labels from file and flattens
        labels = utils.lc_load(patch_id).flatten()

        # Wraps each pixel stack in an numpy.array, appends to a list and adds as a column to df
        df['PATCH'] = [np.array(pixel) for pixel in patch]

        # Adds labels as a column to df
        df['LABELS'] = labels

        return df

    def load_patches(self, row):
        """ Loads the patches associated with the patch IDs in row as pandas.DataFrames nested into a pandas.Series
        object

        Args:
            row (pandas.Series): A row of patch IDs

        Returns:
            (pandas.Series): Series of DataFrames of patches
        """
        return pd.Series([self.load_patch_df(row[1][cls]) for cls in self.streams_df.columns.to_list()])

    def refresh_wheels(self, patch_df):
        for cls in self.streams_df.columns.to_list():
            for pixel in patch_df['PATCH'].loc[patch_df['LABELS'] == cls]:
                self.wheels[cls].appendleft(pixel.flatten())

    # THIS IS BODGY AF
    def emergency_fill(self, cls):
        print('EMERGENCY FILL INITIATED')
        patches = pd.Series([self.load_patch_df(patch_id) for patch_id in self.streams_df[cls].sample(frac=1)])

        for patch in patches:
            print('ATTEMPTING TO INIT WHEEL')
            for pixel in patch['PATCH'].loc[patch['LABELS'] == cls]:
                self.wheels[cls].appendleft(pixel.flatten())

            if self.wheels[cls]:
                print('CRISIS OVER')
                return

    def process_data(self, row):
        """Loads and processes patches into wheels for each class and yields from them,
        periodically refreshing the wheels with new data

        Args:
            row (pandas.Series): Randomly selected row of patch IDs, one for each class

        Yields:
            x (torch.Tensor): A data sample as tensor
            y (torch.Tensor): Corresponding label as int tensor
        """
        # Iterates for the flattened length of a patch and yields x and y for each class from their respective wheels
        for i in range(self.patch_len):
            if i == 0:
                # Loads the patches from the row of IDs supplied into a pandas.Series of pandas.DataFrames
                patches = self.load_patches(row)
                patches.apply(self.refresh_wheels)

            # For every class in the dataset, rotate the corresponding wheel and yield the pixel stack from position [0]
            for cls in self.streams_df.columns.to_list():
                # Rotate current class's wheel 1 turn
                self.wheels[cls].rotate(1)

                # Yield pixel stack at position [0] for this class's wheel and the corresponding class label
                # i.e this class number as a tensor int
                yield torch.tensor(self.wheels[cls][0].flatten(), dtype=torch.float), \
                      torch.tensor(cls, dtype=torch.long)

    def get_stream(self, streams_df):
        return chain.from_iterable(map(self.process_data, streams_df.iterrows()))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # If single threaded process, return full ID stream
        if worker_info is None:
            return self.get_stream(self.streams_df)

        # If multi-threaded, split patch IDs between workers
        else:
            # Calculate fraction of dataset per worker
            per_worker = int(np.math.ceil(1.0 / float(worker_info.num_workers)))

            # Return a random sample of the patch IDs of fractional size per worker
            # and using random seed modulated by the worker ID
            return self.get_stream(self.streams_df.sample(frac=per_worker, random_state=42 * worker_info.id,
                                                          replace=False, axis=0))


class BatchLoader(IterableDataset, ABC):
    """
    Source: https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    """

    def __init__(self, patch_ids, batch_size):
        self.patch_ids = patch_ids
        self.batch_size = batch_size

    def process_data(self, patch_id):
        patch = utils.make_time_series(patch_id)

        x = torch.tensor([pixel.flatten() for pixel in patch.reshape(-1, *patch.shape[-2:])], dtype=torch.float)
        y = torch.tensor(np.array(utils.lc_load(patch_id), dtype=np.int64).flatten(), dtype=torch.long)

        for i in range(len(y)):
            yield x[i], y[i]

    def get_stream(self, patch_ids):
        return chain.from_iterable(map(self.process_data, cycle(patch_ids)))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # If single threaded process, return all patch IDs
        if worker_info is None:
            return self.get_stream(self.patch_ids)

        # If multi-threaded, split patch IDs between workers
        else:
            # Calculate number of patch IDs of the dataset per worker
            per_worker = int(np.math.ceil(len(self.patch_ids) / float(worker_info.num_workers)))

            # Set random seed modulated by the worker ID
            random.seed(42 * worker_info.id)

            # Return a random sample of the patch IDs of size per worker
            return self.get_stream(random.sample(self.patch_ids, per_worker))