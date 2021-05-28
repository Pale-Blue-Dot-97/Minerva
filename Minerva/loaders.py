"""Module containing classes defining custom IterableDataset classes for use in the fitting of neural networks.

    Copyright (C) 2021 Harry James Baker

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program in LICENSE.txt. If not,
    see <https://www.gnu.org/licenses/>.

Author: Harry James Baker

Email: hjb1d20@soton.ac.uk or hjbaker97@gmail.com

Institution: University of Southampton

Created under a project funded by the Ordnance Survey Ltd

TODO:
    * Fully document
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.utils import utils
import random
from abc import ABC
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from itertools import cycle, chain
from collections import deque


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class BalancedBatchDataset(IterableDataset, ABC):
    """Adaptation of BatchDataset to load data with perfect class balance.

    Engineered to work with Landcovernet data and to overcome class imbalance.

    Calibrated to work with a MLP by pre-processing image data into multi-band, time-series pixel stacks.

    Use with a torch.utils.data.DataLoader or more ideally use with Minerva.trainer.Trainer.

    Attributes:
        streams_df (pandas.DataFrame): DataFrame with a column of patch IDs for each class.
        batch_size (int): Sets the number of samples in each batch.
        patch_len (int): Total number of pixels in a patch.
        wheels (dict[deque]): Dict of `wheels' (deques) holding a stream of pixel stacks organised by class in memory.

    """

    def __init__(self, class_streams: pd.DataFrame, batch_size: int = 32, wheel_size: int = 65536,
                 patch_len: int = 65536):
        """Inits BalancedBatchDataset

        Args:
            class_streams (pd.DataFrame): DataFrame with a column of patch IDs for each class.
            batch_size (int): Optional; Sets the number of samples in each batch.
            wheel_size (int): Optional; Sets the maximum size of the wheels (deque) holding pixel stacks in memory.
            patch_len (int): Optional; Total number of pixels in a patch.
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
        patches = pd.Series([utils.load_patch_df(patch_id) for patch_id in self.streams_df.sample(frac=1).iloc[0]])
        patches.apply(self.refresh_wheels)

        # Checks if wheel is empty after adding to wheel
        # If wheel is empty, emergency_fill is activated to make sure there is at least an entry in this class' wheel
        for cls in self.streams_df.columns.to_list():
            if not self.wheels[cls]:
                self.__emergency_fill__(cls)

    def load_patches(self, row: pd.Series):
        """Loads the patches associated with the patch IDs in row.

        Patches are loaded as pandas.DataFrames nested into a pandas.Series object.

        Args:
            row (pd.Series): A row of patch IDs

        Returns:
            Series of DataFrames of patches.
        """
        return pd.Series([utils.load_patch_df(row[1][cls]) for cls in self.streams_df.columns.to_list()])

    def refresh_wheels(self, patch_df: pd.DataFrame):
        """Updates the values in each wheel from the supplied DataFrame.

        Takes the DataFrame representing a patch supplied and for each class, finds any pixels matching that class
        and adds to the wheel of the same class.

        Args:
            patch_df (pd.DataFrame): DataFrame representing a patch of data.
        """
        for cls in self.streams_df.columns.to_list():
            for pixel in patch_df['PATCH'].loc[patch_df['LABELS'] == cls]:
                self.wheels[cls].appendleft(pixel.flatten())

    def __emergency_fill__(self, cls) -> None:
        """Attempts to ensure at least one value is in the wheel of class `cls'.

        Emergency method that loads all patches in the class stream for cls. A patch is loaded at a time,
        looking for any pixels corresponding to cls class until the wheel contains values.

        Args:
            cls: Class number with empty wheel.
        """
        patches = pd.Series([utils.load_patch_df(patch_id) for patch_id in self.streams_df[cls].sample(frac=1)])

        for patch in patches:
            if self.wheels[cls]:
                break
            for pixel in patch['PATCH'].loc[patch['LABELS'] == cls]:
                self.wheels[cls].appendleft(pixel.flatten())

    def process_data(self, row: pd.Series):
        """Loads and processes patches into wheels for each class and yields from them,
        periodically refreshing the wheels with new data.

        Args:
            row (pd.Series): Randomly selected row of patch IDs, one for each class.

        Yields:
            x (torch.Tensor): A data sample as tensor.
            y (torch.Tensor): Corresponding label as int tensor.
            Empty string (for compatibility reasons).
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
                    torch.tensor(cls, dtype=torch.long), ''

    def get_stream(self, streams_df):
        return chain.from_iterable(map(self.process_data, streams_df.iterrows()))

    def __iter__(self):
        # Gets the current worker info
        worker_info = torch.utils.data.get_worker_info()

        # If single threaded process, return full ID stream
        if worker_info is None:
            return self.get_stream(self.streams_df)

        # If multi-threaded, split patch IDs between workers
        else:
            # Calculate fraction of dataset per worker
            per_worker = int(np.math.ceil(1.0 / float(worker_info.num_workers)))

            # Return a random sample of the patch IDs of fractional size per worker
            # and using random seed modulated by the worker ID.
            return self.get_stream(self.streams_df.sample(frac=per_worker, random_state=42 * worker_info.id,
                                                          replace=False, axis=0))


class BatchDataset(IterableDataset, ABC):
    """Adaptation of IterableDataset to work with pixel stacks.

    Engineered to pre-process Landcovernet image data into multi-band, time-series pixel stacks
    and yield to a DataLoader.

    Attributes:
        patch_ids (list[str]): List of patch IDs representing the outline of this dataset.
        batch_size (int): Number of samples returned in each batch.
    """

    def __init__(self, patch_ids, batch_size: int):
        """Inits BatchDataset

        Args:
            patch_ids (list[str]): List of patch IDs representing the outline of this dataset.
            batch_size (int): Number of samples returned in each batch.
        """
        self.patch_ids = patch_ids
        self.batch_size = batch_size

    def process_data(self, patch_id: str):
        """Loads pixel-stacks and yields samples and labels from them.

        Args:
            patch_id (str): Unique ID of a patch of the dataset.

        Yields:
            Pixel-stack, associated label and the patch ID where they came from.
        """
        patch = utils.make_time_series(patch_id)

        x = torch.tensor([pixel.flatten() for pixel in patch.reshape(-1, *patch.shape[-2:])], dtype=torch.float)
        y = torch.tensor(np.array(utils.lc_load(patch_id), dtype=np.int64).flatten(), dtype=torch.long)

        for i in range(len(y)):
            yield x[i], y[i], patch_id

    def get_stream(self, patch_ids):
        return chain.from_iterable(map(self.process_data, cycle(patch_ids)))

    def __iter__(self):
        # Gets the current worker info
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


class IterableImageDataset(IterableDataset, ABC):
    """Adaptation of IterableDataset for handling images for use with a CNN.

    Engineered to pre-process Landcovernet image data into multi-band, time-series pixel stacks
    and yield to a DataLoader.

    Attributes:
        patch_ids (list[str]): List of patch IDs representing the outline of this dataset.
        batch_size (int): Number of samples returned in each batch.
    """

    def __init__(self, patch_ids, batch_size: int, elim: bool = True, forwards=None):
        """Inits BatchDataset

        Args:
            patch_ids (list[str]): List of patch IDs representing the outline of this dataset.
            batch_size (int): Number of samples returned in each batch.
        """
        self.patch_ids = patch_ids
        self.batch_size = batch_size
        self.elim = elim
        self.forwards = forwards

    def process_data(self, patch_id: str):
        """Loads scenes from given patch and yields sample images and labels from them.

        Args:
            patch_id (str): Unique ID of a patch of the dataset.

        Yields:
            Pixel-stack, associated label and the patch ID where they came from.
        """

        y = utils.find_centre_label(patch_id)
        images = [utils.stack_bands(patch_id, scene) for scene in utils.find_best_of(patch_id)]

        if self.elim:
            y = torch.tensor(utils.class_transform(y, self.forwards), dtype=torch.long)
        if not self.elim:
            y = torch.tensor(y, dtype=torch.long)

        for image in images:
            yield torch.tensor(image.reshape((image.shape[2], image.shape[1], image.shape[0])), dtype=torch.float), \
                  y, patch_id

    def get_stream(self, patch_ids):
        return chain.from_iterable(map(self.process_data, cycle(patch_ids)))

    def __iter__(self):
        # Gets the current worker info
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


class ImageDataset(Dataset, ABC):
    """Adaptation of Dataset for handling images for use with a CNN.

    Engineered to pre-process Landcovernet image data into multi-band, time-series pixel stacks
    and yield to a DataLoader.

    Attributes:
        scenes (list[tuple[str, str]): List of tuples of pairs of patch ID and scene date, representing the outline of
            this dataset.
        batch_size (int): Number of samples returned in each batch.
    """

    def __init__(self, scenes, batch_size: int, no_empty_classes: bool = True, forwards=None, transformations=None):
        """Inits ImageDataset

        Args:
            scenes (list[tuple[str, str]): List of tuples of pairs of patch ID and scene date, representing the outline
                of this dataset.
            batch_size (int): Number of samples returned in each batch.
        """
        self.scenes = scenes
        self.batch_size = batch_size
        self.no_empty_classes = no_empty_classes
        self.forwards = forwards
        self.transformations = transformations

    def __len__(self):
        return len(self.scenes)

    def process_data(self, scene: tuple):
        """Loads scenes from given patch and yields sample images and labels from them.

        Args:
            scene (tuple[str, str]): Tuple of Unique patch ID and date of scene within the dataset.

        Yields:
            Multi-spectral image of scene, associated label and the patch ID where they came from.
        """
        patch_id, date = scene

        y = utils.find_centre_label(patch_id)
        image = utils.stack_bands(patch_id, date)

        if self.no_empty_classes:
            y = torch.tensor(utils.class_transform(y, self.forwards), dtype=torch.long)
        if not self.no_empty_classes:
            y = torch.tensor(y, dtype=torch.long)

        sample = torch.tensor(image.reshape((image.shape[2], image.shape[1], image.shape[0])), dtype=torch.float)

        if self.transformations:
            sample = self.transformations(sample)

        return sample, y, patch_id

    def __getitem__(self, idx):
        return self.process_data(self.scenes[idx])


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def make_datasets(patch_ids=None, split=(0.7, 0.15, 0.15), params=None, wheel_size=65536, image_len=65536, seed=42,
                  shuffle=True, plot=False, balance=False, cnn=False, p_dist=False):
    """

    Args:
        patch_ids (list[str]): Optional; List of patch IDs that outline the whole dataset to be used. If not provided,
            the patch IDs are inferred from the directory using patch_grab.
        split (list[float] or tuple[float]): Optional; Three values giving the fractional sizes of the datasets, in the
            order (train, validation, test).
        params:
        wheel_size:
        image_len:
        seed (int): Optional; Random seed number to fix the shuffling of the data split.
        shuffle (bool): Optional; Whether to shuffle the patch IDs in the splitting of the IDs.
        plot (bool): Optional; Whether or not to plot pie charts of the class distributions within each dataset.
        balance (bool):
        cnn (bool):
        p_dist (bool): Optional; Whether to print to screen the distribution of classes within each dataset.

    Returns:
        loaders (dict):
        n_batches (dict):
        class_dist (Counter):
    """

    ids = utils.split_data(patch_ids=patch_ids, split=split, seed=seed, shuffle=shuffle, p_dist=p_dist, plot=plot,
                           ctr_lbl=cnn)

    new_classes, forwards, backwards = utils.eliminate_classes(utils.find_empty_classes(ids['train'],
                                                                                        utils.find_centre_label))

    scenes = {'train': utils.scene_extract(ids['train'], utils.find_best_of),
              'val': utils.scene_extract(ids['val'], utils.find_best_of),
              'test': utils.scene_extract(ids['test'], utils.find_best_of)}
    datasets = {}
    n_batches = {}

    if balance and not cnn:
        train_stream = utils.make_sorted_streams(ids['train'])
        val_stream = utils.make_sorted_streams(ids['val'])

        # Define datasets for train, validation and test using BatchDataset
        datasets['train'] = BalancedBatchDataset(train_stream, batch_size=params['batch_size'],
                                                 wheel_size=wheel_size, patch_len=image_len)
        datasets['val'] = BalancedBatchDataset(val_stream, batch_size=params['batch_size'],
                                               wheel_size=wheel_size, patch_len=image_len)
        datasets['test'] = BatchDataset(ids['test'], batch_size=params['batch_size'])

        n_batches['train'] = utils.num_batches(len(train_stream.columns) * len(train_stream))
        n_batches['val'] = utils.num_batches(len(val_stream.columns) * len(val_stream))
        n_batches['test'] = utils.num_batches(len(ids['test']))

    if not balance and not cnn:
        # Define datasets for train, validation and test using BatchDataset
        datasets['train'] = BatchDataset(ids['train'], batch_size=params['batch_size'])
        datasets['val'] = BatchDataset(ids['val'], batch_size=params['batch_size'])
        datasets['test'] = BatchDataset(ids['test'], batch_size=params['batch_size'])

        n_batches['train'] = utils.num_batches(len(ids['train']))
        n_batches['val'] = utils.num_batches(len(ids['val']))
        n_batches['test'] = utils.num_batches(len(ids['test']))

    if cnn:
        # Define datasets for train, validation and test using ImageDataset
        datasets['train'] = ImageDataset(scenes['train'], batch_size=params['batch_size'],
                                         no_empty_classes=True, forwards=forwards,
                                         transformations=transforms.CenterCrop(128))
        datasets['val'] = ImageDataset(scenes['val'], batch_size=params['batch_size'],
                                       no_empty_classes=True, forwards=forwards,
                                       transformations=transforms.CenterCrop(128))
        datasets['test'] = ImageDataset(scenes['test'], batch_size=params['batch_size'],
                                        no_empty_classes=True, forwards=forwards,
                                        transformations=transforms.CenterCrop(128))

        n_batches['train'] = int((len(ids['train']) * 24.0) / params['batch_size'])
        n_batches['val'] = int((len(ids['val']) * 24.0) / params['batch_size'])
        n_batches['test'] = int((len(ids['test']) * 24.0) / params['batch_size'])

    loaders = {}

    if cnn and balance:
        train_weights = utils.weight_samples(ids['train'], func=utils.find_centre_label)
        val_weights = utils.weight_samples(ids['val'], func=utils.find_centre_label)

        train_weighted_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
        val_weighted_sampler = WeightedRandomSampler(val_weights, len(val_weights), replacement=True)

        loaders['train'] = DataLoader(datasets['train'], **params, sampler=train_weighted_sampler)
        loaders['val'] = DataLoader(datasets['val'], **params, sampler=val_weighted_sampler)

    if cnn and not balance or not cnn:
        loaders['train'] = DataLoader(datasets['train'], **params)
        loaders['val'] = DataLoader(datasets['val'], **params)

    loaders['test'] = DataLoader(datasets['test'], **params)

    class_dist = utils.find_subpopulations(patch_ids, plot=False)

    return loaders, n_batches, class_dist, ids, new_classes, backwards
