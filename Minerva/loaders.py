"""Module containing classes defining custom (Iterable)Dataset classes for use in the fitting of neural networks.

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

Created under a project funded by the Ordnance Survey Ltd.

Attributes:
    manifest (pd.DataFrame): DataFrame outlining every sample in the dataset's cloud cover, centre pixel label
        and fraction class sizes.

TODO:
    * Re-incorporate the use of WeightedRandomSampler as an option
    * Further reduce boilerplate of datasets by introducing abstract classes
    * Incorporate config updating into make_datasets
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Optional, Union, Tuple, Iterator, Dict
from Minerva.utils import utils
import random
from abc import ABC
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader  # , WeightedRandomSampler
from torchvision import transforms
from itertools import cycle, chain
from collections import deque

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
manifest = pd.read_csv(utils.get_manifest())


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
        no_empty_classes (bool): Informs dataset if empty classes have been removed from
            the class label schema. If True, labels will be converted to the new schema using forwards.
        forwards (dict): Mapping of original class labelling schema to new.
        wheels (dict[deque]): Dict of `wheels' (deques) holding a stream of pixel stacks organised by class in memory.

    Args:
        class_streams (pd.DataFrame): DataFrame with a column of patch IDs for each class.
        batch_size (int): Optional; Sets the number of samples in each batch.
        wheel_size (int): Optional; Sets the maximum size of the wheels (deque) holding pixel stacks in memory.
        patch_len (int): Optional; Total number of pixels in a patch.
        no_empty_classes (bool): Optional; Informs dataset if empty classes have been removed from
            the class label schema. If True, labels will be converted to the new schema using forwards.
        forwards (dict): Optional; Mapping of original class labelling schema to new.
    """

    def __init__(self, class_streams: pd.DataFrame, batch_size: int = 32, wheel_size: int = 65536,
                 patch_len: int = 65536, no_empty_classes: bool = True, forwards: Optional[dict] = None) -> None:

        self.streams_df = class_streams
        self.batch_size = batch_size
        self.patch_len = patch_len
        self.no_empty_classes = no_empty_classes
        self.forwards = forwards

        # Dict to hold a `wheel' for each class
        self.wheels = {}

        # Initialise each wheel with a maximum length of wheel_size global parameter
        for cls in self.streams_df.columns.to_list():
            self.wheels[cls] = deque(maxlen=wheel_size)

        # Loads the patches from the row of IDs supplied into a pandas.Series of pandas.DataFrames
        patch_dfs = [utils.load_patch_df(patch_id, manifest) for patch_id in self.streams_df.sample(frac=1).iloc[0]]
        patches = pd.Series(patch_dfs)
        patches.apply(self.refresh_wheels)

        # Checks if wheel is empty after adding to wheel
        # If wheel is empty, emergency_fill is activated to make sure there is at least an entry in this class' wheel
        for cls in self.streams_df.columns.to_list():
            if not self.wheels[cls]:
                self.__emergency_fill__(cls)

    def load_patches(self, row: pd.Series) -> pd.Series:
        """Loads the patches associated with the patch IDs in row.

        Patches are loaded as pandas.DataFrames nested into a pandas.Series object.

        Args:
            row (pd.Series): A row of patch IDs

        Returns:
            Series of DataFrames of patches.
        """
        return pd.Series([utils.load_patch_df(row[1][cls], manifest) for cls in self.streams_df.columns.to_list()])

    def refresh_wheels(self, patch_df: pd.DataFrame) -> None:
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
        patch_dfs = [utils.load_patch_df(patch_id, manifest) for patch_id in self.streams_df[cls].sample(frac=1)]
        patches = pd.Series(patch_dfs)

        for patch in patches:
            if self.wheels[cls]:
                break
            for pixel in patch['PATCH'].loc[patch['LABELS'] == cls]:
                self.wheels[cls].appendleft(pixel.flatten())

    def process_data(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Loads and processes patches into wheels for each class and yields from them,
        periodically refreshing the wheels with new data.

        Args:
            row (pd.Series): Randomly selected row of patch IDs, one for each class.

        Yields:
            x (torch.Tensor): A data sample as tensor.
            y (torch.Tensor): Corresponding label as int tensor.
            Empty string (for compatibility reasons).
        """
        # Iterates for the flattened length of a patch and yields x and y for each class from their respective wheels.
        for i in range(self.patch_len):
            if i == 0:
                # Loads the patches from the row of IDs supplied into a pandas.Series of pandas.DataFrames.
                patches = self.load_patches(row)
                patches.apply(self.refresh_wheels)

            # For every class in the dataset, rotate the corresponding wheel
            # and yield the pixel stack from position [0].
            for cls in self.streams_df.columns.to_list():
                # Rotate current class's wheel 1 turn
                self.wheels[cls].rotate(1)

                if self.no_empty_classes:
                    # Yield pixel stack at position [0] for this class's wheel and the corresponding class label.
                    # i.e this class number as a tensor int.
                    yield torch.tensor(self.wheels[cls][0].flatten(), dtype=torch.float), \
                          torch.tensor(utils.class_transform(cls, self.forwards), dtype=torch.long), ''
                else:
                    # Yield pixel stack at position [0] for this class's wheel and the corresponding class label.
                    # i.e this class number as a tensor int.
                    yield torch.tensor(self.wheels[cls][0].flatten(), dtype=torch.float), \
                        torch.tensor(cls, dtype=torch.long), ''

    def _get_stream(self, streams_df) -> Iterator[Union[torch.Tensor, str]]:
        return chain.from_iterable(map(self.process_data, streams_df.iterrows()))

    def __iter__(self) -> Iterator[Union[torch.Tensor, str]]:
        # Gets the current worker info
        worker_info = torch.utils.data.get_worker_info()

        # If single threaded process, return full ID stream
        if worker_info is None:
            return self._get_stream(self.streams_df)

        # If multi-threaded, split patch IDs between workers
        else:
            # Calculate fraction of dataset per worker
            per_worker = int(np.math.ceil(1.0 / float(worker_info.num_workers)))

            # Return a random sample of the patch IDs of fractional size per worker
            # and using random seed modulated by the worker ID.
            return self._get_stream(self.streams_df.sample(frac=per_worker, random_state=42 * worker_info.id,
                                                           replace=False, axis=0))


class BatchDataset(IterableDataset, ABC):
    """Adaptation of IterableDataset to work with pixel stacks.

    Engineered to pre-process Landcovernet image data into multi-band, time-series pixel stacks
    and yield to a DataLoader.

    Attributes:
        patch_ids (list[str]): List of patch IDs representing the outline of this dataset.
        batch_size (int): Number of samples returned in each batch.
        no_empty_classes (bool): Informs dataset if empty classes have been removed from
            the class label schema. If True, labels will be converted to the new schema using forwards.
        forwards (dict): Mapping of original class labelling schema to new.

    Args:
        patch_ids (list[str]): List of patch IDs representing the outline of this dataset.
        batch_size (int): Number of samples returned in each batch.
        no_empty_classes (bool): Optional; Informs dataset if empty classes have been removed from
            the class label schema. If True, labels will be converted to the new schema using forwards.
        forwards (dict): Optional; Mapping of original class labelling schema to new.
    """

    def __init__(self, patch_ids: list, batch_size: int, no_empty_classes: bool = True,
                 forwards: Optional[dict] = None) -> None:

        self.patch_ids = patch_ids
        self.batch_size = batch_size
        self.no_empty_classes = no_empty_classes
        self.forwards = forwards

    def process_data(self, patch_id: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Loads pixel-stacks and yields samples and labels from them.

        Args:
            patch_id (str): Unique ID of a patch of the dataset.

        Yields:
            Pixel-stack, associated label and the patch ID where they came from.
        """
        patch = utils.make_time_series(patch_id, manifest)

        x = torch.tensor([pixel.flatten() for pixel in patch.reshape(-1, *patch.shape[-2:])], dtype=torch.float)
        y = utils.lc_load(patch_id)

        if self.no_empty_classes:
            y = utils.mask_transform(y, self.forwards)
        y = torch.from_numpy(y.flatten().astype(int))
        y = y.to(torch.long)

        for i in range(len(y)):
            yield x[i], y[i], patch_id

    def _get_stream(self, patch_ids) -> Iterator[Union[torch.Tensor, str]]:
        return chain.from_iterable(map(self.process_data, cycle(patch_ids)))

    def __iter__(self) -> Iterator[Union[torch.Tensor, str]]:
        # Gets the current worker info
        worker_info = torch.utils.data.get_worker_info()

        # If single threaded process, return all patch IDs
        if worker_info is None:
            return self._get_stream(self.patch_ids)

        # If multi-threaded, split patch IDs between workers
        else:
            # Calculate number of patch IDs of the dataset per worker
            per_worker = int(np.math.ceil(len(self.patch_ids) / float(worker_info.num_workers)))

            # Set random seed modulated by the worker ID
            random.seed(42 * worker_info.id)

            # Return a random sample of the patch IDs of size per worker
            return self._get_stream(random.sample(self.patch_ids, per_worker))


class IterableImageDataset(IterableDataset, ABC):
    """Adaptation of IterableDataset for handling images for use with a CNN.

    Engineered to pre-process Landcovernet image data into stacked multi-band images
    and yield to a DataLoader with associated labels.

    Attributes:
        patch_ids (list[str]): List of patch IDs representing the outline of this dataset.
        batch_size (int): Number of samples returned in each batch.
        elim (bool): Informs dataset if empty classes have been removed from
            the class label schema. If True, labels will be converted to the new schema using forwards.
        forwards (dict): Mapping of original class labelling schema to new.

    Args:
        patch_ids (list[str]): List of patch IDs representing the outline of this dataset.
        batch_size (int): Number of samples returned in each batch.
        elim (bool): Optional; Informs dataset if empty classes have been removed from
            the class label schema. If True, labels will be converted to the new schema using forwards.
        forwards (dict): Optional; Mapping of original class labelling schema to new.
    """

    def __init__(self, patch_ids: list, batch_size: int, elim: bool = True, forwards: Optional[dict] = None) -> None:

        self.patch_ids = patch_ids
        self.batch_size = batch_size
        self.elim = elim
        self.forwards = forwards

    def process_data(self, patch_id: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Loads scenes from given patch and yields sample images and labels from them.

        Args:
            patch_id (str): Unique ID of a patch of the dataset.

        Yields:
            Pixel-stack, associated label and the patch ID where they came from.
        """

        y = utils.find_centre_label(patch_id)
        images = [utils.stack_bands(patch_id, scene) for scene in utils.find_best_of(patch_id, manifest)]

        if self.elim:
            y = torch.tensor(utils.class_transform(y, self.forwards), dtype=torch.long)
        if not self.elim:
            y = torch.tensor(y, dtype=torch.long)

        for image in images:
            yield torch.tensor(image.reshape((image.shape[2], image.shape[1], image.shape[0])), dtype=torch.float), \
                  y, patch_id

    def _get_stream(self, patch_ids: list) -> Iterator[Union[torch.Tensor, str]]:
        return chain.from_iterable(map(self.process_data, cycle(patch_ids)))

    def __iter__(self) -> Iterator[Union[torch.Tensor, str]]:
        # Gets the current worker info
        worker_info = torch.utils.data.get_worker_info()

        # If single threaded process, return all patch IDs
        if worker_info is None:
            return self._get_stream(self.patch_ids)

        # If multi-threaded, split patch IDs between workers
        else:
            # Calculate number of patch IDs of the dataset per worker
            per_worker = int(np.math.ceil(len(self.patch_ids) / float(worker_info.num_workers)))

            # Set random seed modulated by the worker ID
            random.seed(42 * worker_info.id)

            # Return a random sample of the patch IDs of size per worker
            return self._get_stream(random.sample(self.patch_ids, per_worker))


class ImageDataset(Dataset, ABC):
    """Adaptation of Dataset for handling images for use with a CNN.

    Engineered to pre-process Landcovernet image data into stacked multi-band images
    and return to a DataLoader with associated labels.

    Attributes:
        scenes (list[tuple[str, str]): List of tuples of pairs of patch ID and scene date, representing the outline of
            this dataset.
        batch_size (int): Number of samples returned in each batch.
        no_empty_classes (bool): Informs dataset if empty classes have been removed from
            the class label schema. If True, labels will be converted to the new schema using forwards.
        forwards (dict): Mapping of original class labelling schema to new.

    Args:
        scenes (list[tuple[str, str]): List of tuples of pairs of patch ID and scene date, representing the outline
            of this dataset.
        batch_size (int): Number of samples returned in each batch.
        no_empty_classes (bool): Optional; Informs dataset if empty classes have been removed from
            the class label schema. If True, labels will be converted to the new schema using forwards.
        forwards (dict): Optional; Mapping of original class labelling schema to new.
    """

    def __init__(self, scenes: list, batch_size: int, model_type: str = 'CNN', no_empty_classes: bool = True,
                 centre_only: bool = False, forwards: Optional[dict] = None, transformations=None) -> None:

        self.scenes = scenes
        self.batch_size = batch_size
        self.model_type = model_type
        self.no_empty_classes = no_empty_classes
        self.centre_only = centre_only
        self.forwards = forwards
        self.transformations = transformations

    def __len__(self) -> int:
        return len(self.scenes)

    def process_data(self, scene: tuple) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Loads scenes from given patch and returns sample images and labels from them.

        Args:
            scene (tuple[str, str]): Tuple of Unique patch ID and date of scene within the dataset.

        Returns:
            Multi-spectral image of scene, associated label and the patch ID where they came from.
        """
        patch_id, date = scene

        func = utils.find_centre_label
        if self.model_type == 'segmentation':
            func = utils.lc_load

        y = func(patch_id)
        image = utils.stack_bands(patch_id, date)

        if self.centre_only:
            image = utils.centre_pixel_only(image)

        if self.model_type == 'segmentation':
            if self.no_empty_classes:
                y = utils.mask_transform(y, self.forwards)
            y = torch.from_numpy(y.astype(int))
            y = y.to(torch.long)

        else:
            if self.no_empty_classes:
                y = torch.tensor(utils.class_transform(y, self.forwards), dtype=torch.long)
            if not self.no_empty_classes:
                y = torch.tensor(y, dtype=torch.long)

        sample = torch.tensor(image.reshape((image.shape[2], image.shape[1], image.shape[0])), dtype=torch.float)

        if self.transformations:
            sample = self.transformations(sample)

        return sample, y, '{}-{}'.format(patch_id, date)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return self.process_data(self.scenes[idx])


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_transform(name: str, params: dict):
    """Creates a TensorBoard transform object based on config parameters.

    Returns:
        Initialised TensorBoard transform object specified by config parameters.
    """
    # Gets the loss function requested by config parameters.
    transform = utils.func_by_str('torchvision.transforms', name)

    return transform(**params)


def make_transformations(transform_params: dict):
    """Constructs a transform or series of transforms based on parameters provided.

    Args:
        transform_params (dict): Parameters defining transforms desired. The name of each transform should be the key,
            while the kwargs for the transform should be the value of that key as a dict.

            e.g. {CenterCrop: {size: 128}}

    Returns:
        If no parameters are parsed, None is returned. If only one transform is defined by the parameters,
            returns a Transforms object. If multiple transforms are defined, a Compose object of Transform
            objects is returned
    """
    # If no transforms are specified, return None.
    if not transform_params:
        return None

    transformations = []

    # Get each transform.
    for name in transform_params:
        transform = get_transform(name, transform_params[name])

        # If only one transform found, return.
        if len(transform_params) is 1:
            return transform

        # If more than one transform found, append to list for composition.
        else:
            transformations.append(transform)

    # Compose transforms together and return.
    return transforms.Compose(transformations)


def make_datasets(patch_ids: Optional[list] = None, frac: Optional[float] = None, n_patches: Optional[int] = None,
                  split: Tuple[float, float, float] = (0.7, 0.15, 0.15), wheel_size: int = 65536,
                  n_pixels: int = 65536, seed: int = 42, shuffle: bool = True, plot: bool = False,
                  balance: bool = False, over_factor: int = 1, model_type: str = 'scene classifier',
                  p_dist: bool = False, **params) -> Tuple[Dict[str, DataLoader], dict, list, dict, dict, dict]:
    """Constructs train, validation and test datasets and places in DataLoaders for use in model fitting and testing.

    Args:
        patch_ids (list[str]): Optional; List of patch IDs that outline the whole dataset to be used. If not provided,
            the patch IDs are inferred from the directory using patch_grab.
        frac (float): Optional; Fraction of the all patch IDs to include in the construction of the datasets.
        n_patches (float): Optional; The number of patches to use in the construction of datasets.
        split (list[float] or tuple[float]): Optional; Three values giving the fractional sizes of the datasets, in the
            order (train, validation, test).
        wheel_size (int): Optional; Length of each `wheel' to be used in class balancing sampling in IterableDataset.
            This is essentially the number of pixel stacks per class to have queued at any one time.
        n_pixels (int): Optional; Total number of pixels in each sample (per band).
        seed (int): Optional; Random seed number to fix the shuffling of the data split.
        shuffle (bool): Optional; Whether to shuffle the patch IDs in the splitting of the IDs.
        plot (bool): Optional; Whether or not to plot pie charts of the class distributions within each dataset.
        balance (bool): Optional; Whether or not to attempt to balance the class distributions within each dataset.
        over_factor (int): Optional; On average, the maximum number of times the same sample will occur in a dataset.
            Will be at the maximum value for the smallest class being over-sampled.
        model_type (str): Optional; Must be either mlp, MLP, scene classifier or segmentation.
        p_dist (bool): Optional; Whether to print to screen the distribution of classes within each dataset.

    Keyword Args:
        hyperparams (dict): Dictionary of hyper-parameters for the model.
        batch_size (int): Number of samples in each batch to be returned by the DataLoaders.
        scene_selector (str): Name of function to use to select which scenes of a patch to include in the datasets.
        elim (bool): Whether or not to eliminate classes with no samples in.
        centre_only (bool): Whether to modify samples to be an array of zeros apart from the original centre pixel.

    Returns:
        loaders (dict): Dictionary of the DataLoader for training, validation and testing.
        n_batches (dict): Dictionary of the number of batches to return/ yield in each train, validation and test epoch.
        class_dist (list): The class distribution of the entire dataset, sorted from largest to smallest class.
        ids (dict): Dictionary of the IDs (patch or scene) defining each dataset and the entire dataset.
        new_classes (dict): Dictionary mapping class labels to class names - modified to remove empty classes.
        new_colours (dict): Dictionary mapping class labels to colours - modified to remove empty classes.
    """
    # Gets out the parameters for the DataLoaders from params.
    dataloader_params = params['hyperparams']['params']
    batch_size = dataloader_params['batch_size']

    # Defines the function to use to load the labels. Either to load the whole mask or just the centre label.
    label_func = utils.lc_load
    if model_type == 'scene classifier':
        label_func = utils.find_centre_label

    # Gets the patch IDs based on options supplied if they were not supplied.
    if frac is not None or n_patches is not None and patch_ids is None:
        patch_ids = utils.patch_grab()
        if frac is not None and n_patches is None:
            n_patches = int(frac * len(patch_ids))
        patch_ids = [patch_ids[i] for i in random.sample(range(len(patch_ids)), n_patches)]

    print('\n# OF PATCHES IN DATASET: {}'.format(len(patch_ids)))

    # Splits the dataset into train, validation and test.
    print('\nSPLITTING DATASET TO {}% TRAIN, {}% VAL, {}% TEST'.format(split[0] * 100, split[1] * 100, split[2] * 100))
    ids, patch_class_dists = utils.split_data(patch_ids=patch_ids, split=split, func=label_func, seed=seed,
                                              shuffle=shuffle, balance=False, p_dist=p_dist, plot=plot)

    # Finds the empty classes and returns modified classes, a dict to convert between the old and new systems
    # and new colours.
    print('\nFINDING EMPTY CLASSES')
    new_classes, forwards, new_colours = utils.eliminate_classes(
        utils.find_empty_classes(class_dist=patch_class_dists['ALL']))

    # Sets the function to select which scenes to include in the dataset based on the option in the params.
    scene_func = utils.ref_scene_select
    if params['scene_selector'] == 'threshold':
        scene_func = utils.threshold_scene_select

    # Inits dicts to hold the variables and lists for train, validation and test.
    scenes = {}
    datasets = {}
    n_batches = {}
    loaders = {}
    class_dists = {}

    for mode in ('train', 'val', 'test'):
        # Finds scenes using the scene selector and manifest.
        print('\nFINDING {} SCENES'.format(mode))
        if model_type == 'scene classifier' and balance:
            # Balances scenes returned using over and under sampling.
            scenes[mode] = utils.hard_balance(utils.scene_extract(ids[mode], manifest, scene_func),
                                              over_factor=over_factor)
        else:
            # No balancing of scenes.
            scenes[mode] = utils.scene_extract(ids[mode], manifest, scene_func, thres=0.1)

        print('\nFINDING CLASS DISTRIBUTION OF SCENES')
        # Find class distribution of dataset by scene IDs, not patch IDs.
        class_dist = utils.subpopulations_from_manifest(utils.select_df_by_scenes(manifest, scenes[mode]),
                                                        func=label_func, plot=plot)

        # Prints class distribution in a pretty text format using tabulate to stdout.
        if p_dist:
            utils.print_class_dist(class_dist)

        # Transform class dist if elimination of classes has occurred.
        if params['elim']:
            class_dist = utils.class_dist_transform(class_dist, forwards)
        class_dists[mode] = class_dist

        # --+ MAKE DATASETS +=========================================================================================+
        # Uses BalancedBatchDataset for creating balanced datasets for use with MLPs.
        if balance and model_type in ['mlp', 'MLP'] and mode != 'test':
            stream = utils.make_sorted_streams(ids[mode])

            # Define datasets for train and validation using BatchDataset.
            datasets[mode] = BalancedBatchDataset(stream, batch_size=batch_size, wheel_size=wheel_size,
                                                  patch_len=n_pixels, no_empty_classes=params['elim'],
                                                  forwards=forwards)

            # Calculates number of batches.
            n_batches[mode] = utils.num_batches(len(stream.columns) * len(stream))

        # Uses BatchDataset for creating unbalanced datasets for use with MLPs.
        if (not balance or (balance and mode == 'test')) and model_type in ['mlp', 'MLP']:
            # Define datasets for train, validation and test using BatchDataset.
            datasets[mode] = BatchDataset(ids[mode], batch_size=batch_size, no_empty_classes=params['elim'],
                                          forwards=forwards)

            # Calculates number of batches.
            n_batches[mode] = utils.num_batches(len(ids[mode]))

        # For non-MLP models, use ImageDataset datasets to handle scene level samples.
        if model_type in ['scene classifier', 'segmentation']:
            # Creates transformations from those defined in params.
            transformations = make_transformations(params['hyperparams']['transforms'])

            # Define datasets for train, validation and test using ImageDataset.
            datasets[mode] = ImageDataset(scenes[mode], batch_size=batch_size, no_empty_classes=params['elim'],
                                          centre_only=params['centre_only'], model_type=model_type,
                                          forwards=forwards, transformations=transformations)

            # Calculates number of batches.
            n_batches[mode] = int(len(scenes[mode]) / batch_size)

        # --+ MAKE DATALOADERS +======================================================================================+
        if model_type == 'scene classifier' and balance:
            loaders[mode] = DataLoader(datasets[mode], **dataloader_params)
        #    weights = utils.weight_samples(scenes[mode], func=utils.find_centre_label, normalise=False)
        #    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.float), len(weights), replacement=True)
        #    loaders[mode] = DataLoader(datasets[mode], **dataloader_params, sampler=sampler)

        if model_type == 'scene classifier' and not balance or model_type != 'scene classifier':
            loaders[mode] = DataLoader(datasets[mode], **dataloader_params)

    # Combines all scenes together to output a class_dist for the entire dataset.
    all_scenes = scenes['train'] + scenes['val'] + scenes['test']
    class_dist = utils.subpopulations_from_manifest(utils.select_df_by_scenes(manifest, all_scenes),
                                                    func=label_func, plot=plot)

    # Transform class dist if elimination of classes has occurred.
    if params['elim']:
        class_dist = utils.class_dist_transform(class_dist, forwards)

    # Prints class distribution in a pretty text format using tabulate to stdout.
    if p_dist:
        utils.print_class_dist(class_dist)

    return loaders, n_batches, class_dist, ids, new_classes, new_colours
