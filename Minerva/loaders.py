"""Module for constructing loaders, samplers, datasets and transforms using torchvision-style structures.

    Copyright (C) 2022 Harry James Baker

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

TODO:
    * Re-incorporate the use of WeightedRandomSampler as an option
    * Re-incorporate class distribution calculations
    * Consider incorporation into utils or trainer
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
from typing import Optional, Union, Tuple, Dict, Iterable
import pandas as pd
from Minerva.utils import utils
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from alive_progress import alive_it
from torchgeo.datasets.utils import BoundingBox


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def intersect_datasets(datasets: list):
    def intersect_pair_datasets(a, b):
        return a & b

    for i in range(len(datasets) - 1):
        datasets[0] = intersect_pair_datasets(datasets[0], datasets[i + 1])

    return datasets[0]


def construct_dataloader(data_dir: Iterable[str], dataset_params: dict, sampler_params: dict, dataloader_params: dict,
                         collator_params: Optional[dict] = None, transform_params: Optional[dict] = None) -> DataLoader:
    """Constructs a DataLoader object from the parameters provided for the datasets, sampler, collator and transforms.

    Args:
        data_dir (Iterable[str]): A list of str defining the common path for all datasets to be constructed.
        dataset_params (dict): Dictionary of parameters defining each sub-datasets to be used.
        sampler_params (dict): Dictionary of parameters for the sampler to be used to sample from the dataset.
        dataloader_params (dict): Dictionary of parameters for the DataLoader itself.
        collator_params (dict): Optional; Dictionary of parameters defining the function to collate
            and stack samples from the sampler.
        transform_params: Optional; Dictionary defining the parameters of the transforms to perform
            when sampling from the dataset.

    Returns:
        loader (DataLoader): Object to handle the returning of batched samples from the dataset.
    """
    # --+ MAKE SUB-DATASETS +=========================================================================================+
    # List to hold all the sub-datasets defined by dataset_params to be intersected together into a single dataset.
    subdatasets = []

    # Iterate through all the sub-datasets defined in dataset_params.
    for key in dataset_params.keys():

        # Get the params for this sub-dataset.
        subdataset_params = dataset_params[key]

        # Get the constructor for the class of dataset defined in params.
        _subdataset = utils.func_by_str(module=subdataset_params['module'],
                                        func=subdataset_params['name'])

        # Construct the root to the sub-dataset's files.
        subdataset_root = os.sep.join((*data_dir, subdataset_params['root']))

        # Construct transforms for samples returned from this sub-dataset -- if found.
        transformations = None
        if transform_params is not None:
            transformations = make_transformations(transform_params[key])

        # Construct the sub-dataset using the objects defined from params, and append to list of sub-datasets.
        subdatasets.append(_subdataset(root=subdataset_root, transforms=transformations,
                                       **dataset_params[key]['params']))

    # Intersect sub-datasets to form single dataset if more than one sub-dataset exists. Else, just set that to dataset.
    dataset = subdatasets[0]
    if len(subdatasets) > 1:
        dataset = intersect_datasets(subdatasets)

    # --+ MAKE SAMPLERS +=============================================================================================+
    sampler = utils.func_by_str(module=sampler_params['module'], func=sampler_params['name'])
    sampler = sampler(dataset=subdatasets[0], roi=make_bounding_box(sampler_params['roi']), **sampler_params['params'])

    # --+ MAKE DATALOADERS +==========================================================================================+
    collator = None
    if collator_params is not None:
        collator = utils.func_by_str(collator_params['module'], collator_params['name'])

    return DataLoader(dataset, sampler=sampler, collate_fn=collator, **dataloader_params)


def load_all_samples(dataloader: DataLoader) -> np.ndarray:
    """Loads all sample masks from parsed DataLoader and computes the modes of their classes.

    Args:
        dataloader (DataLoader): DataLoader containing samples. Must be using a dataset with __len__ attribute
            and a sampler that returns a dict with a 'mask' key.

    Returns:
        sample_modes (np.ndarray): 2D array of the class modes within every sample defined by the parsed DataLoader.
    """
    sample_modes = []
    for sample in alive_it(dataloader):
        modes = utils.find_patch_modes(sample['mask'])
        sample_modes.append(modes)

    sample_modes = np.array(sample_modes)

    return sample_modes


def make_bounding_box(roi: Optional[Union[tuple, list, bool]] = False) -> Optional[BoundingBox]:
    if roi is False:
        return None
    else:
        return BoundingBox(*roi)


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
        if len(transform_params) == 1:
            return transform

        # If more than one transform found, append to list for composition.
        else:
            transformations.append(transform)

    # Compose transforms together and return.
    return transforms.Compose(transformations)


@utils.return_updated_kwargs
def make_datasets(root: Optional[str] = '', n_samples: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                  patch_size: Optional[Union[int, Tuple[int]]] = 256, plot: bool = False, p_dist: bool = False,
                  **params) -> Tuple[Dict[str, DataLoader], dict, list, dict]:
    """Constructs train, validation and test datasets and places in DataLoaders for use in model fitting and testing.

    Args:
        n_samples (list[float] or tuple[float]): Optional; Three values giving the fractional sizes of the datasets,
            in the order (train, validation, test).
        plot (bool): Optional; Whether to plot pie charts of the class distributions within each dataset.
        p_dist (bool): Optional; Whether to print to screen the distribution of classes within each dataset.

    Keyword Args:
        hyperparams (dict): Dictionary of hyper-parameters for the model.
        batch_size (int): Number of samples in each batch to be returned by the DataLoaders.
        elim (bool): Whether to eliminate classes with no samples in.

    Returns:
        loaders (dict): Dictionary of the DataLoader for training, validation and testing.
        n_batches (dict): Dictionary of the number of batches to return/ yield in each train, validation and test epoch.
        class_dist (list): The class distribution of the entire dataset, sorted from largest to smallest class.
        updated_keys (dict):
    """
    # Gets out the parameters for the DataLoaders from params.
    dataloader_params = params['hyperparams']['params']
    dataset_params = params['dataset_params']
    sampler_params = params['sampler_params']
    transform_params = params['transform_params']
    batch_size = dataloader_params['batch_size']

    # Load manifest from cache for this dataset.
    manifest = pd.read_csv(utils.get_manifest())
    class_dist = utils.subpopulations_from_manifest(manifest)

    # Finds the empty classes and returns modified classes, a dict to convert between the old and new systems
    # and new colours.
    classes, forwards, colours = utils.load_data_specs(class_dist=class_dist, elim=params['elim'])

    # Inits dicts to hold the variables and lists for train, validation and test.
    n_batches = {}
    loaders = {}

    for mode in ('train', 'val', 'test'):
        # Calculates number of batches.
        n_batches[mode] = int(sampler_params[mode]['params']['length'] / batch_size)

        # --+ MAKE DATASETS +=========================================================================================+
        print(f'CREATING {mode} DATASET')
        loaders[mode] = construct_dataloader(params['dir']['data'], dataset_params[mode], sampler_params[mode],
                                             dataloader_params, collator_params=params['collator'],
                                             transform_params=transform_params[mode])
        print('DONE')

    # Transform class dist if elimination of classes has occurred.
    if params['elim']:
        class_dist = utils.class_dist_transform(class_dist, forwards)

    # Prints class distribution in a pretty text format using tabulate to stdout.
    if p_dist:
        utils.print_class_dist(class_dist)

    params['hyperparams']['model_params']['n_classes'] = len(classes)
    params['classes'] = classes
    params['colours'] = colours

    return loaders, n_batches, class_dist, params
