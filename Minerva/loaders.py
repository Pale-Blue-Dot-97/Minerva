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
from typing import Optional, Union, Tuple, Dict
from Minerva.utils import utils
from torch.utils.data import DataLoader
from torchvision import transforms


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def intersect_datasets(datasets):
    def intersect_pair_datasets(a, b):
        return a & b
    
    for i in range(len(datasets) - 1):
        datasets[0] = intersect_pair_datasets(datasets[0], datasets[i + 1])

    return datasets[0]


def construct_dataloader(data_dir, dataset_params, sampler_params, dataloader_params, 
                         collator_params=None, transform_params=None):
    subdatasets = []
    for key in dataset_params.keys():
        subdataset_params = dataset_params[key]
        _subdataset = utils.func_by_str(module=subdataset_params['module'],
                                        func=subdataset_params['name'])
        subdataset_root = os.sep.join((*data_dir, subdataset_params['root']))
        
        transformations = None
        if transform_params is not None:
            transformations = make_transformations(transform_params[key])
        
        subdatasets.append(_subdataset(root=subdataset_root, transforms=transformations,
                                       **dataset_params[key]['params']))

    dataset = subdatasets[0]
    if len(subdatasets) > 1:
        dataset = intersect_datasets(subdatasets)

    # --+ MAKE SAMPLERS +=========================================================================================+
    sampler = utils.func_by_str(module=sampler_params['module'], func=sampler_params['name'])
    sampler = sampler(dataset=subdatasets[0], **sampler_params['params'])

    # --+ MAKE DATALOADERS +======================================================================================+
    collator = None
    if collator_params is not None:
        collator = utils.func_by_str(collator_params['module'], collator_params['name'])

    return DataLoader(dataset, sampler=sampler, collate_fn=collator, **dataloader_params)


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

    # TODO: FIND CLASS DISTRIBUTION FROM MANIFEST

    # Finds the empty classes and returns modified classes, a dict to convert between the old and new systems
    # and new colours.
    #print('\nFINDING EMPTY CLASSES')
    #new_classes, forwards, new_colours = utils.eliminate_classes(
    #    utils.find_empty_classes(class_dist=class_dists['ALL']))

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


    # Combines all scenes together to output a class_dist for the entire dataset.
    #all_scenes = scenes['train'] + scenes['val'] + scenes['test']
    #class_dist = utils.subpopulations_from_manifest(utils.select_df_by_scenes(manifest, all_scenes),
    #                                                func=label_func, plot=plot)

    # Transform class dist if elimination of classes has occurred.
    #if params['elim']:
    #    class_dist = utils.class_dist_transform(class_dist, forwards)

    # Prints class distribution in a pretty text format using tabulate to stdout.
    #if p_dist:
    #    utils.print_class_dist(class_dist)

    # TEMP -- ELIMINATE CLASSES NEEDS FIXING!
    _, aux_configs = utils.load_configs('../../config/config.yml')
    data_config = aux_configs['data_config']
    new_classes = data_config['classes']
    new_colours = data_config['colours']

    params['hyperparams']['model_params']['n_classes'] = len(new_classes)
    params['classes'] = new_classes
    params['colours'] = new_colours

    class_dist = []

    return loaders, n_batches, class_dist, params
