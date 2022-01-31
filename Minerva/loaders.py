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

TODO:
    * Re-incorporate the use of WeightedRandomSampler as an option
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
from typing import Optional, Union, Tuple, Dict
from Minerva.utils import utils
from torch.utils.data import DataLoader
from torchvision import transforms
from torchgeo.datasets.utils import download_url


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
        n_samples (list[float] or tuple[float]): Optional; Three values giving the fractional sizes of the datasets, in the
            order (train, validation, test).
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

    """
    # Defines the function to use to load the labels. Either to load the whole mask or just the centre label.
    label_func = utils.lc_load
    if model_type == 'scene classifier':
        label_func = utils.find_centre_label
    """

    # Finds the empty classes and returns modified classes, a dict to convert between the old and new systems
    # and new colours.
    print('\nFINDING EMPTY CLASSES')
    #new_classes, forwards, new_colours = utils.eliminate_classes(
    #    utils.find_empty_classes(class_dist=class_dists['ALL']))

    # Inits dicts to hold the variables and lists for train, validation and test.
    n_batches = {}
    loaders = {}
    class_dists = {}

    for mode in ('train', 'val', 'test'):
        print('\nFINDING CLASS DISTRIBUTION OF SCENES')
        # Find class distribution of dataset by scene IDs, not patch IDs.
        #class_dist = utils.subpopulations_from_manifest(utils.select_df_by_scenes(manifest, scenes[mode]),
        #                                                func=label_func, plot=plot)

        # Prints class distribution in a pretty text format using tabulate to stdout.
        #if p_dist:
        #    utils.print_class_dist(class_dist)

        # Transform class dist if elimination of classes has occurred.
        #if params['elim']:
        #    class_dist = utils.class_dist_transform(class_dist, forwards)
        #class_dists[mode] = class_dist

        # Calculates number of batches.
        n_batches[mode] = int(sampler_params[mode]['params']['length'] / batch_size)

        # --+ MAKE DATASETS +=========================================================================================+
        _image_dataset = utils.func_by_str(module=dataset_params[mode]['imagery']['module'],
                                           func=dataset_params[mode]['imagery']['name'])

        _label_dataset = utils.func_by_str(module=dataset_params[mode]['labels']['module'],
                                           func=dataset_params[mode]['labels']['name'])

        imagery_root = os.sep.join((*params['dir']['data'], dataset_params[mode]['imagery']['root']))
        labels_root = os.sep.join((*params['dir']['data'], dataset_params[mode]['labels']['root']))

        naip_url = "https://naipblobs.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/"
        tiles = [
            "m_3807511_ne_18_060_20181104.tif",
            "m_3807511_se_18_060_20181104.tif",
            "m_3807512_nw_18_060_20180815.tif",
            "m_3807512_sw_18_060_20180815.tif",
        ]
        for tile in tiles:
            download_url(naip_url + tile, imagery_root)

        print(f'CREATING {mode} DATASET')
        image_dataset = _image_dataset(root=imagery_root,
                                       transforms=make_transformations(transform_params[mode]['imagery']),
                                       **dataset_params[mode]['imagery']['params'])
        label_dataset = _label_dataset(root=labels_root,
                                       transforms=make_transformations(transform_params[mode]['labels']),
                                       **dataset_params[mode]['labels']['params'])

        dataset = image_dataset & label_dataset
        print('DONE')

        # --+ MAKE SAMPLERS +=========================================================================================+
        sampler = utils.func_by_str(module=sampler_params[mode]['module'], func=sampler_params[mode]['name'])

        print(f'CREATING {mode} SAMPLER')
        sampler = sampler(dataset=image_dataset, **sampler_params[mode]['params'])
        print('DONE')

        # --+ MAKE DATALOADERS +======================================================================================+
        print(f'CREATING {mode} LOADER')
        collator = utils.func_by_str(params['collator']['module'], params['collator']['name'])
        loaders[mode] = DataLoader(dataset, sampler=sampler, collate_fn=collator, **dataloader_params)
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
