# -*- coding: utf-8 -*-
# Copyright (C) 2022 Harry Baker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
"""Functionality and custom code for constructing datasets, samplers and :class:`DataLoaders` for ``minerva``."""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from numpy.typing import NDArray
except (ModuleNotFoundError, ImportError):
    NDArray = Sequence

import os

import numpy as np
import pandas as pd
import torch
from alive_progress import alive_it
from catalyst.data.sampler import DistributedSamplerWrapper
from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, IntersectionDataset, RasterDataset
from torchgeo.datasets.utils import BoundingBox, concat_samples, stack_samples
from torchvision.transforms import RandomApply

from minerva.transforms import MinervaCompose
from minerva.utils import aux_configs, config, utils

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2022 Harry Baker"


# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
IMAGERY_CONFIG = aux_configs["imagery_config"]

# Path to cache directory.
CACHE_DIR = os.sep.join(config["dir"]["cache"])

__all__ = [
    "PairedDataset",
    "construct_dataloader",
    "get_collator",
    "get_manifest",
    "get_transform",
    "load_all_samples",
    "make_bounding_box",
    "make_dataset",
    "make_loaders",
    "make_manifest",
    "make_transformations",
    "stack_sample_pairs",
    "intersect_datasets",
    "get_manifest_path",
]


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class PairedDataset(RasterDataset):
    """Custom dataset to act as a wrapper to other datasets to handle paired sampling.

    Attributes:
        dataset (RasterDataset): Wrapped dataset to sampled from.

    Args:
        dataset_cls (RasterDataset): Constructor for a :class:`RasterDataset` to be wrapped for paired sampling.
    """

    def __init__(
        self,
        dataset_cls: RasterDataset,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dataset = dataset_cls(*args, **kwargs)

    def __getitem__(
        self, queries: Tuple[BoundingBox, BoundingBox]
    ) -> Tuple[Dict[str, Any], ...]:
        return self.dataset.__getitem__(queries[0]), self.dataset.__getitem__(
            queries[1]
        )

    def __getattr__(self, item):
        if item in self.dataset.__dict__:
            return getattr(self.dataset, item)
        elif item in self.__dict__:
            return getattr(self, item)
        else:
            raise AttributeError

    def __repr__(self) -> str:
        return self.dataset.__repr__()


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_collator(collator_params: Dict[str, str] = config["collator"]) -> Callable:
    """Gets the function defined in parameters to collate samples together to form a batch.

    Args:
        collator_params (Dict[str, str]): Optional; Dictionary that must contain keys for
            'module' and 'name' of the collation function. Defaults to config['collator'].

    Returns:
        Callable: Collation function found from parameters given.
    """
    collator = None
    if collator_params is not None:
        module = collator_params.pop("module", "")
        if module == "":
            collator = globals()[collator_params["name"]]
        else:
            collator = utils.func_by_str(module, collator_params["name"])
    return collator


# TODO: Document :func:`stack_sample_pairs`.
def stack_sample_pairs(
    samples: Iterable[Tuple[Dict[Any, Any]]]
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    a, b = tuple(zip(*samples))
    return stack_samples(a), stack_samples(b)


def intersect_datasets(
    datasets: List[GeoDataset], sample_pairs: bool = False
) -> IntersectionDataset:
    """Intersects a list of :class:`GeoDataset` together to return a single dataset object.

    Args:
        datasets (List[GeoDataset]): List of datasets to intersect together. Should have some geospatial overlap.

    Returns:
        IntersectionDataset: Final dataset object representing an intersection of all the parsed datasets.
    """

    def intersect_pair_datasets(a: GeoDataset, b: GeoDataset) -> IntersectionDataset:
        if sample_pairs:
            return IntersectionDataset(
                a, b, collate_fn=utils.pair_collate(concat_samples)
            )
        else:
            return a & b

    master_dataset = datasets[0]

    # if sample_pairs and not _INTERSECTION_CHANGED:
    #    IntersectionDataset = utils.pair_return(IntersectionDataset)

    for i in range(len(datasets) - 1):
        master_dataset = intersect_pair_datasets(master_dataset, datasets[i + 1])

    return master_dataset


def make_dataset(
    data_directory: Iterable[str],
    dataset_params: Dict[Any, Any],
    transform_params: Optional[Dict[Any, Any]] = None,
    sample_pairs: bool = False,
) -> Tuple[Any, List[Any]]:
    """Constructs a dataset object from `n` sub-datasets given by the parameters supplied.

    Args:
        data_directory (Iterable[str]): List defining the path to the directory containing the data.
        dataset_params (dict): Dictionary of parameters defining each sub-datasets to be used.
        transform_params: Optional; Dictionary defining the parameters of the transforms to perform
            when sampling from the dataset.

    Returns:
        dataset: Dataset object formed by the parameters given.
        sub_datasets (list): List of the sub-datasets created that constitute `dataset`.
    """
    # --+ MAKE SUB-DATASETS +=========================================================================================+
    # List to hold all the sub-datasets defined by dataset_params to be intersected together into a single dataset.
    sub_datasets = []

    # Iterate through all the sub-datasets defined in dataset_params.
    for key in dataset_params.keys():

        # Get the params for this sub-dataset.
        sub_dataset_params = dataset_params[key]

        # Get the constructor for the class of dataset defined in params.
        _sub_dataset = utils.func_by_str(
            module_path=sub_dataset_params["module"], func=sub_dataset_params["name"]
        )

        # Construct the root to the sub-dataset's files.
        sub_dataset_root = os.sep.join((*data_directory, sub_dataset_params["root"]))

        # Construct transforms for samples returned from this sub-dataset -- if found.
        transformations = None
        try:
            if transform_params[key]:
                transformations = make_transformations(transform_params[key], key=key)
        except (KeyError, TypeError):
            pass

        if sample_pairs:
            sub_dataset = PairedDataset(
                _sub_dataset,
                root=sub_dataset_root,
                transforms=transformations,
                **dataset_params[key]["params"],
            )
        else:
            sub_dataset = _sub_dataset(
                root=sub_dataset_root,
                transforms=transformations,
                **dataset_params[key]["params"],
            )

        # Construct the sub-dataset using the objects defined from params, and append to list of sub-datasets.
        sub_datasets.append(sub_dataset)

    # Intersect sub-datasets to form single dataset if more than one sub-dataset exists. Else, just set that to dataset.
    dataset = sub_datasets[0]
    if len(sub_datasets) > 1:
        dataset = intersect_datasets(sub_datasets, sample_pairs=sample_pairs)

    return dataset, sub_datasets


def construct_dataloader(
    data_directory: Iterable[str],
    dataset_params: Dict[str, Any],
    sampler_params: Dict[str, Any],
    dataloader_params: Dict[str, Any],
    collator_params: Optional[Dict[str, Any]] = None,
    transform_params: Optional[Dict[str, Any]] = None,
    rank: int = 0,
    world_size: int = 1,
    sample_pairs: bool = False,
) -> DataLoader:
    """Constructs a DataLoader object from the parameters provided for the datasets, sampler, collator and transforms.

    Args:
        data_directory (Iterable[str]): A list of str defining the common path for all datasets to be constructed.
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
    dataset, subdatasets = make_dataset(
        data_directory, dataset_params, transform_params, sample_pairs=sample_pairs
    )

    # --+ MAKE SAMPLERS +=============================================================================================+
    sampler = utils.func_by_str(
        module_path=sampler_params["module"], func=sampler_params["name"]
    )

    batch_sampler = True if "batch_size" in sampler_params["params"] else False

    if batch_sampler:
        assert sampler_params["params"]["batch_size"] % world_size == 0
        per_device_batch_size = sampler_params["params"]["batch_size"] // world_size
        sampler_params["params"]["batch_size"] = per_device_batch_size

    sampler = sampler(
        dataset=subdatasets[0],
        roi=make_bounding_box(sampler_params["roi"]),
        **sampler_params["params"],
    )

    # --+ MAKE DATALOADERS +==========================================================================================+
    collator = get_collator(collator_params)
    _dataloader_params = dataloader_params.copy()

    if world_size > 1:
        # Wraps sampler for distributed computing.
        sampler = DistributedSamplerWrapper(sampler, num_replicas=world_size, rank=rank)

        # Splits batch size across devices.
        assert dataloader_params["batch_size"] % world_size == 0
        per_device_batch_size = dataloader_params["batch_size"] // world_size
        _dataloader_params["batch_size"] = per_device_batch_size

    if sample_pairs and not torch.cuda.device_count() > 1:
        collator = utils.pair_collate(collator)

    if batch_sampler:
        _dataloader_params["batch_sampler"] = sampler
        del _dataloader_params["batch_size"]
    else:
        _dataloader_params["sampler"] = sampler

    return DataLoader(dataset, collate_fn=collator, **_dataloader_params)


def make_bounding_box(
    roi: Union[Sequence[float], bool] = False
) -> Optional[BoundingBox]:
    """Construct a BoundingBox object from the corners of the box. False for no BoundingBox.

    Args:
        roi (tuple[float] or list[float] or bool): Either a tuple or array of values defining the corners
            of a bounding box or False to designate no BoundingBox is defined.

    Returns:
        BoundingBox object made from parsed values or None if False was given.
    """
    if roi is False:
        return None
    else:
        return BoundingBox(*roi)


def get_transform(name: str, params: Dict[str, Any]) -> Any:
    """Creates a TensorBoard transform object based on config parameters.

    Returns:
        Initialised TensorBoard transform object specified by config parameters.
    """
    module = params.pop("module", "torchvision.transforms")

    # Gets the transform requested by config parameters.
    transform = utils.func_by_str(module, name)

    return transform(**params)


def make_transformations(
    transform_params: Union[Dict[str, Any], bool], key: str = None
) -> Optional[Any]:
    """Constructs a transform or series of transforms based on parameters provided.

    Args:
        transform_params (dict): Parameters defining transforms desired. The name of each transform should be the key,
            while the kwargs for the transform should be the value of that key as a dict.

    Example:
        >>> transform_params = {
        >>>    "CenterCrop": {"module": "torchvision.torch", "size": 128},
        >>>     "RandomHorizontalFlip": {"module": "torchvision.torch", "p": 0.7}
        >>> }
        >>> transforms = make_transforms(transform_params)

    Returns:
        If no parameters are parsed, None is returned.
        If only one transform is defined by the parameters, returns a Transforms object.
        If multiple transforms are defined, a Compose object of Transform objects is returned.
    """
    transformations = []

    # If no transforms are specified, return None.
    if not transform_params:
        return None

    # Get each transform.
    for name in transform_params:
        if name == "RandomApply":
            random_transforms = []
            random_params = transform_params[name].copy()
            p = random_params.pop("p", 0.5)

            for ran_name in random_params:
                random_transforms.append(
                    get_transform(ran_name, random_params[ran_name])
                )

            transformations.append(RandomApply(random_transforms, p=p))

        else:
            transformations.append(get_transform(name, transform_params[name]))

    # Compose transforms together and return.
    return MinervaCompose(transformations, key)


@utils.return_updated_kwargs
def make_loaders(
    rank: int = 0,
    world_size: int = 1,
    p_dist: bool = False,
    **params,
) -> Tuple[
    Dict[str, DataLoader], Dict[str, int], List[Tuple[int, int]], Dict[Any, Any]
]:
    """Constructs train, validation and test datasets and places in DataLoaders for use in model fitting and testing.

    Args:
        p_dist (bool): Optional; Whether to print to screen the distribution of classes within each dataset.

    Keyword Args:
        hyperparams (dict): Dictionary of hyper-parameters for the model.
        batch_size (int): Number of samples in each batch to be returned by the DataLoaders.
        elim (bool): Whether to eliminate classes with no samples in.

    Returns:
        loaders (dict): Dictionary of the DataLoader for training, validation and testing.
        n_batches (dict): Dictionary of the number of batches to return/ yield in each train, validation and test epoch.
        class_dist (list): The class distribution of the entire dataset, sorted from largest to smallest class.
        params (dict):
    """
    # Gets out the parameters for the DataLoaders from params.
    dataloader_params: Dict[Any, Any] = params["hyperparams"]["params"]
    dataset_params: Dict[str, Any] = params["dataset_params"]
    sampler_params: Dict[str, Any] = params["sampler_params"]
    transform_params: Dict[str, Any] = params["transform_params"]
    batch_size: int = dataloader_params["batch_size"]

    # Load manifest from cache for this dataset.
    manifest = get_manifest(get_manifest_path())
    class_dist = utils.modes_from_manifest(manifest)

    # Finds the empty classes and returns modified classes, a dict to convert between the old and new systems
    # and new colours.
    new_classes, forwards, new_colours = utils.load_data_specs(
        class_dist=class_dist, elim=params["elim"]
    )

    # Inits dicts to hold the variables and lists for train, validation and test.
    n_batches = {}
    loaders = {}

    for mode in dataset_params.keys():
        this_transform_params = transform_params[mode]
        if params["elim"]:
            if this_transform_params["mask"] is not Dict:
                this_transform_params["mask"] = {
                    "ClassTransform": {
                        "module": "minerva.transforms",
                        "transform": forwards,
                    }
                }
            else:
                this_transform_params["mask"]["ClassTransform"] = {
                    "module": "minerva.transforms",
                    "transform": forwards,
                }

        # Calculates number of batches.
        n_batches[mode] = int(sampler_params[mode]["params"]["length"] / batch_size)

        # --+ MAKE DATASETS +=========================================================================================+
        print(f"CREATING {mode} DATASET")
        loaders[mode] = construct_dataloader(
            params["dir"]["data"],
            dataset_params[mode],
            sampler_params[mode],
            dataloader_params,
            collator_params=params["collator"],
            transform_params=this_transform_params,
            rank=rank,
            world_size=world_size,
            sample_pairs=params["sample_pairs"],
        )
        print("DONE")

    # Transform class dist if elimination of classes has occurred.
    if params["elim"]:
        class_dist = utils.class_dist_transform(class_dist, forwards)

    # Prints class distribution in a pretty text format using tabulate to stdout.
    if p_dist:
        utils.print_class_dist(class_dist)

    params["hyperparams"]["model_params"]["n_classes"] = len(new_classes)
    params["classes"] = new_classes
    params["colours"] = new_colours
    params["max_pixel_value"] = IMAGERY_CONFIG["data_specs"]["max_value"]

    return loaders, n_batches, class_dist, params


def get_manifest_path() -> str:
    """Gets the path to the manifest for the dataset to be used.

    Returns:
        Path to manifest as string.
    """
    return os.sep.join([CACHE_DIR, f"{utils.get_dataset_name()}_Manifest.csv"])


def get_manifest(manifest_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(manifest_path)
    except FileNotFoundError as err:
        print(err)

        print("CONSTRUCTING MISSING MANIFEST")
        manifest = make_manifest()

        print(f"MANIFEST TO FILE -----> {manifest_path}")
        manifest.to_csv(manifest_path)

        return manifest


def make_manifest(mf_config: Dict[Any, Any] = config) -> pd.DataFrame:
    """Constructs a manifest of the dataset detailing each sample therein.

    The dataset to construct a manifest of is defined by the 'data_config' value in the config.

    Returns:
        df (pd.DataFrame): The completed manifest as a DataFrame.
    """
    dataloader_params = mf_config["dataloader_params"]
    dataset_params = mf_config["dataset_params"]
    sampler_params = mf_config["sampler_params"]
    collator_params = mf_config["collator"]

    print("CONSTRUCTING DATASET")
    loader = construct_dataloader(
        mf_config["dir"]["data"],
        dataset_params,
        sampler_params,
        dataloader_params,
        collator_params=collator_params,
    )

    print("FETCHING SAMPLES")
    df = pd.DataFrame()
    df["MODES"] = load_all_samples(loader)

    print("CALCULATING CLASS FRACTIONS")
    # Calculates the fractional size of each class in each patch.
    df = pd.DataFrame([row for row in df.apply(utils.class_frac, axis=1)])
    df.fillna(0, inplace=True)

    # Delete redundant MODES column.
    del df["MODES"]

    return df


def load_all_samples(dataloader: DataLoader) -> NDArray[Any]:
    """Loads all sample masks from parsed DataLoader and computes the modes of their classes.

    Args:
        dataloader (DataLoader): DataLoader containing samples. Must be using a dataset with __len__ attribute
            and a sampler that returns a dict with a 'mask' key.

    Returns:
        sample_modes (np.ndarray): 2D array of the class modes within every sample defined by the parsed DataLoader.
    """
    sample_modes: List[List[Tuple[int, int]]] = []
    for sample in alive_it(dataloader):
        modes = utils.find_patch_modes(sample["mask"])
        sample_modes.append(modes)

    return np.array(sample_modes)
