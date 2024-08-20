# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2024 Harry Baker

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
"""Functionality for constructing datasets, manifests and :class:`~torch.utils.data.DataLoader` for :mod:`minerva`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "construct_dataloader",
    "make_dataset",
    "make_loaders",
    "get_manifest",
    "make_manifest",
]


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
import platform
import re
from copy import deepcopy
from datetime import timedelta
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pandas import DataFrame
from rasterio.crs import CRS
from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, NonGeoDataset, RasterDataset
from torchgeo.samplers.utils import _to_tuple

from minerva.samplers import DistributedSamplerWrapper, get_sampler
from minerva.transforms import MinervaCompose, init_auto_norm, make_transformations
from minerva.utils import universal_path, utils

from .collators import get_collator, stack_sample_pairs
from .paired import PairedGeoDataset, PairedNonGeoDataset
from .utils import (
    MinervaConcatDataset,
    cache_dataset,
    concatenate_datasets,
    intersect_datasets,
    load_all_samples,
    load_dataset_from_cache,
    masks_or_labels,
    unionise_datasets,
)


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def create_subdataset(
    dataset_class: Union[Callable[..., GeoDataset], Callable[..., NonGeoDataset]],
    paths: Union[str, Iterable[str]],
    subdataset_params: Dict[Literal["params"], Dict[str, Any]],
    transformations: Optional[Any],
    sample_pairs: bool = False,
) -> Union[GeoDataset, NonGeoDataset]:
    """Creates a sub-dataset based on the parameters supplied.

    Args:
        dataset_class (Callable[..., ~typing.datasets.GeoDataset]): Constructor for the sub-dataset.
        paths (str | ~typing.Iterable[str]): Paths to where the data for the dataset is located.
        subdataset_params (dict[Literal[params], dict[str, ~typing.Any]]): Parameters for the sub-dataset.
        transformations (~typing.Any): Transformations to apply to this sub-dataset.
        sample_pairs (bool): Will configure the dataset for paired sampling. Defaults to False.

    Returns:
        ~torchgeo.datasets.GeoDataset: Subdataset requested.
    """
    copy_params = deepcopy(subdataset_params)

    if "crs" in copy_params["params"]:
        copy_params["params"]["crs"] = CRS.from_epsg(copy_params["params"]["crs"])

    if sample_pairs:
        if "paths" in signature(dataset_class).parameters:
            return PairedGeoDataset(
                dataset_class,  # type: ignore[arg-type]
                paths=paths,
                transforms=transformations,
                **copy_params["params"],
            )

        elif "season_transform" in signature(dataset_class).parameters:
            if isinstance(paths, list):
                paths = paths[0]
            assert isinstance(paths, str)
            del copy_params["params"]["season_transform"]
            return PairedNonGeoDataset(
                dataset_class,  # type: ignore[arg-type]
                root=paths,
                transforms=transformations,
                season=True,
                season_transform="pair",
                **copy_params["params"],
            )
        elif "root" in signature(dataset_class).parameters:
            if isinstance(paths, list):
                paths = paths[0]
            assert isinstance(paths, str)
            return PairedNonGeoDataset(
                dataset_class,  # type: ignore[arg-type]
                root=paths,
                transforms=transformations,
                **copy_params["params"],
            )
        else:
            raise TypeError
    else:
        if "paths" in signature(dataset_class).parameters:
            return dataset_class(
                paths=paths,
                transforms=transformations,
                **copy_params["params"],
            )
        elif "root" in signature(dataset_class).parameters:
            if isinstance(paths, list):
                paths = paths[0]
            assert isinstance(paths, str)
            return dataset_class(
                root=paths,
                transforms=transformations,
                **copy_params["params"],
            )
        else:
            raise TypeError


def get_subdataset(
    data_directory: Union[Iterable[str], str, Path],
    dataset_params: Dict[str, Any],
    key: str,
    transformations: Optional[Any],
    sample_pairs: bool = False,
    cache: bool = True,
    cache_dir: Union[str, Path] = "",
) -> Union[GeoDataset, NonGeoDataset]:
    """Get a subdataset based on the parameters specified.

    If ``cache==True``, this will attempt to load a cached version of the dataset instance.
    If ``cache==False`` or the cached dataset does not exist, uses :func:`create_subdataset` to create it instead.

    Args:
        data_directory (~typing.Iterable[str] | str | ~pathlib.Path): Path to the parent data directory
            that the dataset being fetched should be in.
        dataset_params (dict[str, ~typing.Any]): Parameters defining the sub-datasets that will be unionised together.
        key (str): The key for this subdataset within ``dataset_params``.
        transformations (~typing.Any): Transformations to apply to this sub-dataset.
        sample_pairs (bool): Will configure the dataset for paired sampling. Defaults to False.
        cache (bool): Cache the dataset or load from cache if pre-existing. Defaults to True.
        cache_dir (str | ~pathlib.Path): Path to the directory to save the cached dataset (if ``cache==True``).
            Defaults to CWD.

    Returns:
        ~torchgeo.datasets.GeoDataset | ~torchgeo.datasets.NonGeoDataset: Subdataset requested.
    """
    # Get the params for this sub-dataset.
    sub_dataset_params = dataset_params[key]

    # Get the constructor for the class of dataset defined in params.
    _sub_dataset: Callable[..., GeoDataset] = utils.func_by_str(
        module_path=sub_dataset_params["module"], func=sub_dataset_params["name"]
    )

    # Construct the path to the sub-dataset's files.
    sub_dataset_paths = utils.compile_dataset_paths(
        universal_path(data_directory), sub_dataset_params["paths"]
    )

    sub_dataset: Optional[Union[GeoDataset, NonGeoDataset]]

    if cache or sub_dataset_params.get("cache_dataset"):
        this_hash = utils.make_hash(sub_dataset_params)

        cached_dataset_path = universal_path(cache_dir) / f"{this_hash}.obj"

        if cached_dataset_path.exists():
            print(f"\nLoad cached dataset {this_hash}")
            sub_dataset = load_dataset_from_cache(cached_dataset_path)

        else:
            # Ensure that no conflicts from caching datasets made in multiple processes arises.
            if dist.is_available() and dist.is_initialized():  # pragma: no cover
                # Get this process#s rank.
                rank = dist.get_rank()

                # Start a blocking action, ensuring only process 0 can create and cache the dataset.
                # All other processes will wait till 0 is finished.
                dist.monitored_barrier(timeout=timedelta(hours=4))

                if rank == 0:
                    print(f"\nCreating dataset on {rank}...")
                    sub_dataset = create_subdataset(
                        _sub_dataset,
                        sub_dataset_paths,
                        sub_dataset_params,
                        transformations,
                        sample_pairs=sample_pairs,
                    )

                    print(f"\nSaving dataset {this_hash}")
                    cache_dataset(sub_dataset, cached_dataset_path)

                # Other processes wait...
                else:
                    sub_dataset = None

                # End of blocking action.
                dist.monitored_barrier(timeout=timedelta(hours=4))

                # Now the other processes can load the newly created cached dataset from 0.
                if rank != 0:
                    print(f"\nLoading dataset from cache {this_hash} on {rank}")
                    sub_dataset = load_dataset_from_cache(cached_dataset_path)

            else:
                print("\nCreating dataset...")
                sub_dataset = create_subdataset(
                    _sub_dataset,
                    sub_dataset_paths,
                    sub_dataset_params,
                    transformations,
                    sample_pairs=sample_pairs,
                )

                print(f"\nSaving dataset {this_hash}")
                cache_dataset(sub_dataset, cached_dataset_path)

    else:
        sub_dataset = create_subdataset(
            _sub_dataset,
            sub_dataset_paths,
            sub_dataset_params,
            transformations,
            sample_pairs=sample_pairs,
        )

    assert sub_dataset is not None
    return sub_dataset


def make_dataset(
    data_directory: Union[Iterable[str], str, Path],
    dataset_params: Dict[Any, Any],
    sample_pairs: bool = False,
    cache: bool = True,
    cache_dir: Union[str, Path] = "",
) -> Tuple[Any, List[Any]]:
    """Constructs a dataset object from ``n`` sub-datasets given by the parameters supplied.

    Args:
        data_directory (~typing.Iterable[str] | str | ~pathlib.Path]): List defining the path to the directory
            containing the data.
        dataset_params (dict[~typing.Any, ~typing.Any]): Dictionary of parameters defining each sub-datasets to be used.
        sample_pairs (bool): Optional; ``True`` if paired sampling. This will ensure paired samples are handled
            correctly in the datasets.
        cache (bool): Cache the dataset or load from cache if pre-existing. Defaults to True.
        cache_dir (str | ~pathlib.Path): Path to the directory to save the cached dataset (if ``cache==True``).
            Defaults to CWD.

    Returns:
        tuple[~typing.Any, list[~typing.Any]]: Tuple of Dataset object formed by the parameters given and list of
        the sub-datasets created that constitute ``dataset``.
    """
    # --+ MAKE SUB-DATASETS +=========================================================================================+
    # List to hold all the sub-datasets defined by dataset_params to be intersected together into a single dataset.
    sub_datasets: Union[
        List[GeoDataset], List[Union[NonGeoDataset, MinervaConcatDataset]]
    ] = []

    if OmegaConf.is_config(dataset_params):
        dataset_params = OmegaConf.to_object(dataset_params)  # type: ignore[assignment]

    add_target_transforms = None
    add_multi_modal_transforms = None

    # Iterate through all the sub-datasets defined in `dataset_params`.
    for type_key in dataset_params.keys():
        # If this the sampler params, skip.
        if type_key == "sampler":
            continue

        if type_key in ("image", "mask", "label"):
            type_dataset_params = dataset_params[type_key]
        elif type_key == "transforms":
            add_multi_modal_transforms = dataset_params[type_key]
            continue
        else:
            continue

        # If there are no params, assume this is just a marker and no datasets are defined so skip.
        if type_dataset_params is None:
            continue

        # If the only params in the type are transforms, store them for later and skip making datasets for this type.
        if (
            len(type_dataset_params.keys()) == 1
            and "transforms" in type_dataset_params.keys()
        ):
            add_target_transforms = type_dataset_params["transforms"]
            continue

        type_subdatasets: Union[List[GeoDataset], List[NonGeoDataset]] = []

        multi_datasets_exist = False

        auto_norm = None
        master_transforms: Optional[Any] = None
        for area_key in type_dataset_params.keys():
            # If any of these keys are present, this must be a parameter set for a singular dataset at this level.
            if area_key in ("module", "name", "params", "paths"):
                multi_datasets_exist = False
                continue

            # If there are transforms specified, make them. These could cover a single dataset or many.
            elif area_key == "transforms":
                if isinstance(type_dataset_params[area_key], dict):
                    transform_params = type_dataset_params[area_key]
                    auto_norm = transform_params.get("AutoNorm")

                    # If transforms aren't specified for a particular modality of the sample,
                    # assume they're for the same type as the dataset.
                    if (
                        not ("image", "mask", "label")
                        in type_dataset_params[area_key].keys()
                    ):
                        transform_params = {type_key: type_dataset_params[area_key]}
                else:
                    transform_params = False

                master_transforms = make_transformations(transform_params)

            # Assuming that these keys are names of datasets.
            else:
                multi_datasets_exist = True

                if isinstance(type_dataset_params[area_key].get("transforms"), dict):
                    transform_params = type_dataset_params[area_key]["transforms"]
                    auto_norm = transform_params.get("AutoNorm")
                else:
                    transform_params = False

                transformations = make_transformations({type_key: transform_params})

                # Send the params for this area key back through this function to make the sub-dataset.
                sub_dataset = get_subdataset(
                    data_directory,
                    type_dataset_params,
                    area_key,
                    transformations,
                    sample_pairs=sample_pairs,
                    cache=cache,
                    cache_dir=cache_dir,
                )

                # Performs an auto-normalisation initialisation which finds the mean and std of the dataset
                # to make a transform, then adds the transform to the dataset's existing transforms.
                if auto_norm:
                    if isinstance(sub_dataset, RasterDataset):
                        init_auto_norm(sub_dataset, auto_norm)
                    else:
                        raise TypeError(  # pragma: no cover
                            "AutoNorm only supports normalisation of data "
                            + f"from RasterDatasets, not {type(sub_dataset)}!"
                        )

                    # Reset back to None.
                    auto_norm = None

                type_subdatasets.append(sub_dataset)  # type: ignore[arg-type]

        # Unionise all the sub-datsets of this modality together.
        if multi_datasets_exist:
            if isinstance(type_subdatasets[0], GeoDataset):
                sub_datasets.append(unionise_datasets(type_subdatasets, master_transforms))  # type: ignore[arg-type]
            else:
                sub_datasets.append(concatenate_datasets(type_subdatasets, master_transforms))  # type: ignore[arg-type]

        # Add the subdataset of this modality to the list.
        else:
            sub_dataset = get_subdataset(
                data_directory,
                dataset_params,
                type_key,
                master_transforms,
                sample_pairs=sample_pairs,
                cache=cache,
                cache_dir=cache_dir,
            )

            # Performs an auto-normalisation initialisation which finds the mean and std of the dataset
            # to make a transform, then adds the transform to the dataset's existing transforms.
            if auto_norm:
                if isinstance(sub_dataset, RasterDataset):
                    init_auto_norm(sub_dataset, auto_norm)

                    # Reset back to None.
                    auto_norm = None
                else:
                    raise TypeError(  # pragma: no cover
                        f"AutoNorm only supports normalisation of data from RasterDatasets, not {type(sub_dataset)}!"
                    )

            sub_datasets.append(sub_dataset)  # type: ignore[arg-type]

    # Intersect sub-datasets of differing modalities together to form single dataset
    # if more than one sub-dataset exists. Else, just set that to dataset.
    dataset = sub_datasets[0]
    if len(sub_datasets) > 1 and all(isinstance(x, GeoDataset) for x in sub_datasets):
        dataset = intersect_datasets(sub_datasets)  # type: ignore[arg-type]

    if add_target_transforms is not None:
        target_key = masks_or_labels(dataset_params)
        target_transforms = make_transformations({target_key: add_target_transforms})

        if hasattr(dataset, "transforms"):
            if isinstance(dataset.transforms, MinervaCompose):
                assert target_transforms is not None
                dataset.transforms += target_transforms
            else:
                dataset.transforms = target_transforms
        else:
            raise TypeError(
                f"dataset of type {type(dataset)} has no ``transforms`` atttribute!"
            )

    if add_multi_modal_transforms is not None:
        multi_modal_transforms = make_transformations(
            {"both": add_multi_modal_transforms}
        )
        if hasattr(dataset, "transforms"):
            if isinstance(dataset.transforms, MinervaCompose):
                assert multi_modal_transforms is not None
                dataset.transforms += multi_modal_transforms
            else:
                dataset.transforms = multi_modal_transforms
        else:
            raise TypeError(
                f"dataset of type {type(dataset)} has no ``transforms`` atttribute!"
            )

    return dataset, sub_datasets


def construct_dataloader(
    data_directory: Union[Iterable[str], str, Path],
    dataset_params: Dict[str, Any],
    sampler_params: Dict[str, Any],
    dataloader_params: Dict[str, Any],
    batch_size: int,
    collator_params: Optional[Dict[str, Any]] = None,
    rank: int = 0,
    world_size: int = 1,
    sample_pairs: bool = False,
    cache: bool = True,
    cache_dir: Union[Path, str] = "",
) -> DataLoader[Iterable[Any]]:
    """Constructs a :class:`~torch.utils.data.DataLoader` object from the parameters provided for the
    datasets, sampler, collator and transforms.

    Args:
        data_directory (~typing.Iterable[str]): A list of :class:`str` defining the common path
            for all datasets to be constructed.
        dataset_params (dict[str, ~typing.Any]): Dictionary of parameters defining each sub-datasets to be used.
        sampler_params (dict[str, ~typing.Any]): Dictionary of parameters for the sampler to be used
            to sample from the dataset.
        dataloader_params (dict[str, ~typing.Any]): Dictionary of parameters for the DataLoader itself.
        batch_size (int): Number of samples per (global) batch.
        collator_params (dict[str, ~typing.Any]): Optional; Dictionary of parameters defining the function to collate
            and stack samples from the sampler.
        rank (int): Optional; The rank of this process for distributed computing.
        world_size (int): Optional; The total number of processes within a distributed run.
        sample_pairs (bool): Optional; True if paired sampling. This will wrap the collation function
            for paired samples.

    Returns:
        ~torch.utils.data.DataLoader: Object to handle the returning of batched samples from the dataset.
    """
    dataset, subdatasets = make_dataset(
        data_directory,
        dataset_params,
        sample_pairs=sample_pairs,
        cache=cache,
        cache_dir=cache_dir,
    )

    # --+ MAKE SAMPLERS +=============================================================================================+
    per_device_batch_size = None

    batch_sampler = True if re.search(r"Batch", sampler_params["_target_"]) else False
    if batch_sampler and dist.is_available() and dist.is_initialized():  # type: ignore[attr-defined]
        assert sampler_params["batch_size"] % world_size == 0  # pragma: no cover
        per_device_batch_size = (
            sampler_params["batch_size"] // world_size
        )  # pragma: no cover

    sampler = get_sampler(
        sampler_params, subdatasets[0], batch_size=per_device_batch_size
    )

    # --+ MAKE DATALOADERS +==========================================================================================+
    collator = get_collator(collator_params)

    # Add batch size from top-level parameters to the dataloader parameters.
    dataloader_params["batch_size"] = batch_size

    if world_size > 1:
        # Wraps sampler for distributed computing.
        sampler = DistributedSamplerWrapper(sampler, num_replicas=world_size, rank=rank)

        # Splits batch size across devices.
        assert batch_size % world_size == 0
        per_device_batch_size = batch_size // world_size
        dataloader_params["batch_size"] = per_device_batch_size

    if sample_pairs:
        if not torch.cuda.device_count() > 1 and platform.system() != "Windows":
            collator = utils.pair_collate(collator)

        # Can't wrap functions in distributed runs due to pickling error.
        # This also seems to occur on Windows systems, even when not multiprocessing?
        # Therefore, the collator is set to `stack_sample_pairs` automatically.
        else:  # pragma: no cover
            collator = stack_sample_pairs

    if batch_sampler:
        dataloader_params["batch_sampler"] = sampler
        del dataloader_params["batch_size"]
    else:
        dataloader_params["sampler"] = sampler

    return DataLoader(dataset, collate_fn=collator, **dataloader_params)


def _add_class_transform(
    class_matrix: Dict[int, int], dataset_params: Dict[str, Any], target_key: str
) -> Dict[str, Any]:
    class_transform = {
        "ClassTransform": {
            "_target_": "minerva.transforms.ClassTransform",
            "transform": class_matrix,
        }
    }
    if dataset_params[target_key] is None:
        dataset_params[target_key] = {"transforms": class_transform}
    elif not isinstance(dataset_params[target_key].get("transforms"), dict):
        dataset_params[target_key]["transforms"] = class_transform
    else:
        dataset_params[target_key]["transforms"]["ClassTransform"] = class_transform[
            "ClassTransform"
        ]
    return dataset_params


def _make_loader(
    rank,
    world_size,
    data_dir,
    cache_dir,
    dataset_params,
    sampler_params,
    dataloader_params,
    collator_params,
    class_matrix,
    batch_size,
    model_type,
    elim,
    sample_pairs,
    cache,
):
    target_key = None

    if not utils.check_substrings_in_string(model_type, "siamese"):
        target_key = masks_or_labels(dataset_params)

        if elim:
            dataset_params = _add_class_transform(
                class_matrix, dataset_params, target_key
            )

    # --+ MAKE DATASETS +=========================================================================================+
    loaders = construct_dataloader(
        data_dir,
        dataset_params,
        sampler_params,
        dataloader_params,
        batch_size,
        collator_params=collator_params,
        rank=rank,
        world_size=world_size,
        sample_pairs=sample_pairs,
        cache=cache,
        cache_dir=cache_dir,
    )

    # Calculates number of batches.
    assert hasattr(loaders.dataset, "__len__")
    n_batches = int(
        sampler_params.get(
            "length",
            sampler_params.get("num_samples", len(loaders.dataset)),
        )
        / batch_size
    )

    return loaders, n_batches, target_key


def make_loaders(
    rank: int = 0,
    world_size: int = 1,
    p_dist: bool = False,
    task_name: Optional[str] = None,
    **params,
) -> Tuple[
    Union[Dict[str, DataLoader[Iterable[Any]]], DataLoader[Iterable[Any]]],
    Union[Dict[str, int], int],
    List[Tuple[int, int]],
    Dict[Any, Any],
]:
    """Constructs train, validation and test datasets and places into :class:`~torch.utils.data.DataLoader` objects.

    Args:
        rank (int): Rank number of the process. For use with :class:`~torch.nn.parallel.DistributedDataParallel`.
        world_size (int): Total number of processes across all nodes. For use with
            :class:`~torch.nn.parallel.DistributedDataParallel`.
        p_dist (bool): Print to screen the distribution of classes within each dataset.
        task_name (str): Optional; If specified, only create the loaders for this task,
            overwriting experiment-wode parameters with those found in this task's params if found.

    Keyword Args:
        batch_size (int): Number of samples in each batch to be returned by the DataLoaders.
        elim (bool): Whether to eliminate classes with no samples in.
        model_type (str): Defines the type of the model. If ``siamese``, ensures inappropiate functionality is not used.
        dir (dict[str, ~typing.Any]): Dictionary providing the paths to directories needed.
            Must include the ``data`` path.
        loader_params (dict[str, ~typing.Any]): Parameters to be parsed to construct the
            :class:`~torch.utils.data.DataLoader`.
        dataset_params (dict[str, ~typing.Any]): Parameters to construct each dataset.
            See documentation on structure of these.
        sampler_params (dict[str, ~typing.Any]): Parameters to construct the samplers for each mode of model fitting.
        transform_params (dict[str, ~typing.Any]): Parameters to construct the transforms for each dataset.
            See documentation for the structure of these.
        collator (dict[str, ~typing.Any]): Defines the collator to use that will collate samples together into batches.
            Contains the ``module`` key to define the import path and the ``name`` key
            for name of the collation function.
        sample_pairs (bool): Activates paired sampling for Siamese models. Only used for ``train`` datasets.

    Returns:
        tuple[dict[str, ~torch.utils.data.DataLoader[~typing.Iterable]], dict[str, int], list[tuple[int, int]], dict]:
        :class:`tuple` of;
            * Dictionary of the :class:`~torch.utils.data.DataLoader` (s) for training, validation and testing.
            * Dictionary of the number of batches to return/ yield in each train, validation and test epoch.
            * The class distribution of the entire dataset, sorted from largest to smallest class.
            * Unused and updated kwargs.
    """
    task_params = params
    if task_name is not None:
        task_params = params["tasks"][task_name]

    data_dir = utils.fallback_params("data_root", task_params, params)
    cache_dir = params["cache_dir"]

    # Gets out the parameters for the DataLoaders from params.
    dataloader_params: Dict[Any, Any] = deepcopy(
        utils.fallback_params("loader_params", task_params, params)
    )

    if OmegaConf.is_config(dataloader_params):
        dataloader_params = OmegaConf.to_object(dataloader_params)  # type: ignore[assignment]

    dataset_params: Dict[str, Any] = utils.fallback_params(
        "dataset_params", task_params, params
    )

    imagery_config = task_params.get("imagery_config", {})
    data_config = task_params.get("data_config", {})

    batch_size: int = utils.fallback_params("batch_size", task_params, params)

    model_type = utils.fallback_params("model_type", task_params, params)
    class_dist: List[Tuple[int, int]] = [(0, 0)]

    classes = utils.fallback_params("classes", data_config, params, None)
    cmap_dict = utils.fallback_params("colours", data_config, params, None)

    # If no taxonomy is specified, create a basic one from the ``n_classes`` (if given).
    if classes is None:
        n_classes = utils.fallback_params("n_classes", task_params, params)
        if n_classes:
            classes = {i: f"class {i}" for i in range(n_classes)}

    new_classes: Dict[int, str] = {}
    new_colours: Dict[int, str] = {}
    class_matrix: Dict[int, int] = {}

    sample_pairs: Union[bool, Any] = utils.fallback_params(
        "sample_pairs", task_params, params, False
    )
    if not isinstance(sample_pairs, bool):  # pragma: no cover
        sample_pairs = False

    elim = utils.fallback_params("elim", task_params, params, False)
    cache = utils.fallback_params("cache_dataset", task_params, params, True)

    n_batches: Union[Dict[str, int], int]
    loaders: Union[Dict[str, DataLoader[Iterable[Any]]], DataLoader[Iterable[Any]]]

    collator_params = deepcopy(utils.fallback_params("collator", task_params, params))
    if OmegaConf.is_config(collator_params):
        collator_params = OmegaConf.to_object(collator_params)

    if "sampler" in dataset_params.keys():
        sampler_params: Dict[str, Any] = dataset_params["sampler"]

        if not utils.check_substrings_in_string(model_type, "siamese"):
            new_classes, class_matrix, new_colours, class_dist = get_data_specs(
                data_config["name"],
                classes,
                cmap_dict,
                cache_dir,
                data_dir,
                dataset_params,
                sampler_params,
                dataloader_params,
                collator_params,
                elim=elim,
            )

        print(f"CREATING {task_name} DATASET")
        loaders, n_batches, target_key = _make_loader(
            rank,
            world_size,
            data_dir,
            cache_dir,
            dataset_params,
            sampler_params,
            dataloader_params,
            collator_params,
            class_matrix,
            batch_size,
            model_type,
            elim=elim,
            sample_pairs=sample_pairs,
            cache=cache,
        )

    else:
        # Inits dicts to hold the variables and lists for train, validation and test.
        n_batches = {}
        loaders = {}

        for mode in dataset_params.keys():
            mode_sampler_params: Dict[str, Any] = dataset_params[mode]["sampler"]

            if (
                not utils.check_substrings_in_string(model_type, "siamese")
                and mode == "train"
            ):
                new_classes, class_matrix, new_colours, class_dist = get_data_specs(
                    data_config["name"],
                    classes,
                    cmap_dict,
                    cache_dir,
                    data_dir,
                    dataset_params[mode],
                    mode_sampler_params,
                    dataloader_params,
                    collator_params,
                    elim=elim,
                )

            # --+ MAKE DATASETS +=====================================================================================+
            print(f"CREATING {mode} DATASET")
            loaders[mode], n_batches[mode], target_key = _make_loader(
                rank,
                world_size,
                data_dir,
                cache_dir,
                dataset_params[mode],
                mode_sampler_params,
                dataloader_params,
                collator_params,
                class_matrix,
                batch_size,
                model_type,
                elim=elim,
                sample_pairs=sample_pairs if mode == "train" else False,
                cache=cache,
            )

            print("DONE")

    if (
        not utils.check_substrings_in_string(model_type, "siamese")
        and "sampler" in dataset_params
    ):
        # Transform class dist if elimination of classes has occurred.
        if elim:
            class_dist = utils.class_dist_transform(class_dist, class_matrix)

        # Prints class distribution in a pretty text format using tabulate to stdout.
        if p_dist:
            utils.print_class_dist(class_dist, new_classes)

        task_params["n_classes"] = len(new_classes)
        model_params = utils.fallback_params("model_params", task_params, params, {})
        model_params["n_classes"] = len(new_classes)

        if "model_params" in task_params:
            task_params["model_params"]["params"] = model_params
        else:
            task_params["model_params"] = {"params": model_params}

        task_params["classes"] = new_classes
        task_params["colours"] = new_colours

    if task_params.get("max_pixel_value") is None:
        task_params["max_pixel_value"] = imagery_config.get("max_pixel_value", 256)

    if task_params.get("model_type") is None:
        task_params["model_type"] = model_type

    # Store the name of the target key (either `mask` or `label`)
    task_params["target_key"] = target_key

    return loaders, n_batches, class_dist, task_params


def get_data_specs(
    manifest_name: Union[str, Path],
    classes: Dict[int, str],
    cmap_dict: Dict[int, str],
    cache_dir: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    dataset_params: Optional[Dict[str, Any]] = None,
    sampler_params: Optional[Dict[str, Any]] = None,
    dataloader_params: Optional[Dict[str, Any]] = None,
    collator_params: Optional[Dict[str, Any]] = None,
    elim: bool = True,
):
    # Load manifest from cache for this dataset.
    manifest = get_manifest(
        universal_path(cache_dir) / manifest_name,
        data_dir,
        dataset_params,
        sampler_params,
        dataloader_params,
        collator_params=collator_params,
    )

    class_dist = utils.modes_from_manifest(manifest, classes)

    if elim:
        # Finds the empty classes and returns modified classes, a dict to convert between the old and new systems
        # and new colours.
        new_classes, class_matrix, new_colours = utils.eliminate_classes(
            utils.find_empty_classes(class_dist, classes),
            classes,
            cmap_dict,
        )
    else:
        new_classes = classes
        class_matrix = {}
        new_colours = cmap_dict

    return new_classes, class_matrix, new_colours, class_dist


def get_manifest(
    manifest_path: Union[str, Path],
    data_dir: Optional[Union[str, Path]] = None,
    dataset_params: Optional[Dict[str, Any]] = None,
    sampler_params: Optional[Dict[str, Any]] = None,
    loader_params: Optional[Dict[str, Any]] = None,
    collator_params: Optional[Dict[str, Any]] = None,
) -> DataFrame:
    """Attempts to return the :class:`~pandas.DataFrame` located at ``manifest_path``.

    If a ``csv`` file is not found at ``manifest_path``, the manifest is constructed from
    the parameters specified in the experiment :data:`CONFIG`.

    .. warning::
        As your :data:`CONFIG` is likely not setup for constructing manifests, not providing
        a valid ``manifest_path`` to an existing manifest is likely to result in an error
        in trying to construct the missing manifest.

        It is therefore recommended that you construct the missing manifest by parsing
        an appropriate manifest config to :func:`make_manifest` to get the manifest and
        save it to ``manifest_path``.

    Args:
        manifest_path (str | ~pathlib.Path): Path (including filename and extension) to the manifest
            saved as a ``csv``.
        task_name (str): Optional; Name of the task to which the dataset to create a manifest of belongs to.

    Returns:
        ~pandas.DataFrame: Manifest either loaded from ``manifest_path`` or created from parameters in :data:`CONFIG`.
    """
    manifest_path = universal_path(manifest_path).with_suffix(".csv")
    try:
        return pd.read_csv(manifest_path)
    except FileNotFoundError as err:
        print(err)

        print("CONSTRUCTING MISSING MANIFEST")
        assert data_dir
        assert dataset_params
        assert sampler_params
        assert loader_params

        manifest = make_manifest(
            data_dir,
            dataset_params,
            sampler_params,
            loader_params,
            collator_params=collator_params,
        )

        print(f"MANIFEST TO FILE -----> {manifest_path}")
        path = manifest_path.parent
        if not path.exists():
            os.makedirs(path)

        manifest.to_csv(manifest_path)

        return manifest


def make_manifest(
    data_dir: Union[str, Path],
    dataset_params: Dict[str, Any],
    sampler_params: Dict[str, Any],
    loader_params: Dict[str, Any],
    collator_params: Optional[Dict[str, Any]] = None,
) -> DataFrame:
    """Constructs a manifest of the dataset detailing each sample therein.

    The dataset to construct a manifest of is defined by the ``data_config`` value in the config.

    Args:
        mf_config (dict[~typing.Any, ~typing.Any]): Config to use to construct the manifest with.
        task_name (str): Optional; Name of the task to which the dataset to create a manifest of belongs to.

    Returns:
        ~pandas.DataFrame: The completed manifest as a :class:`~pandas.DataFrame`.
    """

    def delete_transforms(params: Dict[str, Any]) -> None:
        assert target_key is not None
        if params[target_key] is None:
            return

        # Delete class transforms.
        if "transforms" in params[target_key]:
            if isinstance(params[target_key]["transforms"], dict):
                if "ClassTransform" in params[target_key]["transforms"]:
                    del params[target_key]["transforms"]["ClassTransform"]

        # Delete the transforms for both imagery and targets.
        # This assumes that it is geometric transforms and therefore distort the actual dataset composition.
        if "transforms" in params:
            del params["transforms"]

    _sampler_params = deepcopy(sampler_params)
    if OmegaConf.is_config(_sampler_params):
        _sampler_params = OmegaConf.to_object(_sampler_params)  # type: ignore[assignment]

    _dataset_params = deepcopy(dataset_params)
    if OmegaConf.is_config(_dataset_params):
        _dataset_params = OmegaConf.to_object(_dataset_params)  # type: ignore[assignment]

    if _sampler_params["_target_"].rpartition(".")[2] in (
        "RandomGeoSampler",
        "RandomPairGeoSampler",
        "RandomBatchGeoSampler",
        "RandomPairBatchGeoSampler",
    ):
        _sampler_params["_target_"] = "torchgeo.samplers.GridGeoSampler"
        _sampler_params["stride"] = [
            0.9 * x for x in _to_tuple(_sampler_params["size"])
        ]

    if _sampler_params["_target_"].rpartition(".")[2] == "RandomSampler":
        _sampler_params = {"_target_": "torch.utils.data.sampler.SequentialSampler"}

    if "length" in _sampler_params:
        del _sampler_params["length"]

    # Ensure there are no errant `ClassTransform` transforms in the parameters from previous runs.
    # A `ClassTransform` can only be defined with a correct manifest so we cannot use an old one to
    # sample the dataset. We need the original, un-transformed labels.
    target_key = None
    if "mask" in dataset_params or "label" in _dataset_params:
        target_key = masks_or_labels(_dataset_params)
        delete_transforms(_dataset_params)

    # There must be targets to construct a manifest of classes!
    assert target_key is not None

    print("CONSTRUCTING DATASET")

    loader = construct_dataloader(
        data_dir,
        _dataset_params,
        _sampler_params,
        loader_params,
        batch_size=1,  # To prevent issues with stacking different sized patches, set batch size to 1.
        collator_params=collator_params,
        cache=False,
    )

    print("FETCHING SAMPLES")
    df = DataFrame()

    modes = load_all_samples(loader, target_key=target_key)  # type: ignore[arg-type]

    df["MODES"] = [np.array([]) for _ in modes]

    for i, mode in enumerate(modes):
        df.loc[i, "MODES"] = mode

    print("CALCULATING CLASS FRACTIONS")
    # Calculates the fractional size of each class in each patch.
    df = DataFrame([row for row in df.apply(utils.class_frac, axis=1)])  # type: ignore[arg-type]
    df.fillna(0, inplace=True)

    # Delete redundant MODES column.
    del df["MODES"]

    return df
