# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2023 Harry Baker

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

Attributes:
    IMAGERY_CONFIG (dict[str, ~typing.Any]): Config defining the properties of the imagery used in the experiment.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "construct_dataloader",
    "make_dataset",
    "make_loaders",
    "get_manifest_path",
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
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from catalyst.data.sampler import DistributedSamplerWrapper
from pandas import DataFrame
from rasterio.crs import CRS
from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, RasterDataset
from torchgeo.samplers import BatchGeoSampler, GeoSampler

from minerva.transforms import init_auto_norm, make_transformations
from minerva.utils import AUX_CONFIGS, CONFIG, universal_path, utils

from .collators import get_collator, stack_sample_pairs
from .paired import PairedDataset
from .utils import (
    intersect_datasets,
    load_all_samples,
    make_bounding_box,
    unionise_datasets,
)

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
IMAGERY_CONFIG: Dict[str, Any] = AUX_CONFIGS["imagery_config"]

# Path to cache directory.
CACHE_DIR: Path = universal_path(CONFIG["dir"]["cache"])


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def make_dataset(
    data_directory: Union[Iterable[str], str, Path],
    dataset_params: Dict[Any, Any],
    sample_pairs: bool = False,
) -> Tuple[Any, List[Any]]:
    """Constructs a dataset object from ``n`` sub-datasets given by the parameters supplied.

    Args:
        data_directory (~typing.Iterable[str] | str | ~pathlib.Path]): List defining the path to the directory
            containing the data.
        dataset_params (dict[~typing.Any, ~typing.Any]): Dictionary of parameters defining each sub-datasets to be used.
        sample_pairs (bool): Optional; ``True`` if paired sampling. This will ensure paired samples are handled
            correctly in the datasets.

    Returns:
        tuple[~typing.Any, list[~typing.Any]]: Tuple of Dataset object formed by the parameters given and list of
        the sub-datasets created that constitute ``dataset``.
    """

    def get_subdataset(
        this_dataset_params: Dict[str, Any], key: str
    ) -> Tuple[Callable[..., GeoDataset], str]:
        # Get the params for this sub-dataset.
        sub_dataset_params = this_dataset_params[key]

        # Get the constructor for the class of dataset defined in params.
        _sub_dataset: Callable[..., GeoDataset] = utils.func_by_str(
            module_path=sub_dataset_params["module"], func=sub_dataset_params["name"]
        )

        # Construct the paths to the sub-dataset's files.
        sub_dataset_path: Path = (
            universal_path(data_directory) / sub_dataset_params["paths"]
        )
        sub_dataset_path = sub_dataset_path.absolute()

        return _sub_dataset, str(sub_dataset_path)

    def create_subdataset(
        dataset_class: Callable[..., GeoDataset],
        paths: str,
        subdataset_params: Dict[Literal["params"], Dict[str, Any]],
        _transformations: Optional[Any],
    ) -> GeoDataset:
        copy_params = deepcopy(subdataset_params)

        if "crs" in copy_params["params"]:
            copy_params["params"]["crs"] = CRS.from_epsg(copy_params["params"]["crs"])

        if sample_pairs:
            return PairedDataset(
                dataset_class,
                paths,
                transforms=_transformations,
                **copy_params["params"],
            )
        else:
            return dataset_class(
                paths,
                transforms=_transformations,
                **copy_params["params"],
            )

    # --+ MAKE SUB-DATASETS +=========================================================================================+
    # List to hold all the sub-datasets defined by dataset_params to be intersected together into a single dataset.
    sub_datasets: List[GeoDataset] = []

    # Iterate through all the sub-datasets defined in `dataset_params`.
    for type_key in dataset_params.keys():
        if type_key == "sampler":
            continue
        type_dataset_params = dataset_params[type_key]

        type_subdatasets = []

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
                else:
                    transform_params = False

                master_transforms = make_transformations(transform_params, type_key)

            # Assuming that these keys are names of datasets.
            else:
                multi_datasets_exist = True

                _subdataset, subdataset_root = get_subdataset(
                    type_dataset_params, area_key
                )

                if isinstance(type_dataset_params[area_key].get("transforms"), dict):
                    transform_params = type_dataset_params[area_key]["transforms"]
                    auto_norm = transform_params.get("AutoNorm")
                else:
                    transform_params = False

                transformations = make_transformations(transform_params, type_key)

                # Send the params for this area key back through this function to make the sub-dataset.
                sub_dataset = create_subdataset(
                    _subdataset,
                    subdataset_root,
                    type_dataset_params[area_key],
                    transformations,
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

                type_subdatasets.append(sub_dataset)

        # Unionise all the sub-datsets of this modality together.
        if multi_datasets_exist:
            sub_datasets.append(unionise_datasets(type_subdatasets, master_transforms))

        # Add the subdataset of this modality to the list.
        else:
            sub_dataset = create_subdataset(
                *get_subdataset(dataset_params, type_key),
                type_dataset_params,
                master_transforms,
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

            sub_datasets.append(sub_dataset)

    # Intersect sub-datasets of differing modalities together to form single dataset
    # if more than one sub-dataset exists. Else, just set that to dataset.
    dataset = sub_datasets[0]
    if len(sub_datasets) > 1:
        dataset = intersect_datasets(sub_datasets)

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
        data_directory, dataset_params, sample_pairs=sample_pairs
    )

    # --+ MAKE SAMPLERS +=============================================================================================+
    _sampler: Callable[..., Union[BatchGeoSampler, GeoSampler]] = utils.func_by_str(
        module_path=sampler_params["module"], func=sampler_params["name"]
    )

    batch_sampler = True if re.search(r"Batch", sampler_params["name"]) else False
    if batch_sampler:
        sampler_params["params"]["batch_size"] = batch_size

        if dist.is_available() and dist.is_initialized():  # type: ignore[attr-defined]
            assert (
                sampler_params["params"]["batch_size"] % world_size == 0
            )  # pragma: no cover
            per_device_batch_size = (
                sampler_params["params"]["batch_size"] // world_size
            )  # pragma: no cover
            sampler_params["params"][
                "batch_size"
            ] = per_device_batch_size  # pragma: no cover

    sampler: Union[BatchGeoSampler, GeoSampler, DistributedSamplerWrapper] = _sampler(
        dataset=subdatasets[0],
        roi=make_bounding_box(sampler_params["roi"]),
        **sampler_params["params"],
    )

    # --+ MAKE DATALOADERS +==========================================================================================+
    collator = get_collator(collator_params)
    _dataloader_params = dataloader_params.copy()

    # Add batch size from top-level parameters to the dataloader parameters.
    _dataloader_params["batch_size"] = batch_size

    if world_size > 1:
        # Wraps sampler for distributed computing.
        sampler = DistributedSamplerWrapper(sampler, num_replicas=world_size, rank=rank)

        # Splits batch size across devices.
        assert batch_size % world_size == 0
        per_device_batch_size = batch_size // world_size
        _dataloader_params["batch_size"] = per_device_batch_size

    if sample_pairs:
        if not torch.cuda.device_count() > 1 and platform.system() != "Windows":
            collator = utils.pair_collate(collator)

        # Can't wrap functions in distributed runs due to pickling error.
        # This also seems to occur on Windows systems, even when not multiprocessing?
        # Therefore, the collator is set to `stack_sample_pairs` automatically.
        else:  # pragma: no cover
            collator = stack_sample_pairs

    if batch_sampler:
        _dataloader_params["batch_sampler"] = sampler
        del _dataloader_params["batch_size"]
    else:
        _dataloader_params["sampler"] = sampler

    return DataLoader(dataset, collate_fn=collator, **_dataloader_params)


@utils.return_updated_kwargs
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

    # Gets out the parameters for the DataLoaders from params.
    dataloader_params: Dict[Any, Any] = utils.fallback_params(
        "loader_params", task_params, params
    )
    dataset_params: Dict[str, Any] = utils.fallback_params(
        "dataset_params", task_params, params
    )
    batch_size: int = utils.fallback_params("batch_size", task_params, params)

    model_type = utils.fallback_params("model_type", task_params, params)
    class_dist: List[Tuple[int, int]] = [(0, 0)]

    new_classes: Dict[int, str] = {}
    new_colours: Dict[int, str] = {}
    forwards: Dict[int, int] = {}

    sample_pairs: Union[bool, Any] = utils.fallback_params(
        "sample_pairs", task_params, params, False
    )
    if not isinstance(sample_pairs, bool):
        sample_pairs = False

    elim = utils.fallback_params("elim", task_params, params, False)

    if not utils.check_substrings_in_string(model_type, "siamese"):
        # Load manifest from cache for this dataset.
        manifest = get_manifest(get_manifest_path(), task_name)
        class_dist = utils.modes_from_manifest(manifest)

        # Finds the empty classes and returns modified classes, a dict to convert between the old and new systems
        # and new colours.
        new_classes, forwards, new_colours = utils.load_data_specs(
            class_dist=class_dist,
            elim=elim,
        )

    n_batches: Union[Dict[str, int], int]
    loaders: Union[Dict[str, DataLoader[Iterable[Any]]], DataLoader[Iterable[Any]]]

    if "sampler" in dataset_params.keys():
        if elim and not utils.check_substrings_in_string(model_type, "siamese"):
            class_transform = {
                "ClassTransform": {
                    "module": "minerva.transforms",
                    "transform": forwards,
                }
            }

            if type(dataset_params["mask"].get("transforms")) != dict:
                dataset_params["mask"]["transforms"] = class_transform
            else:
                dataset_params["mask"]["transforms"][
                    "ClassTransform"
                ] = class_transform["ClassTransform"]

        sampler_params: Dict[str, Any] = dataset_params["sampler"]

        # Calculates number of batches.
        n_batches = int(sampler_params["params"]["length"] / batch_size)

        # --+ MAKE DATASETS +=========================================================================================+
        print(f"CREATING {task_name} DATASET")
        loaders = construct_dataloader(
            params["dir"]["data"],
            dataset_params,
            sampler_params,
            dataloader_params,
            batch_size,
            collator_params=utils.fallback_params("collator", task_params, params),
            rank=rank,
            world_size=world_size,
            sample_pairs=sample_pairs,
        )
        print("DONE")

    else:
        # Inits dicts to hold the variables and lists for train, validation and test.
        n_batches = {}
        loaders = {}

        for mode in dataset_params.keys():
            if elim and not utils.check_substrings_in_string(model_type, "siamese"):
                class_transform = {
                    "ClassTransform": {
                        "module": "minerva.transforms",
                        "transform": forwards,
                    }
                }

                if type(dataset_params[mode]["mask"].get("transforms")) != dict:
                    dataset_params[mode]["mask"]["transforms"] = class_transform
                else:
                    dataset_params[mode]["mask"]["transforms"][
                        "ClassTransform"
                    ] = class_transform["ClassTransform"]

            mode_sampler_params: Dict[str, Any] = dataset_params[mode]["sampler"]

            # Calculates number of batches.
            n_batches[mode] = int(mode_sampler_params["params"]["length"] / batch_size)

            # --+ MAKE DATASETS +=====================================================================================+
            print(f"CREATING {mode} DATASET")
            loaders[mode] = construct_dataloader(
                params["dir"]["data"],
                dataset_params[mode],
                mode_sampler_params,
                dataloader_params,
                batch_size,
                collator_params=utils.fallback_params("collator", task_params, params),
                rank=rank,
                world_size=world_size,
                sample_pairs=sample_pairs if mode == "train" else False,
            )
            print("DONE")

    if not utils.check_substrings_in_string(model_type, "siamese"):
        # Transform class dist if elimination of classes has occurred.
        if elim:
            class_dist = utils.class_dist_transform(class_dist, forwards)

        # Prints class distribution in a pretty text format using tabulate to stdout.
        if p_dist:
            utils.print_class_dist(class_dist)

        task_params["n_classes"] = len(new_classes)
        model_params = utils.fallback_params("model_params", task_params, params, {})
        model_params["n_classes"] = len(new_classes)

        if "model_params" in task_params:
            task_params["model_params"]["params"] = model_params
        else:
            task_params["model_params"] = {"params": model_params}

        task_params["classes"] = new_classes
        task_params["colours"] = new_colours

    task_params["max_pixel_value"] = IMAGERY_CONFIG["data_specs"]["max_value"]

    return loaders, n_batches, class_dist, task_params


def get_manifest_path() -> str:
    """Gets the path to the manifest for the dataset to be used.

    Returns:
        str: Path to manifest as string.
    """
    return str(Path(CACHE_DIR, f"{utils.get_dataset_name()}_Manifest.csv"))


def get_manifest(
    manifest_path: Union[str, Path], task_name: Optional[str] = None
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

    Returns:
        ~pandas.DataFrame: Manifest either loaded from ``manifest_path`` or created from parameters in :data:`CONFIG`.
    """
    manifest_path = Path(manifest_path)
    try:
        return pd.read_csv(manifest_path)
    except FileNotFoundError as err:
        print(err)

        print("CONSTRUCTING MISSING MANIFEST")
        mf_config = CONFIG.copy()

        manifest = make_manifest(mf_config, task_name)

        print(f"MANIFEST TO FILE -----> {manifest_path}")
        path = manifest_path.parent
        if not path.exists():
            os.makedirs(path)

        manifest.to_csv(manifest_path)

        return manifest


def make_manifest(
    mf_config: Dict[Any, Any], task_name: Optional[str] = None
) -> DataFrame:
    """Constructs a manifest of the dataset detailing each sample therein.

    The dataset to construct a manifest of is defined by the ``data_config`` value in the config.

    Args:
        mf_config (dict[~typing.Any, ~typing.Any]): Config to use to construct the manifest with.

    Returns:
        ~pandas.DataFrame: The completed manifest as a :class:`~pandas.DataFrame`.
    """

    def delete_class_transform(params: Dict[str, Any]) -> None:
        if "transforms" in params["mask"]:
            if isinstance(params["mask"]["transforms"], dict):
                if "ClassTransform" in params["mask"]["transforms"]:
                    del params["mask"]["transforms"]["ClassTransform"]

    task_params = mf_config
    if task_name is not None:
        task_params = mf_config["tasks"][task_name]

    batch_size: int = utils.fallback_params("batch_size", task_params, mf_config)
    loader_params: Dict[str, Any] = utils.fallback_params(
        "loader_params", task_params, mf_config
    )
    dataset_params: Dict[str, Any] = utils.fallback_params(
        "dataset_params", task_params, mf_config
    )
    collator_params: Dict[str, Any] = utils.fallback_params(
        "collator_params", task_params, mf_config
    )
    sampler_params = dataset_params["sampler"]

    # Ensure there are no errant `ClassTransform` transforms in the parameters from previous runs.
    # A `ClassTransform` can only be defined with a correct manifest so we cannot use an old one to
    # sample the dataset. We need the original, un-transformed labels.
    if "sampler" in dataset_params.keys():
        delete_class_transform(dataset_params)

        print("CONSTRUCTING DATASET")

        loader = construct_dataloader(
            mf_config["dir"]["data"],
            dataset_params,
            sampler_params,
            loader_params,
            batch_size,
            collator_params=collator_params,
        )
    else:
        for mode in dataset_params.keys():
            delete_class_transform(dataset_params[mode])

        keys = list(dataset_params.keys())
        sampler_params = dataset_params[keys[0]]["sampler"]

        print("CONSTRUCTING DATASET")

        loader = construct_dataloader(
            mf_config["dir"]["data"],
            dataset_params[keys[0]],
            sampler_params,
            loader_params,
            batch_size,
            collator_params=collator_params,
        )

    print("FETCHING SAMPLES")
    df = DataFrame()

    modes = load_all_samples(loader)

    df["MODES"] = [np.array([]) for _ in range(len(modes))]

    for i in range(len(modes)):
        df["MODES"][i] = modes[i]

    print("CALCULATING CLASS FRACTIONS")
    # Calculates the fractional size of each class in each patch.
    df = DataFrame([row for row in df.apply(utils.class_frac, axis=1)])  # type: ignore[arg-type]
    df.fillna(0, inplace=True)

    # Delete redundant MODES column.
    del df["MODES"]

    return df
