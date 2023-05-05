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
"""Functionality for constructing datasets, samplers and :class:`~torch.utils.data.DataLoader` for :mod:`minerva`.

Attributes:
    IMAGERY_CONFIG (dict[str, ~typing.Any]): Config defining the properties of the imagery used in the experiment.
    CACHE_DIR (~pathlib.Path): Path to the cache directory used to store dataset manifests, cached model weights etc.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU LGPLv3"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "PairedDataset",
    "PairedUnionDataset",
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
    "unionise_datasets",
    "get_manifest_path",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import inspect
import os
import platform
import re
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from alive_progress import alive_it
from catalyst.data.sampler import DistributedSamplerWrapper
from matplotlib.figure import Figure
from nptyping import NDArray
from pandas import DataFrame
from torch.utils.data import DataLoader
from torchgeo.datasets import (
    GeoDataset,
    IntersectionDataset,
    RasterDataset,
    UnionDataset,
)
from torchgeo.datasets.utils import BoundingBox, concat_samples, stack_samples
from torchgeo.samplers import BatchGeoSampler, GeoSampler
from torchgeo.samplers.utils import get_random_bounding_box
from torchvision.transforms import RandomApply

from minerva.transforms import MinervaCompose
from minerva.utils import AUX_CONFIGS, CONFIG, universal_path, utils

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
IMAGERY_CONFIG: Dict[str, Any] = AUX_CONFIGS["imagery_config"]

# Path to cache directory.
CACHE_DIR: Path = universal_path(CONFIG["dir"]["cache"])


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class TstImgDataset(RasterDataset):
    """Test dataset for imagery.

    Attributes:
        filename_glob (str): Pattern for image tiff files within dataset root to construct dataset from.
    """

    filename_glob = "*_img.tif"


class TstMaskDataset(RasterDataset):
    """Test dataset for land cover data.

    Attributes:
        filename_glob (str): Pattern for mask tiff files within dataset root to construct dataset from.
        is_image (bool): Sets flag to false to mark this as not a imagery dataset.
    """

    filename_glob = "*_lc.tif"
    is_image = False


class PairedDataset(RasterDataset):
    """Custom dataset to act as a wrapper to other datasets to handle paired sampling.

    Attributes:
        dataset (~torchgeo.datasets.RasterDataset): Wrapped dataset to sampled from.

    Args:
        dataset_cls (~typing.Callable[..., ~torchgeo.datasets.GeoDataset]): Constructor for a
            :class:`~torchgeo.datasets.RasterDataset` to be wrapped for paired sampling.
    """

    def __init__(
        self,
        dataset_cls: Callable[..., GeoDataset],
        *args,
        **kwargs,
    ) -> None:
        super_sig = inspect.signature(RasterDataset.__init__).parameters.values()
        super_kwargs = {
            key.name: kwargs[key.name] for key in super_sig if key.name in kwargs
        }

        super().__init__(*args, **super_kwargs)
        self.dataset = dataset_cls(*args, **kwargs)

    def __getitem__(  # type: ignore[override]
        self, queries: Tuple[BoundingBox, BoundingBox]
    ) -> Tuple[Dict[str, Any], ...]:
        return self.dataset.__getitem__(queries[0]), self.dataset.__getitem__(
            queries[1]
        )

    def __or__(self, other: GeoDataset) -> "PairedUnionDataset":
        """Take the union of two GeoDatasets, extended for :class:`PairedDataset`.

        Args:
            other (~torchgeo.datasets.GeoDataset): Another dataset.

        Returns:
            PairedUnionDataset: A single dataset.

        Raises:
            ValueError: If ``other`` is not a :class:`GeoDataset`
        """
        return PairedUnionDataset(self, other)

    def __getattr__(self, item):
        if item in self.dataset.__dict__:
            return getattr(self.dataset, item)  # pragma: no cover
        elif item in self.__dict__:
            return getattr(self, item)
        else:
            raise AttributeError

    def __repr__(self) -> Any:
        return self.dataset.__repr__()

    @staticmethod
    def plot(
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plots a sample from the dataset.

        Adapted from :meth:`torchgeo.datasets.NAIP.plot`.

        Args:
            sample (dict[str, ~typing.Any]): Sample to plot.
            show_titles (bool): Optional; Add title to the figure. Defaults to True.
            suptitle (str): Optional; Super title to add to figure. Defaults to None.

        Returns:
            ~matplotlib.figure.Figure: :mod:`matplotlib` Figure object with plot of the random patch imagery.
        """

        image = sample["image"][0:3, :, :].permute(1, 2, 0)

        # Setup the figure.
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        # Plot the image.
        ax.imshow(image)

        # Turn the axis off.
        ax.axis("off")

        # Add title to figure.
        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

    def plot_random_sample(
        self,
        size: Union[Tuple[int, int], int],
        res: float,
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plots a random sample the dataset at a given size and resolution.

        Adapted from :meth:`torchgeo.datasets.NAIP.plot`.

        Args:
            size (tuple[int, int] | int): Size of the patch to plot.
            res (float): Resolution of the patch.
            show_titles (bool): Optional; Add title to the figure. Defaults to ``True``.
            suptitle (str): Optional; Super title to add to figure. Defaults to ``None``.

        Returns:
            ~matplotlib.figure.Figure: :mod:`matplotlib` Figure object with plot of the random patch imagery.
        """

        # Get a random sample from the dataset at the given size and resolution.
        sample = get_random_sample(self.dataset, size, res)
        return self.plot(sample, show_titles, suptitle)


class PairedUnionDataset(UnionDataset):
    """Adapted form of :class:`~torchgeo.datasets.UnionDataset` to handle paired samples.

    ..warning::

        Do not use with :class:`PairedDataset` as this will essentially account for paired sampling twice
        and cause a :class:`TypeError`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        new_datasets = []
        for _dataset in self.datasets:
            if isinstance(_dataset, PairedDataset):
                new_datasets.append(_dataset.dataset)

        self.datasets = new_datasets

    def __getitem__(
        self, query: Tuple[BoundingBox, BoundingBox]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Retrieve image and metadata indexed by query.

        Uses :meth:`torchgeo.datasets.UnionDataset.__getitem__` to send each query of the pair off to get a
        sample for each and returns as a tuple.

        Args:
            query (tuple[~torchgeo.datasets.utils.BoundingBox, ~torchgeo.datasets.utils.BoundingBox]): Coordinates
                to index in the form (minx, maxx, miny, maxy, mint, maxt).

        Returns:
            tuple[dict[str, ~typing.Any], dict[str, ~typing.Any]]: Sample of data/labels and metadata at that index.
        """
        return super().__getitem__(query[0]), super().__getitem__(query[1])


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_collator(
    collator_params: Optional[Dict[str, str]] = None
) -> Callable[..., Any]:
    """Gets the function defined in parameters to collate samples together to form a batch.

    Args:
        collator_params (dict[str, str]): Optional; Dictionary that must contain keys for
            ``'module'`` and ``'name'`` of the collation function. Defaults to ``config['collator']``.

    Returns:
        ~typing.Callable[..., ~typing.Any]: Collation function found from parameters given.
    """
    collator: Callable[..., Any]
    if collator_params is not None:
        module = collator_params.pop("module", "")
        if module == "":
            collator = globals()[collator_params["name"]]
        else:
            collator = utils.func_by_str(module, collator_params["name"])
    else:
        collator = stack_samples

    assert callable(collator)
    return collator


def stack_sample_pairs(
    samples: Iterable[Tuple[Dict[Any, Any], Dict[Any, Any]]]
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """Takes a list of paired sample dicts and stacks them into a tuple of batches of sample dicts.

    Args:
        samples (~typing.Iterable[tuple[dict[~typing.Any, ~typing.Any], dict[~typing.Any, ~typing.Any]]]): List of
            paired sample dicts to be stacked.

    Returns:
        tuple[dict[~typing.Any, ~typing.Any], dict[~typing.Any, ~typing.Any]]: Tuple of batches within dicts.
    """
    a, b = tuple(zip(*samples))
    return stack_samples(a), stack_samples(b)


def intersect_datasets(
    datasets: Sequence[GeoDataset], sample_pairs: bool = False
) -> IntersectionDataset:
    r"""
    Intersects a list of :class:`~torchgeo.datasets.GeoDataset` together to return a single dataset object.

    Args:
        datasets (list[~torchgeo.datasets.GeoDataset]): List of datasets to intersect together.
            Should have some geospatial overlap.
        sample_pairs (bool): Optional; True if paired sampling. This will wrap the collation function
            for paired samples.

    Returns:
        ~torchgeo.datasets.IntersectionDataset: Final dataset object representing an intersection
        of all the parsed datasets.
    """

    def intersect_pair_datasets(a: GeoDataset, b: GeoDataset) -> IntersectionDataset:
        if sample_pairs:
            return IntersectionDataset(
                a, b, collate_fn=utils.pair_collate(concat_samples)
            )
        else:
            return a & b

    master_dataset: Union[GeoDataset, IntersectionDataset] = datasets[0]

    for i in range(len(datasets) - 1):
        master_dataset = intersect_pair_datasets(master_dataset, datasets[i + 1])

    assert isinstance(master_dataset, IntersectionDataset)
    return master_dataset


def unionise_datasets(datasets: Sequence[GeoDataset]) -> UnionDataset:
    """Unionises a list of :class:`~torchgeo.datasets.GeoDataset` together to return a single dataset object.

    Args:
        datasets (list[~torchgeo.datasets.GeoDataset]): List of datasets to unionise together.

    Returns:
        ~torchgeo.datasets.UnionDataset: Final dataset object representing an union of all the parsed datasets.
    """
    master_dataset: Union[GeoDataset, UnionDataset] = datasets[0]

    for i in range(len(datasets) - 1):
        master_dataset = master_dataset | datasets[i + 1]

    assert isinstance(master_dataset, UnionDataset)
    return master_dataset


def make_dataset(
    data_directory: Union[Iterable[str], str, Path],
    dataset_params: Dict[Any, Any],
    transform_params: Optional[Dict[Any, Any]] = None,
    sample_pairs: bool = False,
) -> Tuple[Any, List[Any]]:
    """Constructs a dataset object from ``n`` sub-datasets given by the parameters supplied.

    Args:
        data_directory (~typing.Iterable[str] | str | ~pathlib.Path]): List defining the path to the directory
            containing the data.
        dataset_params (dict[~typing.Any, ~typing.Any]): Dictionary of parameters defining each sub-datasets to be used.
        transform_params: Optional; Dictionary defining the parameters of the transforms to perform
            when sampling from the dataset.
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

        # Construct the root to the sub-dataset's files.
        sub_dataset_root = str(
            universal_path(data_directory) / sub_dataset_params["root"]
        )

        return _sub_dataset, sub_dataset_root

    def create_transforms(
        this_transform_params: Any, key: str, data_type_key: Optional[str] = None
    ) -> Optional[Any]:
        """Construct transforms for samples returned from this sub-dataset -- if found.

        Args:
            this_transform_params (~typing.Any): Parameters defining the transforms for the dataset for the
                whole mode of fitting.
            key (str): The key for the transforms for this particular subdataset.
            data_type_key (str): Optional; The type of data the transform is acting on.
                Most likely ``"image"`` or ``"mask"``. This may differ from ``key`` if using unionisation of datasets.
                If ``None``, defaults to ``key``.

        Returns:
            ~typing.Any | None: The transformatins for this subdataset or ``None`` if no parameters found.
        """
        data_type_key = key if data_type_key is None else data_type_key

        _transformations: Optional[Any] = None
        if type(this_transform_params) == dict:
            assert this_transform_params is not None
            try:
                if this_transform_params[key]:
                    _transformations = make_transformations(
                        this_transform_params[key], key=data_type_key
                    )
            except (KeyError, TypeError):
                pass
        else:
            pass

        return _transformations

    def create_subdataset(
        dataset_class: Callable[..., GeoDataset],
        root: str,
        subdataset_params: Dict[Literal["params"], Dict[str, Any]],
        _transformations: Optional[Any],
    ) -> GeoDataset:
        if sample_pairs:
            return PairedDataset(
                dataset_class,
                root=root,
                transforms=_transformations,
                **subdataset_params["params"],
            )
        else:
            return dataset_class(
                root=root,
                transforms=_transformations,
                **subdataset_params["params"],
            )

    # --+ MAKE SUB-DATASETS +=========================================================================================+
    # List to hold all the sub-datasets defined by dataset_params to be intersected together into a single dataset.
    sub_datasets: List[GeoDataset] = []

    # Iterate through all the sub-datasets defined in `dataset_params`.
    for type_key in dataset_params.keys():
        type_dataset_params = dataset_params[type_key]

        type_subdatasets = []

        multi_datasets_exist = False
        for area_key in type_dataset_params.keys():
            if area_key in ("module", "name", "params", "root"):
                multi_datasets_exist = False
                continue
            else:
                multi_datasets_exist = True
                _subdataset, subdataset_root = get_subdataset(
                    type_dataset_params, area_key
                )
                transformations: Optional[Any] = None
                try:
                    assert transform_params
                    if transform_params[type_key]:
                        transformations = create_transforms(
                            transform_params[type_key],
                            area_key,
                            type_key,
                        )
                except (KeyError, TypeError, AssertionError):
                    pass

                type_subdatasets.append(
                    create_subdataset(
                        _subdataset,
                        subdataset_root,
                        type_dataset_params[area_key],
                        transformations,
                    )
                )

        if multi_datasets_exist:
            sub_datasets.append(unionise_datasets(type_subdatasets, sample_pairs))
        else:
            sub_datasets.append(
                create_subdataset(
                    *get_subdataset(dataset_params, type_key),
                    type_dataset_params,
                    create_transforms(transform_params, type_key),
                )
            )

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
    batch_size: int,
    collator_params: Optional[Dict[str, Any]] = None,
    transform_params: Optional[Dict[str, Any]] = None,
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
        transform_params (dict[str, ~typing.Any]): Optional; Dictionary defining the parameters of the transforms
            to perform when sampling from the dataset.
        rank (int): Optional; The rank of this process for distributed computing.
        world_size (int): Optional; The total number of processes within a distributed run.
        sample_pairs (bool): Optional; True if paired sampling. This will wrap the collation function
            for paired samples.

    Returns:
        ~torch.utils.data.DataLoader: Object to handle the returning of batched samples from the dataset.
    """
    dataset, subdatasets = make_dataset(
        data_directory, dataset_params, transform_params, sample_pairs=sample_pairs
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


def make_bounding_box(
    roi: Union[Sequence[float], bool] = False
) -> Optional[BoundingBox]:
    """Construct a :class:`~torchgeo.datasets.utils.BoundingBox` object from the corners of the box.
    ``False`` for no :class:`~torchgeo.datasets.utils.BoundingBox`.

    Args:
        roi (~typing.Sequence[float] | bool): Either a :class:`tuple` or array of values defining
            the corners of a bounding box or False to designate no BoundingBox is defined.

    Returns:
        ~torchgeo.datasets.utils.BoundingBox | None: Bounding box made from parsed values
        or ``None`` if ``False`` was given.
    """
    if roi is False:
        return None
    elif roi is True:
        raise ValueError(
            "``roi`` must be a sequence of floats or ``False``, not ``True``"
        )
    else:
        return BoundingBox(*roi)


def get_transform(name: str, transform_params: Dict[str, Any]) -> Callable[..., Any]:
    """Creates a transform object based on config parameters.

    Args:
        name (str): Name of transform object to import e.g :class:`~torchvision.transforms.RandomResizedCrop`.
        transform_params (dict[str, ~typing.Any]): Arguements to construct transform with.
            Should also include ``"module"`` key defining the import path to the transform object.

    Returns:
        Initialised transform object specified by config parameters.

    .. note::
        If ``transform_params`` contains no ``"module"`` key, it defaults to ``torchvision.transforms``.

    Example:
        >>> name = "RandomResizedCrop"
        >>> params = {"module": "torchvision.transforms", "size": 128}
        >>> transform = get_transform(name, params)

    Raises:
        TypeError: If created transform object is itself not :class:`~typing.Callable`.
    """
    params = transform_params.copy()
    module = params.pop("module", "torchvision.transforms")

    # Gets the transform requested by config parameters.
    _transform: Callable[..., Any] = utils.func_by_str(module, name)

    transform: Callable[..., Any] = _transform(**params)
    if callable(transform):
        return transform
    else:
        raise TypeError(f"Transform has type {type(transform)}, not a callable!")


def _construct_random_transforms(random_params: Dict[str, Any]) -> Any:
    p = random_params.pop("p", 0.5)

    random_transforms = []
    for ran_name in random_params:
        random_transforms.append(get_transform(ran_name, random_params[ran_name]))

    return RandomApply(random_transforms, p=p)


def _manual_compose(
    manual_params: Dict[str, Any],
    key: str,
    other_transforms: Optional[List[Any]] = None,
) -> MinervaCompose:
    manual_transforms = []

    for manual_name in manual_params:
        manual_transforms.append(get_transform(manual_name, manual_params[manual_name]))

    if other_transforms:
        manual_transforms = manual_transforms + other_transforms

    return MinervaCompose(manual_transforms, key=key)


def make_transformations(
    transform_params: Union[Dict[str, Any], Literal[False]], key: Optional[str] = None
) -> Optional[Any]:
    """Constructs a transform or series of transforms based on parameters provided.

    Args:
        transform_params (dict[str, ~typing.Any] | ~typing.Literal[False]): Parameters defining transforms desired.
            The name of each transform should be the key, while the kwargs for the transform should
            be the value of that key as a dict.
        key (str): Optional; Key of the type of data within the sample to be transformed.
            Must be ``"image"`` or ``"mask"``.

    Example:
        >>> transform_params = {
        >>>    "CenterCrop": {"module": "torchvision.transforms", "size": 128},
        >>>     "RandomHorizontalFlip": {"module": "torchvision.transforms", "p": 0.7}
        >>> }
        >>> transforms = make_transformations(transform_params)

    Returns:
        If no parameters are parsed, None is returned.
        If only one transform is defined by the parameters, returns a Transforms object.
        If multiple transforms are defined, a Compose object of Transform objects is returned.
    """
    transformations = []

    # If no transforms are specified, return None.
    if not transform_params:
        return None

    manual_compose = False

    # Get each transform.
    for name in transform_params:
        if name == "MinervaCompose":
            manual_compose = True

        elif name == "RandomApply":
            random_params = transform_params[name].copy()
            transformations.append(_construct_random_transforms(random_params))

        else:
            transformations.append(get_transform(name, transform_params[name]))

    # Compose transforms together and return.
    if manual_compose:
        assert key is not None
        return _manual_compose(
            transform_params["MinervaCompose"].copy(),
            key=key,
            other_transforms=transformations,
        )
    else:
        return MinervaCompose(transformations, key)


@utils.return_updated_kwargs
def make_loaders(
    rank: int = 0,
    world_size: int = 1,
    p_dist: bool = False,
    **params,
) -> Tuple[
    Dict[str, DataLoader[Iterable[Any]]],
    Dict[str, int],
    List[Tuple[int, int]],
    Dict[Any, Any],
]:
    """Constructs train, validation and test datasets and places into :class:`~torch.utils.data.DataLoader` objects.

    Args:
        rank (int): Rank number of the process. For use with :class:`~torch.nn.parallel.DistributedDataParallel`.
        world_size (int): Total number of processes across all nodes. For use with
            :class:`~torch.nn.parallel.DistributedDataParallel`.
        p_dist (bool): Print to screen the distribution of classes within each dataset.

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
    # Gets out the parameters for the DataLoaders from params.
    dataloader_params: Dict[Any, Any] = params["loader_params"]
    dataset_params: Dict[str, Any] = params["dataset_params"]
    sampler_params: Dict[str, Any] = params["sampler_params"]
    transform_params: Dict[str, Any] = params["transform_params"]
    batch_size: int = params["batch_size"]

    model_type = params["model_type"]
    class_dist: List[Tuple[int, int]] = [(0, 0)]

    new_classes: Dict[int, str] = {}
    new_colours: Dict[int, str] = {}
    forwards: Dict[int, int] = {}

    sample_pairs: Union[bool, Any] = params.get("sample_pairs", False)
    if type(sample_pairs) != bool:
        sample_pairs = False

    if model_type != "siamese":
        # Load manifest from cache for this dataset.
        manifest = get_manifest(get_manifest_path())
        class_dist = utils.modes_from_manifest(manifest)

        # Finds the empty classes and returns modified classes, a dict to convert between the old and new systems
        # and new colours.
        new_classes, forwards, new_colours = utils.load_data_specs(
            class_dist=class_dist, elim=params.get("elim", False)
        )

    # Inits dicts to hold the variables and lists for train, validation and test.
    n_batches = {}
    loaders = {}

    for mode in dataset_params.keys():
        this_transform_params = transform_params[mode]
        if params.get("elim", False) and model_type != "siamese":
            if type(this_transform_params["mask"]) != dict:
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
            batch_size,
            collator_params=params["collator"],
            transform_params=this_transform_params,
            rank=rank,
            world_size=world_size,
            sample_pairs=sample_pairs if mode == "train" else False,
        )
        print("DONE")

    if model_type != "siamese":
        # Transform class dist if elimination of classes has occurred.
        if params.get("elim", False):
            class_dist = utils.class_dist_transform(class_dist, forwards)

        # Prints class distribution in a pretty text format using tabulate to stdout.
        if p_dist:
            utils.print_class_dist(class_dist)

        params["n_classes"] = len(new_classes)
        model_params_params = params["model_params"].get("params", {})
        model_params_params["n_classes"] = len(new_classes)
        params["model_params"]["params"] = model_params_params
        params["classes"] = new_classes
        params["colours"] = new_colours

    params["max_pixel_value"] = IMAGERY_CONFIG["data_specs"]["max_value"]

    return loaders, n_batches, class_dist, params


def get_manifest_path() -> str:
    """Gets the path to the manifest for the dataset to be used.

    Returns:
        str: Path to manifest as string.
    """
    return str(Path(CACHE_DIR, f"{utils.get_dataset_name()}_Manifest.csv"))


def get_manifest(manifest_path: Union[str, Path]) -> DataFrame:
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

        mf_config["dataloader_params"] = CONFIG["loader_params"]

        manifest = make_manifest(mf_config)

        print(f"MANIFEST TO FILE -----> {manifest_path}")
        path = manifest_path.parent
        if not path.exists():
            os.makedirs(path)

        manifest.to_csv(manifest_path)

        return manifest


def make_manifest(mf_config: Dict[Any, Any]) -> DataFrame:
    """Constructs a manifest of the dataset detailing each sample therein.

    The dataset to construct a manifest of is defined by the ``data_config`` value in the config.

    Args:
        mf_config (dict[~typing.Any, ~typing.Any]): Config to use to construct the manifest with.

    Returns:
        ~pandas.DataFrame: The completed manifest as a :class:`~pandas.DataFrame`.
    """
    batch_size = mf_config["batch_size"]
    dataloader_params = mf_config["dataloader_params"]
    dataset_params = mf_config["dataset_params"]
    sampler_params = mf_config["sampler_params"]
    collator_params = mf_config["collator"]

    keys = list(dataset_params.keys())
    print("CONSTRUCTING DATASET")
    loader = construct_dataloader(
        mf_config["dir"]["data"],
        dataset_params[keys[0]],
        sampler_params[keys[0]],
        dataloader_params,
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


def load_all_samples(dataloader: DataLoader[Iterable[Any]]) -> NDArray[Any, Any]:
    """Loads all sample masks from parsed :class:`~torch.utils.data.DataLoader` and computes the modes of their classes.

    Args:
        dataloader (~torch.utils.data.DataLoader): DataLoader containing samples. Must be using a dataset with
            ``__len__`` attribute and a sampler that returns a dict with a ``"mask"`` key.

    Returns:
        ~numpy.ndarray: 2D array of the class modes within every sample defined by the parsed
        :class:`~torch.utils.data.DataLoader`.
    """
    sample_modes: List[List[Tuple[int, int]]] = []
    for sample in alive_it(dataloader):
        modes = utils.find_modes(sample["mask"])
        sample_modes.append(modes)

    return np.array(sample_modes, dtype=object)


def get_random_sample(
    dataset: GeoDataset, size: Union[Tuple[int, int], int], res: float
) -> Dict[str, Any]:
    """Gets a random sample from the provided dataset of size ``size`` and at ``res`` resolution.

    Args:
        dataset (~torchgeo.datasets.GeoDataset): Dataset to sample from.
        size (tuple[int, int] | int): Size of the patch to sample.
        res (float): Resolution of the patch.

    Returns:
        dict[str, ~typing.Any]: Random sample from the dataset.
    """
    return dataset[get_random_bounding_box(dataset.bounds, size, res)]
