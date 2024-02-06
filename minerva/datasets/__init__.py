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
r"""Functionality for constructing datasets, samplers and :class:`~torch.utils.data.DataLoader` for :mod:`minerva`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = ["Harry Baker", "Jonathon Hare"]
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "MinervaNonGeoDataset",
    "MinervaConcatDataset",
    "PairedGeoDataset",
    "PairedNonGeoDataset",
    "PairedUnionDataset",
    "PairedConcatDataset",
    "GeoSSL4EOS12Sentinel2",
    "NonGeoSSL4EOS12Sentinel2",
    "NAIPChesapeakeCVPR",
    "DFC2020",
    "SEN12MS",
    "MultiSpectralDataset",
    "construct_dataloader",
    "get_collator",
    "get_manifest",
    "load_all_samples",
    "make_bounding_box",
    "make_dataset",
    "make_loaders",
    "make_manifest",
    "stack_sample_pairs",
    "intersect_datasets",
    "unionise_datasets",
    "get_manifest_path",
    "get_random_sample",
]

from .collators import get_collator, stack_sample_pairs
from .dfc import DFC2020, SEN12MS
from .factory import (
    construct_dataloader,
    get_manifest,
    get_manifest_path,
    make_dataset,
    make_loaders,
    make_manifest,
)
from .multispectral import MultiSpectralDataset
from .naip import NAIPChesapeakeCVPR
from .paired import (
    PairedConcatDataset,
    PairedGeoDataset,
    PairedNonGeoDataset,
    PairedUnionDataset,
)
from .ssl4eos12 import GeoSSL4EOS12Sentinel2, NonGeoSSL4EOS12Sentinel2
from .utils import (
    MinervaConcatDataset,
    MinervaNonGeoDataset,
    get_random_sample,
    intersect_datasets,
    load_all_samples,
    make_bounding_box,
    unionise_datasets,
)
