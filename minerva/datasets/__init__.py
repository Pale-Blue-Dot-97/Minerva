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
r"""Functionality for constructing datasets, samplers and :class:`~torch.utils.data.DataLoader` for :mod:`minerva`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "PairedDataset",
    "PairedUnionDataset",
    "SSL4EOS12Sentinel2",
    "NAIPChesapeakeCVPR",
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
    "get_random_sample",
]

from .collators import get_collator, stack_sample_pairs
from .factory import (
    construct_dataloader,
    get_manifest,
    get_manifest_path,
    make_dataset,
    make_loaders,
    make_manifest,
)
from .naip import NAIPChesapeakeCVPR as NAIPChesapeakeCVPR
from .paired import PairedDataset as PairedDataset
from .paired import PairedUnionDataset as PairedUnionDataset
from .ssl4eos12 import SSL4EOS12Sentinel2 as SSL4EOS12Sentinel2
from .utils import (
    get_random_sample,
    intersect_datasets,
    load_all_samples,
    make_bounding_box,
    unionise_datasets,
)
