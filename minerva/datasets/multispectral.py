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
r"""Adaption of :class:`~torchvision.datasets.VisionDataset` for use with :class:`~torchgeo.datasets.NonGeoDataset`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = ["Jonathon Hare", "Harry Baker"]
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = ["MultiSpectralDataset"]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import tifffile
import torch
from torchgeo.datasets import NonGeoDataset
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import resize


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MultiSpectralDataset(VisionDataset, NonGeoDataset):
    """Generic dataset class for multi-spectral images that works within :mod:`torchgeo`"""

    all_bands = []
    rgb_bands = []

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        bands: Optional[List[str]] = None,
    ) -> None:
        super().__init__(root, transform=transforms, target_transform=None)

        if bands is None:
            bands = self.all_bands

        self.loader = partial(tifffile.imread, key=0)
        self.bands = bands
        self.samples = self.make_dataset()

    def make_dataset(self) -> List[str]:
        directory = os.path.expanduser(self.root)

        dirs = set()
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                if fname == f"{self.bands[0]}.tif":
                    dirs.add(root)
        return sorted(list(dirs))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.samples[index]

        images = []
        h, w = 0, 0
        for b in self.bands:
            img = torch.from_numpy(self.loader(f"{path}/{b}.tif").astype(np.float32))
            h = max(img.shape[0], h)
            w = max(img.shape[1], w)
            images.append(img.unsqueeze(0))

        for i in range(len(images)):
            images[i] = resize(images[i], [h, w], antialias=True)

        bands = torch.cat(images, dim=0)

        if self.transform is not None:
            bands = self.transform(bands)

        sample = {"image": bands}

        return sample

    def __len__(self) -> int:
        return len(self.samples)
