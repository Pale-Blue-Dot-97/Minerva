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
r"""Implementation for the DFC2020 competition dataset using Sentinel 1&2 data and IGBP labels in :mod:`torchgeo`.

.. note::
    Adapted from https://github.com/lukasliebel/dfc2020_baseline/blob/master/code/datasets.py
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = ["Lukas Liebel", "Harry Baker"]
__contact__ = ["hjb1d20@soton.ac.uk"]
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = ["DFC2020"]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
import torch
from torch import FloatTensor, Tensor
from torchgeo.datasets import NonGeoDataset
from tqdm import tqdm


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class BaseSenS12MS(NonGeoDataset):
    # Mapping from IGBP to DFC2020 classes.
    DFC2020_CLASSES = [
        0,  # Class 0 unused in both schemes.
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        3,  # --> will be masked if no_savanna == True
        3,  # --> will be masked if no_savanna == True
        4,
        5,
        6,  # 12 --> 6
        7,  # 13 --> 7
        6,  # 14 --> 6
        8,
        9,
        10,
    ]

    # Indices of sentinel-2 high-/medium-/low-resolution bands
    S2_BANDS_HR = [2, 3, 4, 8]
    S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
    S2_BANDS_LR = [1, 10, 11]

    splits: List[str] = []

    igbp = False

    def __init__(
        self,
        root: str,
        split="val",
        no_savanna=False,
        use_s2hr=False,
        use_s2mr=False,
        use_s2lr=False,
        use_s1=False,
        labels=False,
    ) -> None:
        super().__init__()

        # Make sure at least one of the band sets are requested.
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError(
                "No input specified, set at least one of "
                + "use_[s2hr, s2mr, s2lr, s1] to True!"
            )

        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1

        assert split in self.splits
        self.no_savanna = no_savanna

        # Provide number of input channels
        self.n_inputs = self.get_ninputs()

        # Provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = self.get_display_channels()

        # Provide number of classes.
        if no_savanna:
            self.n_classes = max(self.DFC2020_CLASSES) - 1
        else:
            self.n_classes = max(self.DFC2020_CLASSES)

        self.labels = labels

        self.root = Path(root)

        # Make sure parent dir exists.
        assert self.root.exists()

        self.samples: List[Dict[str, str]]

    def load_sample(
        self,
        sample: Dict[str, str],
    ) -> Dict[str, Any]:
        """Util function for reading data from single sample.

        Args:
            sample (dict[str, str]): Dictionary defining the paths to the files for each modality of the sample.

        Returns:
            dict[str, ~typing.Any]: Dictionary defining the sample in each modality.
        """

        use_s2 = self.use_s2hr or self.use_s2mr or self.use_s2lr

        # load s2 data.
        if use_s2:
            img = self.load_s2(sample["s2"])

        # load s1 data.
        if self.use_s1:
            if use_s2:
                img = torch.concatenate((img, self.load_s1(sample["s1"])), dim=0)  # type: ignore[assignment]
            else:
                img = self.load_s1(sample["s1"])

        # load labels.
        if self.labels:
            lc = self.load_lc(sample["lc"])
            return {"image": img, "mask": lc, "id": sample["id"]}
        else:
            return {"image": img, "id": sample["id"]}

    def get_ninputs(self) -> int:
        """Calculate number of input channels.

        Returns:
            int: Number of input channels.
        """
        n_inputs = 0
        if self.use_s2hr:
            n_inputs += len(self.S2_BANDS_HR)
        if self.use_s2mr:
            n_inputs += len(self.S2_BANDS_MR)
        if self.use_s2lr:
            n_inputs += len(self.S2_BANDS_LR)
        if self.use_s1:
            n_inputs += 2
        return n_inputs

    def get_display_channels(self) -> Tuple[List[int], int]:
        """Select channels for preview images.

        Returns:
            tuple[list[int] | int, int]: Tuple of the index of display channels and the brightness factor.
        """
        if self.use_s2hr and self.use_s2lr:
            display_channels = [3, 2, 1]
            brightness_factor = 3
        elif self.use_s2hr:
            display_channels = [2, 1, 0]
            brightness_factor = 3
        elif not (self.use_s2hr or self.use_s2mr or self.use_s2lr):
            display_channels = [0]
            brightness_factor = 1
        else:
            display_channels = [0]
            brightness_factor = 3

        return display_channels, brightness_factor

    def load_s2(self, path: str) -> FloatTensor:
        """Util function for reading and cleaning Sentinel2 data.

        Args:
            path (str): Path to patch of Sentinel2 data.

        Returns:
            FloatTensor: Patch of Sentinel2 data.
        """
        bands_selected = []
        if self.use_s2hr:
            bands_selected.extend(self.S2_BANDS_HR)
        if self.use_s2mr:
            bands_selected.extend(self.S2_BANDS_MR)
        if self.use_s2lr:
            bands_selected.extend(self.S2_BANDS_LR)

        bands_selected = sorted(bands_selected)

        # Read data out of the TIFF file.
        with rasterio.open(path) as data:
            s2 = data.read(bands_selected)

        s2 = s2.astype(np.float32)
        s2 = np.clip(s2, 0, 10000)
        s2 /= 10000

        # Cast to 32-bit float.
        s2 = s2.astype(np.float32)

        # Convert to Tensor.
        s2 = torch.tensor(s2)
        assert isinstance(s2, FloatTensor)

        return s2

    def load_s1(self, path: str) -> FloatTensor:
        """Util function for reading and cleaning Sentinel1 data.

        Args:
            path (str): Path to patch of Sentinel1 data.

        Returns:
            FloatTensor: Patch of Sentinel1 data.
        """
        with rasterio.open(path) as data:
            s1 = data.read()

        s1 = s1.astype(np.float32)
        s1 = np.nan_to_num(s1)
        s1 = np.clip(s1, -25, 0)
        s1 /= 25
        s1 += 1

        # Cast to 32-bit float.
        s1 = s1.astype(np.float32)

        # Convert to Tensor.
        s1 = torch.tensor(s1)
        assert isinstance(s1, FloatTensor)

        return s1

    def load_lc(self, path: str) -> Tensor:
        """Util function for reading Sentinel land cover data.

        Args:
            path (str): Path to the land cover labels for the patch.

        Returns:
            Tensor: Label mask for the patch.
        """

        # Load labels.
        with rasterio.open(path) as data:
            lc = data.read(1)

        # Convert IGBP to dfc2020 classes.
        if self.igbp:
            lc = np.take(self.DFC2020_CLASSES, lc)
        else:
            lc = lc.astype(np.int64)

        # Adjust class scheme to ignore class savanna.
        if self.no_savanna:
            lc[lc == 3] = 0
            lc[lc > 3] -= 1

        # Convert to zero-based labels and set ignore mask.
        lc -= 1
        lc[lc == -1] = 255

        # Convert to tensor.
        lc = torch.tensor(lc, dtype=torch.int8)
        assert isinstance(lc, Tensor)

        return lc

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single example from the dataset"""

        # Get and load sample from index file.
        sample = self.samples[index]
        return self.load_sample(sample)

    def __len__(self) -> int:
        """Get number of samples in the dataset"""
        return len(self.samples)


class DFC2020(BaseSenS12MS):
    """PyTorch dataset class for the DFC2020 dataset"""

    splits = ["val", "test"]
    igbp = False

    def __init__(
        self,
        root: str,
        split="val",
        no_savanna=False,
        use_s2hr=False,
        use_s2mr=False,
        use_s2lr=False,
        use_s1=False,
        labels=False,
    ) -> None:
        super(DFC2020, self).__init__(
            root, split, no_savanna, use_s2hr, use_s2mr, use_s2lr, use_s1, labels
        )

        # Build list of sample paths.
        if split == "val":
            path = self.root / "ROIs0000_validation" / "s2_validation"
        else:
            path = self.root / "ROIs0000_test" / "s2_0"

        s2_locations = glob(str(path / "*.tif"), recursive=True)
        self.samples = []
        for s2_loc in tqdm(s2_locations, desc="[Load]"):
            s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
            lc_loc = s2_loc.replace("_dfc_", "_lc_").replace("s2_", "dfc_")
            self.samples.append(
                {"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": Path(s2_loc).name}
            )

        # Sort list of samples.
        self.samples = sorted(self.samples, key=lambda i: i["id"])

        print(f"Loaded {len(self.samples)} samples from the DFC2020 {split} subset")
