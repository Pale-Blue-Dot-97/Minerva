# -*- coding: utf-8 -*-
# Copyright (C) 2024  Harry Baker
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "DFC2020",
    "SEN12MS",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from glob import glob
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
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

    splits: list[str] = []

    igbp = False

    def __init__(
        self,
        root: str,
        split="val",
        use_s2hr=False,
        use_s2mr=False,
        use_s2lr=False,
        use_s1=False,
        labels=False,
        transforms: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__()

        self.transforms = transforms

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

        # Provide number of input channels
        self.n_inputs = self.get_ninputs()

        # Provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = self.get_display_channels()

        # Provide number of classes.
        self.n_classes = max(self.DFC2020_CLASSES)

        self.labels = labels

        self.root = Path(root)

        # Make sure parent dir exists.
        assert self.root.exists()

        self.samples: list[dict[str, str]]

    def load_sample(
        self,
        sample: dict[str, str],
        index: int,
    ) -> dict[str, Any]:
        """Util function for reading data from single sample.

        Args:
            sample (dict[str, str]): Dictionary defining the paths to the files for each modality of the sample.

        Returns:
            dict[str, ~typing.Any]: Dictionary defining the sample in each modality.
        """

        use_s2 = self.use_s2hr or self.use_s2mr or self.use_s2lr

        # Load just S2 data.
        if use_s2 and not self.use_s1:
            img = self.load_s2(sample["s2"])

        # Load S1 and S2 data.
        elif self.use_s1 and use_s2:
            img = torch.concatenate(
                (self.load_s2(sample["s2"]), self.load_s1(sample["s1"])), dim=0
            )  # type: ignore[assignment]

        # Load just S1 data.
        elif self.use_s1 and not use_s2:
            img = self.load_s1(sample["s1"])

        else:
            raise ValueError("No data selected")

        # Load labels.
        if self.labels:
            lc = self.load_lc(sample["lc"])
            return {"image": img, "mask": lc, "id": index}
        else:
            return {"image": img, "id": index}

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

    def get_display_channels(self) -> tuple[list[int], int]:
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

        # Cast to 32-bit float.
        s2 = s2.astype(np.float32)

        # Convert to Tensor.
        s2 = torch.tensor(s2)
        assert isinstance(s2, FloatTensor)

        return s2

    @staticmethod
    def load_s1(path: str) -> FloatTensor:
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

        # Convert to zero-based labels and set ignore mask.
        lc -= 1
        lc[lc == -1] = 255

        # Convert to tensor.
        lc = torch.tensor(lc, dtype=torch.int8)
        assert isinstance(lc, Tensor)

        return lc

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single example from the dataset"""

        # Get and load sample from index file.
        sample = self.load_sample(self.samples[index], index)
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def __len__(self) -> int:
        """Get number of samples in the dataset"""
        return len(self.samples)


class DFC2020(BaseSenS12MS):
    """PyTorch dataset class for the DFC2020 dataset"""

    classes = {
        0: "Forest",
        1: "Shrubland",
        2: "Savana",
        3: "Grassland",
        4: "Wetlands",
        5: "Croplands",
        6: "Urban",
        7: "Snow/Ice",
        8: "Barren",
        9: "Water",
    }

    colours = {
        0: "#009900",
        1: "#c6b044",
        2: "#fbff13",
        3: "#bbff05",
        4: "#27ff87",
        5: "#c24f44",
        6: "#a5a5a5",
        7: "#69fff8",
        8: "#f9ffa4",
        9: "#1c0dff",
    }

    splits = ["val", "test"]
    igbp = False

    def __init__(
        self,
        root: str,
        split="val",
        use_s2hr=False,
        use_s2mr=False,
        use_s2lr=False,
        use_s1=False,
        labels=False,
        transforms: Optional[Callable[..., Any]] = None,
    ) -> None:
        super(DFC2020, self).__init__(
            root,
            split,
            use_s2hr,
            use_s2mr,
            use_s2lr,
            use_s1,
            labels,
            transforms,
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

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        classes: Optional[dict[int, str]] = None,
        colours: Optional[dict[int, str]] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Adapted from :meth:`torchgeo.datasets.DFC2022.plot` for DFC2020.

        Args:
            sample (dict[str, ~torch.Tensor]): A sample returned by :meth:`__getitem__`.
            show_titles (bool): Flag indicating whether to show titles above each panel.
            suptitle (str): Optional; String to use as a suptitle.
            classes (dict[int, str]): Optional; Dictionary mapping class labels to class names.
                Default: :attr:`self.classes`.
            colours (dict[int, str]): Optional; Dictionary mapping class labels to colours.
                Default: :attr:`self.colours`.

        .. versionadded:: 0.28

        Returns:
            ~matplotlib.figure.Figure: A :mod:`matplotlib` ``Figure`` with the rendered sample.
        """
        # Reorder image from BGR to RGB.
        from kornia.color import BgrToRgb

        from minerva.transforms import AdjustGamma

        bgr_to_rgb = BgrToRgb()
        adjust_gamma = AdjustGamma(gamma=0.8)

        # Reorder channels from BGR to RGB and adjust the gamma ready for plotting.
        image = adjust_gamma(bgr_to_rgb(sample["image"][:3])).permute(1, 2, 0).numpy()

        block_size_factor = 8

        # Use inbuilt class colours and classes mappings.
        if classes is None or colours is None:
            classes = self.classes
            colours = self.colours

        # Number of columns of sub-plots in figure. Will increase for masks and predictions.
        ncols = 1

        # Will plot mask and prediction if found in the sample provided.
        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample

        # Create the cmap from the colours mapping.
        cmap = ListedColormap(colours.values(), N=len(colours))  # type: ignore[arg-type]
        vmin = -0.5
        vmax = len(classes) - 0.5

        if showing_mask:
            mask = sample["mask"].numpy()
            ncols += 1
        if showing_prediction:
            pred = sample["prediction"].numpy()
            ncols += 1

        # Make the figure and axes objects.
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 5))

        # Plot the image.
        axs[0].imshow(image)

        # Sets tick intervals to block size.
        axs[0].set_xticks(
            np.arange(0, image.shape[0] + 1, image.shape[0] // block_size_factor)
        )
        axs[0].set_yticks(
            np.arange(0, image.shape[1] + 1, image.shape[1] // block_size_factor)
        )

        # Add grid overlay.
        axs[0].grid(which="both", color="#CCCCCC", linestyle=":")

        # Plot the ground truth mask and predicted mask.
        mask_plot: Axes
        if showing_mask:
            mask_plot = axs[1].imshow(
                mask, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
            )

            # Sets tick intervals to block size.
            axs[1].set_xticks(
                np.arange(0, mask.shape[0] + 1, mask.shape[0] // block_size_factor)
            )
            axs[1].set_yticks(
                np.arange(0, mask.shape[1] + 1, mask.shape[1] // block_size_factor)
            )

            # Add grid overlay.
            axs[1].grid(which="both", color="#CCCCCC", linestyle=":")

            if showing_prediction:
                axs[2].imshow(
                    pred, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )

                # Sets tick intervals to block size.
                axs[2].set_xticks(
                    np.arange(0, pred.shape[0] + 1, pred.shape[0] // block_size_factor)
                )
                axs[2].set_yticks(
                    np.arange(0, pred.shape[1] + 1, pred.shape[1] // block_size_factor)
                )

                # Add grid overlay.
                axs[2].grid(which="both", color="#CCCCCC", linestyle=":")

        elif showing_prediction:
            mask_plot = axs[1].imshow(
                pred, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
            )

            # Sets tick intervals to block size.
            axs[1].set_xticks(
                np.arange(0, pred.shape[0] + 1, pred.shape[0] // block_size_factor)
            )
            axs[1].set_yticks(
                np.arange(0, pred.shape[1] + 1, pred.shape[1] // block_size_factor)
            )

            # Add grid overlay.
            axs[1].grid(which="both", color="#CCCCCC", linestyle=":")

        if showing_mask or showing_prediction:
            # Plots colour bar onto figure.
            clb = fig.colorbar(
                mask_plot,  # type: ignore[arg-type]
                ax=axs,
                location="top",
                ticks=np.arange(0, len(colours)),
                aspect=75,
                drawedges=True,
            )

            # Sets colour bar ticks to class labels.
            clb.ax.set_xticklabels(classes.values(), fontsize=9)

        # Add the sub-titles to each of the sub-plots.
        if show_titles:
            axs[0].set_title("Image")

            if showing_mask:
                axs[1].set_title("Ground Truth")
                if showing_prediction:
                    axs[2].set_title("Predictions")
            elif showing_prediction:
                axs[1].set_title("Predictions")

        # Add super-title.
        if suptitle is not None:
            plt.suptitle(suptitle)

        # Return the completed figure object. Will still need to be shown or saved to view.
        return fig


# TODO: Add tests to cover SEN12MS dataset
class SEN12MS(BaseSenS12MS):  # pragma: no cover
    """PyTorch dataset class for the SEN12MS dataset

    Expects dataset dir as:
    >>> - SEN12MS_holdOutScenes.txt
    >>> - ROIsxxxx_y
    >>>     - lc_n
    >>>     - s1_n
    >>>     - s2_n

    SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    train/val split and can be obtained from:
    https://github.com/MSchmitt1984/SEN12MS/blob/master/splits
    """

    splits = ["train", "holdout"]
    igbp = True

    def __init__(
        self,
        root: str,
        split="train",
        use_s2hr=False,
        use_s2mr=False,
        use_s2lr=False,
        use_s1=False,
        labels=False,
        transforms: Optional[Callable[..., Any]] = None,
    ) -> None:
        super(SEN12MS, self).__init__(
            root,
            split,
            use_s2hr,
            use_s2mr,
            use_s2lr,
            use_s1,
            labels,
            transforms,
        )

        # Find and index samples.
        self.samples = []
        if split == "train":
            pbar = tqdm(total=162556)  # we expect 541,986 / 3 * 0.9 samples
        else:
            pbar = tqdm(total=18106)  # we expect 541,986 / 3 * 0.1 samples
        pbar.set_description("[Load]")

        val_list = list(
            pd.read_csv(self.root / "SEN12MS_holdOutScenes.txt", header=None)[0]
        )
        val_list = [x.replace("s1_", "s2_") for x in val_list]

        # Compile a list of paths to all samples
        if split == "train":
            train_list = []
            for seasonfolder in [
                "ROIs1970_fall",
                "ROIs1158_spring",
                "ROIs2017_winter",
                "ROIs1868_summer",
            ]:
                train_list += [
                    str(Path(seasonfolder) / x)
                    for x in (self.root / seasonfolder).iterdir()
                ]
            train_list = [x for x in train_list if "s2_" in x]
            train_list = [x for x in train_list if x not in val_list]
            sample_dirs = train_list
        else:
            sample_dirs = val_list

        for folder in sample_dirs:
            s2_locations = glob(str(self.root / f"{folder}/*.tif"), recursive=True)

            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == "ROIs1868_summer/s2_146":
                broken_file = str(
                    self.root
                    / "ROIs1868_summer"
                    / "s2_146"
                    / "ROIs1868_summer_s2_146_p202.tif"
                )
                s2_locations.remove(broken_file)
                pbar.write("ignored one sample because of nan values in the s1 data")

            for s2_loc in s2_locations:
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
                lc_loc = s2_loc.replace("_s2_", "_lc_").replace("s2_", "lc_")

                pbar.update()
                self.samples.append(
                    {"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": Path(s2_loc).name}
                )

        pbar.close()

        # Sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i["id"])

        print(f"Loaded {len(self.samples)} samples from the SEN12MS {split} subset")
