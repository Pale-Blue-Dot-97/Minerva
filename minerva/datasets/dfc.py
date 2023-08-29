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
r"""Generic dataloading routines for the SEN12MS dataset of corresponding Sentinel 1,
Sentinel 2 and IGBP landcover specifically targeting the 2020 Data Fusion Contest
requirements.

The SEN12MS class is meant to provide a set of helper routines for loading individual
image patches as well as triplets of patches from the dataset. These routines can easily
be wrapped or extended for use with many Deep Learning frameworks or as standalone helper
methods. For an example use case please see the "main" routine at the end of this file.

.. note::
    Adapted from ``dfc_sen12ms_dataset.py`` authored by Lloyd Hughes, provided with
    the DFC2020 dataset.

.. note::
    Some folder/file existence and validity checks are implemented but it is
    by no means complete.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = ["Lloyd Hughes", "Harry Baker"]
__contact__ = ["lloyd.hughes@tum.de", "hjb1d20@soton.ac.uk"]
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = ["DFC2020"]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from enum import Enum, EnumType
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets import BoundingBox, GeoDataset, RasterDataset
from torchgeo.datasets.utils import BoundingBox

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# Remapping IGBP classes to simplified DFC classes.
IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class DFC(GeoDataset):
    def __init__(self) -> None:
        pass

    def __getitem__(self):
        pass


class S1Bands(Enum):
    """Sentinel 1 band specs"""

    VV = 1
    VH = 2
    ALL = [VV, VH]
    NONE = None


class S2Bands(Enum):
    "Sentinel 2 band specs"

    B01 = aerosol = 1
    B02 = blue = 2
    B03 = green = 3
    B04 = red = 4
    B05 = re1 = 5
    B06 = re2 = 6
    B07 = re3 = 7
    B08 = nir1 = 8
    B08A = nir2 = 9
    B09 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    RGB = [B04, B03, B02]
    NONE = None


class LCBands(Enum):
    LC = lc = 0
    DFC = dfc = 1
    ALL = [DFC]
    NONE = None


class Seasons(Enum):
    SPRING = "ROIs1158_spring"
    SUMMER = "ROIs1868_summer"
    FALL = "ROIs1970_fall"
    WINTER = "ROIs2017_winter"
    TESTSET = "ROIs0000_test"
    VALSET = "ROIs0000_validation"
    TEST = [TESTSET]
    VALIDATION = [VALSET]
    TRAIN = [SPRING, SUMMER, FALL, WINTER]
    ALL = [SPRING, SUMMER, FALL, WINTER, VALIDATION, TEST]


class Sensor(Enum):
    s1 = "s1"
    s2 = "s2"
    lc = "lc"
    dfc = "dfc"


class DFC2020(RasterDataset):
    """DFC2020 dataset.

    .. note::
        The order in which you request the bands is the same order they will be returned in.
    """

    metadata = {
        "spring",
        "summer",
        "autumn",
        "winter",
    }

    def __init__(
        self, base_dir: Path, seasons: List[str], bands: Union[List[EnumType], EnumType]
    ) -> None:
        self.base_dir = base_dir

        if not self.base_dir.exists():
            raise Exception("The specified base_dir for SEN12MS dataset does not exist")

        self.seasons = seasons
        self.bands = bands

    def get_scene_ids(self, season) -> Set[int]:
        """Returns a list of scene ids for a specific season.

        Args:
            season ():

        Returns:
            set[int]:
        """

        season = Seasons(season).value
        path: Path = self.base_dir / season

        if not path.exists():
            raise NameError(
                f"Could not find season {season} in base directory {self.base_dir}"
            )

        scene_list = [
            int(str(Path(s).name).split("_")[1]) for s in glob(str(path / "*"))
        ]
        return set(scene_list)

    def get_patch_ids(self, season, scene_id: int, sensor=Sensor.s1) -> List[int]:
        """Returns a list of patch ids for a specific scene within a specific season.

        Args:
            season ():
            scene_id (int):
            sensor ():

        Returns:
            list[int]:
        """
        season = Seasons(season).value
        path: Path = self.base_dir / str(season) / f"{sensor.value}_{scene_id}"

        if not path.exists():
            raise NameError(f"Could not find scene {scene_id} within season {season}")

        patch_ids: list[int] = []
        for p in glob(str(path / "*.tif")):
            patch_ids.append(int(str(Path(p).stem).rsplit("_", 1)[1].split("p")[1]))

        return patch_ids

    def get_season_ids(self, season) -> Dict[int, List[int]]:
        """Return a dict of scene ids and their corresponding patch ids.

        key => scene_ids, value => list of patch_ids

        Args:
            season ():

        Returns:
            dict[int, list[int]]:
        """
        season = Seasons(season).value
        ids = {}
        scene_ids = self.get_scene_ids(season)

        for sid in scene_ids:
            ids[sid] = self.get_patch_ids(season, sid)

        return ids

    def get_patch(
        self, season, scene_id=None, patch_id=None, bands=None
    ) -> Tuple[Optional[Tensor], Optional[BoundingBox]]:
        """Returns raster data and image bounds for the defined bands of a specific patch.

        .. note::
            This method only loads a single patch from a single sensor as defined by the bands specified.

        Args:
            season ():
            scene_id ():
            patch_id ():
            bands ():

        Returns:
            tuple:
        """
        season = Seasons(season).value
        sensor = None

        if not bands:
            return None, None

        if isinstance(bands, (list, tuple)):
            b = bands[0]
        else:
            b = bands

        bandEnum: EnumType
        if isinstance(b, S1Bands):
            sensor = Sensor.s1.value
            bandEnum = S1Bands
        elif isinstance(b, S2Bands):
            sensor = Sensor.s2.value
            bandEnum = S2Bands
        elif isinstance(b, LCBands):
            if LCBands(bands) == LCBands.LC:
                sensor = Sensor.lc.value
            else:
                sensor = Sensor.dfc.value

            bands = LCBands(1)
            bandEnum = LCBands
        else:
            raise Exception("Invalid bands specified")

        if isinstance(bands, (list, tuple)):
            bands = [b.value for b in bands]
        else:
            bands = bandEnum(bands).value

        scene = f"{sensor}_{scene_id}"
        filename = f"{season}_{scene}_p{patch_id}.tif"
        patch_path = self.base_dir / season / scene / filename

        with rasterio.open(patch_path) as patch:
            data = patch.read(bands)
            bounds = patch.bounds

        # Remap IGBP to DFC bands
        if sensor == "lc":
            data = IGBP2DFC[data]

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        # Fix numpy dtypes which are not supported by pytorch tensors
        if data.dtype == np.uint16:
            data = data.astype(np.int32)
        elif data.dtype == np.uint32:
            data = data.astype(np.int64)

        tensor = torch.tensor(data)
        bounds = BoundingBox(*(bounds, 0, np.inf))

        return tensor, bounds

    def get_s1_s2_lc_dfc_quad(
        self,
        season,
        scene_id=None,
        patch_id=None,
        s1_bands=S1Bands.ALL,
        s2_bands=S2Bands.ALL,
        lc_bands=LCBands.ALL,
        dfc_bands=LCBands.NONE,
    ) -> Tuple[
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[BoundingBox],
    ]:
        """
        Returns a quadruple of patches. S1, S2, LC and DFC as well as the geo-bounds of the patch. If the number of bands is NONE
        then a None value will be returned instead of image data
        """

        s1, bounds1 = self.get_patch(season, scene_id, patch_id, s1_bands)
        s2, bounds2 = self.get_patch(season, scene_id, patch_id, s2_bands)
        lc, bounds3 = self.get_patch(season, scene_id, patch_id, lc_bands)
        dfc, bounds4 = self.get_patch(season, scene_id, patch_id, dfc_bands)

        bounds = next(filter(None, [bounds1, bounds2, bounds3, bounds4]), None)

        return s1, s2, lc, dfc, bounds

    def __getitem__(self, index) -> dict[str, Any]:
        sample = {}
        for band in self.bands:
            data, bounds = self.get_patch(self.seasons, bands=band)

            if isinstance(band, (S1Bands, S2Bands)):
                sample["image"] = data
                sample["bbox"] = bounds

            elif isinstance(band, LCBands):
                sample["mask"] = data
                sample["bbox"] = bounds

        return sample

    def get_quad_stack(
        self,
        season,
        scene_ids: Optional[Union[List[int], int]] = None,
        patch_ids=None,
        s1_bands=S1Bands.ALL,
        s2_bands=S2Bands.ALL,
        lc_bands=LCBands.ALL,
        dfc_bands=LCBands.NONE,
    ):
        """
        Returns a triplet of numpy arrays with dimensions D, B, W, H where D is the number of patches specified
        using scene_ids and patch_ids and B is the number of bands for S1, S2 or LC
        """
        season = Seasons(season)
        scene_list: List[int] = []
        patch_list = []
        bounds = []
        s1_data = []
        s2_data = []
        lc_data = []
        dfc_data = []

        # This is due to the fact that not all patch ids are available in all scenes
        # And not all scenes exist in all seasons
        if isinstance(scene_ids, list) and isinstance(patch_ids, list):
            raise Exception("Only scene_ids or patch_ids can be a list, not both.")

        if scene_ids is None:
            scene_list = self.get_scene_ids(season)
        else:
            try:
                scene_list.extend(scene_ids)
            except TypeError:
                scene_list.append(scene_ids)

        if patch_ids is not None:
            try:
                patch_list.extend(patch_ids)
            except TypeError:
                patch_list.append(patch_ids)

        for sid in scene_list:
            if patch_ids is None:
                patch_list = self.get_patch_ids(season, sid)

            for pid in patch_list:
                s1, s2, lc, dfc, bound = self.get_s1_s2_lc_dfc_quad(
                    season, sid, pid, s1_bands, s2_bands, lc_bands, dfc_bands
                )
                s1_data.append(s1)
                s2_data.append(s2)
                lc_data.append(lc)
                dfc_data.append(dfc)
                bounds.append(bound)

        return (
            np.stack(s1_data, axis=0),
            np.stack(s2_data, axis=0),
            np.stack(lc_data, axis=0),
            np.stack(dfc_data, axis=0),
            bounds,
        )
