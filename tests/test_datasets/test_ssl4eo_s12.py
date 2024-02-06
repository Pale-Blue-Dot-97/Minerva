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
r"""Tests for :mod:`minerva.datasets.ssl4eo_s12`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
from pathlib import Path

from rasterio.crs import CRS

from minerva.datasets import (
    GeoSSL4EOS12Sentinel2,
    NonGeoSSL4EOS12Sentinel2,
    PairedGeoDataset,
    PairedNonGeoDataset,
)


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_geossl4eos12sentinel2(data_root: Path) -> None:
    path = str(data_root / "SSL4EO-S12")
    bands = ["B2", "B3", "B4", "B8"]
    crs = CRS.from_epsg(25832)
    res = 10.0

    all_bands_dataset = GeoSSL4EOS12Sentinel2(paths=path, res=res, crs=crs)

    assert isinstance(all_bands_dataset, GeoSSL4EOS12Sentinel2)

    rgbi_dataset = GeoSSL4EOS12Sentinel2(paths=path, bands=bands, res=res, crs=crs)
    assert isinstance(rgbi_dataset, GeoSSL4EOS12Sentinel2)

    paired_dataset = PairedGeoDataset(rgbi_dataset)

    assert isinstance(paired_dataset, PairedGeoDataset)

    init_as_paired = PairedGeoDataset(
        GeoSSL4EOS12Sentinel2, paths=path, bands=bands, res=res, crs=crs
    )

    assert isinstance(init_as_paired, PairedGeoDataset)


def test_nongeossl4eos12sentinel2(data_root: Path) -> None:
    path = str(data_root / "SSL4EO-S12")
    bands = ["B2", "B3", "B4", "B8"]

    all_bands_dataset = NonGeoSSL4EOS12Sentinel2(root=path)

    assert isinstance(all_bands_dataset, NonGeoSSL4EOS12Sentinel2)

    rgbi_dataset = NonGeoSSL4EOS12Sentinel2(root=path, bands=bands)
    assert isinstance(rgbi_dataset, NonGeoSSL4EOS12Sentinel2)

    paired_dataset = PairedNonGeoDataset(rgbi_dataset, 32, 32)

    assert isinstance(paired_dataset, PairedNonGeoDataset)

    init_as_paired = PairedNonGeoDataset(
        NonGeoSSL4EOS12Sentinel2,
        root=path,
        bands=bands,
        size=32,
        max_r=32,
    )

    assert isinstance(init_as_paired, PairedNonGeoDataset)
