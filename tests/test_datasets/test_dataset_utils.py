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
r"""Tests for :mod:`minerva.datasets.utils`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
from pathlib import Path

import pytest
from torchgeo.datasets import IntersectionDataset
from torchgeo.datasets.utils import BoundingBox

from minerva import datasets as mdt
from minerva.datasets import PairedDataset
from minerva.datasets.__testing import TstImgDataset, TstMaskDataset


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_make_bounding_box() -> None:
    assert mdt.make_bounding_box() is None
    assert mdt.make_bounding_box(False) is None

    bbox = (1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
    assert mdt.make_bounding_box(bbox) == BoundingBox(*bbox)

    with pytest.raises(
        ValueError,
        match="``roi`` must be a sequence of floats or ``False``, not ``True``",
    ):
        _ = mdt.make_bounding_box(True)


def test_intersect_datasets(img_root: Path, lc_root: Path) -> None:
    imagery = PairedDataset(TstImgDataset, str(img_root))
    labels = PairedDataset(TstMaskDataset, str(lc_root))

    assert isinstance(mdt.intersect_datasets([imagery, labels]), IntersectionDataset)
