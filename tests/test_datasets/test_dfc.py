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
r"""Tests for :mod:`minerva.datasets.dfc`.
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
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from minerva.datasets import DFC2020


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
@pytest.mark.parametrize(
    ["split", "no_savanna", "use_s2hr", "use_s2mr", "use_s2lr", "use_s1", "labels"],
    [
        ("val", False, False, False, False, False, False),  # Expect error
        ("val", False, True, True, True, False, True),  # Validation, S2 and labels
        ("test", True, True, False, False, False, False),  # Test, just high-res S2
        ("test", False, False, False, False, True, False),  # Test, just S1
        ("val", True, True, True, True, True, True),  # Validation, S1&2, labels
    ],
)
def test_dfc2020(
    data_root: Path,
    split: str,
    no_savanna: bool,
    use_s2hr: bool,
    use_s2mr: bool,
    use_s2lr: bool,
    use_s1: bool,
    labels: bool,
) -> None:
    root = str(data_root / "DFC" / "DFC2020")
    if not any((use_s2hr, use_s2mr, use_s2lr, use_s1)):
        with pytest.raises(ValueError):
            _ = DFC2020(
                root, split, no_savanna, use_s2hr, use_s2mr, use_s2lr, use_s1, labels
            )
        return
    else:
        dataset = DFC2020(
            root, split, no_savanna, use_s2hr, use_s2mr, use_s2lr, use_s1, labels
        )

    sampler = RandomSampler(dataset, replacement=True)
    dataloader = DataLoader(dataset, sampler=sampler)

    sample = next(iter(dataloader))

    assert isinstance(sample["image"], FloatTensor)
    if labels:
        assert isinstance(sample["mask"], Tensor)
