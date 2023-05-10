# -*- coding: utf-8 -*-
# Copyright (C) 2023 Harry Baker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
r"""Tests for :mod:`minerva.pytorchtools`.
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
import tempfile
from pathlib import Path

from torchvision.models import alexnet

from minerva.pytorchtools import EarlyStopping


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_earlystopping() -> None:
    path = Path(tempfile.gettempdir(), "exp1.pt")

    path.unlink(missing_ok=True)

    stopper = EarlyStopping(patience=3, verbose=True, path=path)

    assert isinstance(stopper, EarlyStopping)

    model = alexnet()

    for loss in (2.2, 2.1, 3.4, 2.5, 2.0, 1.8, 1.9, 2.0):
        stopper(loss, model)
        assert stopper.early_stop is False

    stopper(2.1, model)
    assert stopper.early_stop

    path.unlink(missing_ok=True)
