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
r"""Tests for :mod:`minerva.tasks.epoch`."""
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
import pytest

from minerva.tasks import MinervaTask, StandardEpoch
from minerva.utils import CONFIG, universal_path, utils


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_standard_epoch(std_batch_size, default_device, exp_cnn):
    exp_name = "{}_{}".format(
        CONFIG["model_name"], utils.timestamp_now(fmt="%d-%m-%Y_%H%M")
    )
    exp_fn = universal_path(CONFIG["dir"]["results"]) / exp_name / exp_name

    params = CONFIG.copy()

    task = StandardEpoch(
        name="pytest",
        model=exp_cnn,
        batch_size=std_batch_size,
        device=default_device,
        exp_fn=exp_fn,
        **params,
    )

    assert isinstance(task, MinervaTask)

    task.step()

    assert isinstance(task.get_logs, dict)

    assert repr(task) == "StandardEpoch-pytest"
