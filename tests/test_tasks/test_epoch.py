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
r"""Tests for :mod:`minerva.tasks.epoch`."""
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
from torch.optim import SGD

from omegaconf import DictConfig, OmegaConf

from minerva.models import MinervaModel
from minerva.tasks import MinervaTask, StandardEpoch
from minerva.utils import universal_path, utils


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_standard_epoch(default_device, default_config: DictConfig, exp_fcn: MinervaModel):
    exp_fcn.determine_output_dim()
    optimiser = SGD(exp_fcn.parameters(), lr=1.0e-3)
    exp_fcn.set_optimiser(optimiser)
    exp_fcn.to(default_device)

    exp_name = "{}_{}".format(
        default_config["model_name"], utils.timestamp_now(fmt="%d-%m-%Y_%H%M")
    )
    exp_fn = universal_path(default_config["dir"]["results"]) / exp_name / exp_name

    params = OmegaConf.to_object(default_config)
    assert isinstance(params, dict)

    task = StandardEpoch(  # type: ignore[arg-type]
        name="fit-train",
        model=exp_fcn,
        device=default_device,
        exp_fn=exp_fn,
        **params,
    )

    assert isinstance(task, MinervaTask)

    task.step()

    assert isinstance(task.get_logs, dict)

    assert repr(task) == "StandardEpoch-fit-train"
