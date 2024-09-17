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
r"""Tests for :mod:`minerva.utils.runner`."""

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
import os
import subprocess
import time
from typing import Optional

import pytest
import requests
from internet_sabotage import no_connection
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from minerva.utils import runner


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
def test_wandb_connection_manager() -> None:
    try:
        requests.head("http://www.wandb.ai/", timeout=0.1)
    except requests.ConnectionError:
        pass
    else:
        with runner.WandbConnectionManager():
            assert os.environ["WANDB_MODE"] == "online"

    with no_connection():
        with runner.WandbConnectionManager():
            assert os.environ["WANDB_MODE"] == "offline"


def test_config_env_vars(default_config: DictConfig) -> None:
    OmegaConf.set_struct(default_config, False)
    default_config.ngpus_per_node = 1

    with pytest.raises(AttributeError):
        default_config.rank
        default_config.dist_url
        default_config.world_size

    new_cfg = runner.config_env_vars(default_config)

    if "SLURM_JOB_ID" in os.environ:
        slurm_job_nodelist: Optional[str] = os.getenv("SLURM_JOB_NODELIST")
        slurm_nodeid: Optional[str] = os.getenv("SLURM_NODEID")
        slurm_nnodes: Optional[str] = os.getenv("SLURM_NNODES")
        slurm_jobid: Optional[str] = os.getenv("SLURM_JOB_ID")

        assert slurm_job_nodelist is not None
        assert slurm_nodeid is not None
        assert slurm_nnodes is not None
        assert slurm_jobid is not None

        cmd = "scontrol show hostnames " + slurm_job_nodelist
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        default_config.rank = int(slurm_nodeid) * default_config.ngpus_per_node
        default_config.world_size = int(slurm_nnodes) * default_config.ngpus_per_node
        default_config.dist_url = f"tcp://{host_name}:58472"
        default_config.jobid = slurm_jobid

        assert default_config.rank == new_cfg.rank
        assert default_config.world_size == new_cfg.world_size
        assert default_config.dist_url == new_cfg.dist_url
        assert default_config.jobid == new_cfg.jobid

    else:
        assert new_cfg.rank == 0
        assert new_cfg.world_size == 1
        assert new_cfg.dist_url == "tcp://localhost:58472"


@runner.distributed_run
def _run_func(
    gpu: int, wandb_run: Optional[Run | RunDisabled], cfg: DictConfig
) -> None:
    time.sleep(0.5)
    return


def test_distributed_run(default_config: DictConfig) -> None:
    # Disable wandb logging on Windows in CI/CD due to pwd.
    if os.name == "nt":
        OmegaConf.update(default_config, "wandb_log", False, force_add=True)

    _run_func(default_config)
