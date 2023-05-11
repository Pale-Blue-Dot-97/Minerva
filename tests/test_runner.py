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
r"""Tests for :mod:`minerva.utils.runner`.
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
import os
import subprocess
import time
from typing import Optional

import pytest
import requests
import torch
from internet_sabotage import no_connection

from minerva.utils import CONFIG, runner


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


def test_config_env_vars() -> None:
    args, _ = runner.GENERIC_PARSER.parse_known_args()

    args.ngpus_per_node = 1

    with pytest.raises(AttributeError):
        args.rank
        args.dist_url
        args.world_size

    new_args = runner.config_env_vars(args)

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
        args.rank = int(slurm_nodeid) * args.ngpus_per_node
        args.world_size = int(slurm_nnodes) * args.ngpus_per_node
        args.dist_url = f"tcp://{host_name}:58472"
        args.jobid = slurm_jobid

    else:
        assert new_args.rank == 0
        assert new_args.world_size == 1
        assert new_args.dist_url == "tcp://localhost:58472"


def test_config_args() -> None:
    args, _ = runner.GENERIC_PARSER.parse_known_args()

    args_dict = vars(args)

    # Find which CLI arguments are not in the config.
    new_args = {key: args_dict[key] for key in args_dict if key not in CONFIG}

    returned_args = runner.config_args(args)

    assert returned_args.ngpus_per_node == torch.cuda.device_count()
    assert CONFIG["seed"] is not None

    for key in new_args.keys():
        assert CONFIG[key] == new_args[key]


def _run(gpu: int, args) -> None:
    time.sleep(0.5)
    return


def test_distributed_run() -> None:
    args, _ = runner.GENERIC_PARSER.parse_known_args()

    args = runner.config_args(args)

    runner.distributed_run(_run, args)
