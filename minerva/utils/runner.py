# -*- coding: utf-8 -*-
# Copyright (C) 2023 Harry Baker
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
# along with this program in LICENSE.txt. If not,
# see <https://www.gnu.org/licenses/>.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
"""Module to handle generic functionality for running :mod:`minerva` scripts.

Attributes:
    GENERIC_PARSER (ArgumentParser): A standard argparser with arguments for use in :mod:`minerva`.
        Can be used as the basis for a user defined extended argparser.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "GNU GPLv3"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "GENERIC_PARSER",
    "config_env_vars",
    "config_args",
    "distributed_run",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import argparse
import os
import signal
import subprocess
from argparse import Namespace
from typing import Any, Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

import wandb
from minerva.utils import CONFIG, MASTER_PARSER, utils

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# ---+ CLI +--------------------------------------------------------------+
GENERIC_PARSER = argparse.ArgumentParser(parents=[MASTER_PARSER])

GENERIC_PARSER.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Set seed number",
)

GENERIC_PARSER.add_argument(
    "--model-name",
    dest="model_name",
    type=str,
    help="Name of model."
    + " Sub-string before hyphen is taken as model class name."
    + " Sub-string past hyphen can be used to differeniate between versions.",
)

GENERIC_PARSER.add_argument(
    "--model-type",
    dest="model_type",
    type=str,
    help="Type of model. Should be 'segmentation', 'scene_classifier', 'siamese' or 'mlp'",
)

GENERIC_PARSER.add_argument(
    "--pre-train",
    action="store_true",
    help="Sets experiment type to pre-train. Will save model to cache at end of training.",
)

GENERIC_PARSER.add_argument(
    "--fine-tune",
    action="store_true",
    help="Sets experiment type to fine-tune. Will load pre-trained backbone from file.",
)

GENERIC_PARSER.add_argument(
    "--eval",
    action="store_true",
    help="Sets experiment type to pre-train. Will save model to cache at end of training.",
)

GENERIC_PARSER.add_argument(
    "--balance",
    action="store_true",
    help="Activates class balancing."
    + " Depending on `model_type`, this will either be via sampling or weighting of the loss function.",
)

GENERIC_PARSER.add_argument(
    "--class-elim",
    dest="elim",
    action="store_true",
    help="Eliminates classes that are specified in config but not present in the data.",
)

GENERIC_PARSER.add_argument(
    "--sample-pairs",
    dest="sample_pairs",
    action="store_true",
    help="Use paired sampling. E.g. For Siamese models.",
)

GENERIC_PARSER.add_argument(
    "--save-model",
    dest="save_model",
    type=str,
    default=False,
    help="Whether to save the model at end of testing. Must be 'true', 'false' or 'auto'."
    + " Setting 'auto' will automatically save the model to file."
    + " 'true' will ask the user whether to or not at runtime."
    + " 'false' will not save the model and will not ask the user at runtime.",
)

GENERIC_PARSER.add_argument(
    "--run-tensorboard",
    dest="run_tensorboard",
    type=str,
    default=False,
    help="Whether to run the Tensorboard logs at end of testing. Must be 'true', 'false' or 'auto'."
    + " Setting 'auto' will automatically locate and run the logs on a local browser."
    + " 'true' will ask the user whether to or not at runtime."
    + " 'false' will not save the model and will not ask the user at runtime.",
)

GENERIC_PARSER.add_argument(
    "--save-plots-no",
    dest="save",
    action="store_false",
    help="Plots created will not be saved to file.",
)

GENERIC_PARSER.add_argument(
    "--show-plots",
    dest="show",
    action="store_true",
    help="Show plots created in a window."
    + " Warning: Do not use with a terminal-less operation, e.g. SLURM.",
)

GENERIC_PARSER.add_argument(
    "--print-dist",
    dest="p_dist",
    action="store_true",
    help="Print the distribution of classes within the data to `stdout`.",
)

GENERIC_PARSER.add_argument(
    "--plot-last-epoch",
    dest="plot_last_epoch",
    action="store_true",
    help="Plot the results from the final validation epoch.",
)

GENERIC_PARSER.add_argument(
    "--wandb-log",
    dest="wandb_log",
    action="store_true",
    help="Activate Weights and Biases logging.",
)

GENERIC_PARSER.add_argument(
    "--project_name",
    dest="project",
    type=str,
    help="Name of the Weights and Biases project this experiment belongs to.",
)

GENERIC_PARSER.add_argument(
    "--wandb-entity",
    dest="entity",
    type=str,
    help="The Weights and Biases entity to send runs to.",
)

GENERIC_PARSER.add_argument(
    "--wandb-log-all",
    dest="log_all",
    action="store_true",
    help="Will log each process on Weights and Biases. Otherwise, logging will be performed from the master process.",
)

# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def _handle_sigusr1(signum, frame) -> None:
    subprocess.Popen(  # nosec B602
        f'scontrol requeue {os.getenv("SLURM_JOB_ID")}',
        shell=True,
    )
    exit()


def _handle_sigterm(signum, frame) -> None:
    pass


def setup_wandb_run(gpu: int, args: Namespace) -> Optional[Union[Run, RunDisabled]]:
    """Sets up a :mod:`wandb` logger for either every process, the master process or not if not logging.

    .. note::
        ``args`` must contain these keys:
            * ``wandb_log`` (bool): Activate :mod:`wandb` logging.
            * ``log_all`` (bool): :mod:`wandb` logging on every process if ``True``.
                Only log on the master process if ``False``.
            * ``entity`` (str): :mod:`wandb` entity where to send runs to.
            * ``project`` (str): Name of the :mod:`wandb` project this experiment belongs to.
            * ``world_size`` (int): Total number of processes across the experiment.

    Args:
        gpu (int): Local process (GPU) number.
        args (Namespace): CLI arguments from :mod:`argparse`.

    Returns:
        Optional[Union[Run, RunDisabled]]: The :mod:`wandb` run object for this process
            or ``None`` if ``log_all=False`` and ``rank!=0``.
    """
    run: Optional[Union[Run, RunDisabled]] = None
    if args.wandb_log or args.project:
        try:
            if args.log_all and args.world_size > 1:
                run = wandb.init(
                    entity=args.entity,
                    project=args.project,
                    group="DDP",
                )
            else:
                if gpu == 0:
                    run = wandb.init(
                        entity=args.entity,
                        project=args.project,
                    )
        except wandb.UsageError:
            print(
                "wandb API Key has not been inited.",
                "\nEither call wandb.login(key=[your_api_key]) or use `wandb login` in the shell.",
                "\nOr if not using wandb, safely ignore this message.",
            )
    else:
        print("Weights and Biases logging OFF")

    return run


def config_env_vars(args: Namespace) -> Namespace:
    """Finds SLURM environment variables (if they exist) and configures args accordingly.

    If SLURM variables are found in the environment variables, the arguments are configured for a SLURM job:

    * ``args.rank`` is set to the ``SLURM_NODEID * args.ngpus_per_node``.
    * ``args.world_size`` is set to ``SLURM_NNODES * args.ngpus_per_node``.
    * ``args.dist_url`` is set to ``tcp://{host_name}:58472``

    If SLURM variables are not detected, the arguments are configured for a single-node job:

    * ``args.rank=0``.
    * ``args.world_size=args.ngpus_per_node``.
    * ``args.dist_url = "tcp://localhost:58472"``.

    Args:
        args (Namespace): Arguments from the CLI ``parser`` from :mod:`argparse`.

    Returns:
        Namespace: Inputted arguments with the addition of ``rank``, ``dist_url`` and ``world_sized`` attributes.
    """
    if "SLURM_JOB_ID" in os.environ:
        # Single-node and multi-node distributed training on SLURM cluster.
        # Requeue job on SLURM preemption.
        signal.signal(signal.SIGUSR1, _handle_sigusr1)
        signal.signal(signal.SIGTERM, _handle_sigterm)

        # Get SLURM variables.
        slurm_job_nodelist: Optional[str] = os.getenv("SLURM_JOB_NODELIST")
        slurm_nodeid: Optional[str] = os.getenv("SLURM_NODEID")
        slurm_nnodes: Optional[str] = os.getenv("SLURM_NNODES")

        # Check that SLURM variables have been found.
        assert slurm_job_nodelist is not None
        assert slurm_nodeid is not None
        assert slurm_nnodes is not None

        # Find a common host name on all nodes.
        # Assume scontrol returns hosts in the same order on all nodes.
        cmd = "scontrol show hostnames " + slurm_job_nodelist
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(slurm_nodeid) * args.ngpus_per_node
        args.world_size = int(slurm_nnodes) * args.ngpus_per_node
        args.dist_url = f"tcp://{host_name}:58472"

    else:
        # Single-node distributed training.
        args.rank = 0
        args.dist_url = "tcp://localhost:58472"
        args.world_size = args.ngpus_per_node

    return args


def config_args(args: Namespace) -> Namespace:
    """Prepare the arguments generated from the :mod:`argparser` CLI for the job run.

    * Finds and sets ``args.ngpus_per_node``;
    * updates the ``CONFIG`` with new arguments from the CLI;
    * sets the seeds from the seed found in ``CONFIG`` or from CLI;
    * uses :func:`config_env_vars` to determine the correct arguments for distributed computing jobs e.g. SLURM.

    Args:
        args (Namespace): Arguments from the CLI ``parser`` from :mod:`argparse`.

    Returns:
        Namespace: Inputted arguments with the addition of ``rank``, ``dist_url`` and ``world_sized`` attributes.
    """
    args.ngpus_per_node = torch.cuda.device_count()

    # Convert CLI arguments to dict.
    args_dict = vars(args)

    # Find which CLI arguments are not in the config.
    new_args = {key: args_dict[key] for key in args_dict if key not in CONFIG}

    # Updates the config with new arguments from the CLI.
    CONFIG.update(new_args)

    # Get seed from config.
    seed = CONFIG.get("seed", 42)

    # Set torch, numpy and inbuilt seeds for reproducibility.
    utils.set_seeds(seed)

    return config_env_vars(args)


def distributed_run(run: Callable[[int, Namespace], Any], args: Namespace) -> None:
    """Runs the supplied function and arguments with distributed computing according to arguments.

    :func:`run_preamble` adds some additional commands to initialise the process group for each run
    and allocating the GPU device number to use before running the supplied function.

    Note:
        ``args`` must contain the attributes ``rank``, ``world_size`` and ``dist_url``. These can be
        configured using :func:`config_env_vars` or :func:`config_args`.

    Args:
        run (Callable[[int, Namespace], Any]): Function to run with distributed computing.
        args (Namespace): Arguments for the run and to specify the variables for distributed computing.
    """

    def run_preamble(gpu: int, _args: Namespace) -> None:
        # Calculates the global rank of this process.
        _args.rank += gpu

        # Setups the `wandb` run for this process.
        _args.wandb_run = setup_wandb_run(gpu, _args)

        if _args.world_size > 1:
            dist.init_process_group(  # type: ignore[attr-defined]
                backend="gloo",
                init_method=_args.dist_url,
                world_size=_args.world_size,
                rank=_args.rank,
            )
            print(f"INITIALISED PROCESS ON {_args.rank}")

        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
            torch.backends.cudnn.benchmark = True  # type: ignore

        # Start this this process run.
        run(gpu, _args)

    if args.world_size <= 1:
        # Setups up the `wandb` run.
        args.wandb_run = setup_wandb_run(0, args)

        # Run the experiment.
        run(0, args)

    else:
        try:
            mp.spawn(run_preamble, (args,), args.ngpus_per_node)  # type: ignore[attr-defined]
        except KeyboardInterrupt:
            dist.destroy_process_group()  # type: ignore[attr-defined]
