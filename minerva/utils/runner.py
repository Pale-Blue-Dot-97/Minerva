# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
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
"""Module to handle generic functionality for running :mod:`minerva` scripts.

Attributes:
    GENERIC_PARSER (~argparse.ArgumentParser): A standard argparser with arguments for use in :mod:`minerva`.
        Can be used as the basis for a user defined extended argparser.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "GENERIC_PARSER",
    "WandbConnectionManager",
    "setup_wandb_run",
    "config_env_vars",
    "config_args",
    "distributed_run",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import argparse
import os
import shlex
import signal
import subprocess
from argparse import Namespace
from typing import Any, Callable, Optional, Union

import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from minerva.utils import CONFIG, MASTER_PARSER, utils

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# ---+ CLI +--------------------------------------------------------------+
GENERIC_PARSER = argparse.ArgumentParser(parents=[MASTER_PARSER])

GENERIC_PARSER.add_argument(
    "-o",
    "--override",
    dest="override",
    action="store_true",
    help="Override config arguments with the CLI arguments where they overlap.",
)

GENERIC_PARSER.add_argument(
    "--seed",
    dest="seed",
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
    choices=("segmentation", "ssl", "siamese", "scene_classifier", "mlp"),
)

GENERIC_PARSER.add_argument(
    "--max_epochs",
    dest="max_epochs",
    type=int,
    default=100,
    help="Maximum number of training epochs.",
)

GENERIC_PARSER.add_argument(
    "--batch-size",
    dest="batch_size",
    type=int,
    default=8,
    help="Number of samples in each batch.",
)

GENERIC_PARSER.add_argument(
    "--lr",
    dest="lr",
    type=float,
    default=0.01,
    help="Learning rate of the optimiser.",
)

GENERIC_PARSER.add_argument(
    "--optim-func",
    dest="optim_func",
    type=str,
    default="SGD",
    help="Name of the optimiser to use. Only works for ``torch`` losses"
    + "(or if ``module`` is specified in the ``optim_params`` in the config)",
)

GENERIC_PARSER.add_argument(
    "--loss-func",
    dest="loss_func",
    type=str,
    default="CrossEntropyLoss",
    help="Name of the loss function to use. Only works for ``torch`` losses"
    + "(or if ``module`` is specified in the ``loss_params`` in the config)",
)

GENERIC_PARSER.add_argument(
    "--pre-train",
    dest="pre_train",
    action="store_true",
    help="Sets experiment type to pre-train. Will save model to cache at end of training.",
)

GENERIC_PARSER.add_argument(
    "--fine-tune",
    dest="fine_tune",
    action="store_true",
    help="Sets experiment type to fine-tune. Will load pre-trained backbone from file.",
)

GENERIC_PARSER.add_argument(
    "--eval",
    dest="eval",
    action="store_true",
    help="Sets experiment type to pre-train. Will save model to cache at end of training.",
)

GENERIC_PARSER.add_argument(
    "--balance",
    dest="balance",
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
    choices=("true", "false", "auto"),
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
    choices=("true", "false", "auto"),
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
    "--wandb-dir",
    dest="wandb_dir",
    type=str,
    default="./wandb",
    help="Where to store the Weights and Biases logs locally.",
)

GENERIC_PARSER.add_argument(
    "--wandb-log-all",
    dest="log_all",
    action="store_true",
    help="Will log each process on Weights and Biases. Otherwise, logging will be performed from the master process.",
)

GENERIC_PARSER.add_argument(
    "--knn-k",
    dest="knn_k",
    type=int,
    default=200,
    help="Top k most similar images used to predict the image for KNN validation.",
)

GENERIC_PARSER.add_argument(
    "--val-freq",
    dest="val_freq",
    type=int,
    default=5,
    help="Perform a validation epoch with KNN for every ``val_freq``"
    + "training epochs for SSL or Siamese models.",
)


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class WandbConnectionManager:
    """Checks for a connection to :mod:`wandb`. If not, sets :mod:`wandb` to offline during context."""

    def __init__(self) -> None:
        try:
            requests.head("http://www.wandb.ai/", timeout=0.1)
            self._on = True
        except requests.ConnectionError:
            self._on = False

    def __enter__(self) -> None:
        if self._on:
            os.environ["WANDB_MODE"] = "online"
        else:
            os.environ["WANDB_MODE"] = "offline"

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        os.environ["WANDB_MODE"] = "online"


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def _handle_sigusr1(signum, frame) -> None:  # pragma: no cover
    subprocess.Popen(  # nosec B602
        shlex.split(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}'),
        shell=True,
    )
    exit()


def _handle_sigterm(signum, frame) -> None:  # pragma: no cover
    pass


def setup_wandb_run(gpu: int, args: Namespace) -> Optional[Union[Run, RunDisabled]]:
    """Sets up a :mod:`wandb` logger for either every process, the master process or not if not logging.

    Note:
        ``args`` must contain these keys:

        * ``wandb_log`` (bool): Activate :mod:`wandb` logging.
        * | ``log_all`` (bool): :mod:`wandb` logging on every process if ``True``.
          | Only log on master process if ``False``.
        * ``entity`` (str): :mod:`wandb` entity where to send runs to.
        * ``project`` (str): Name of the :mod:`wandb` project this experiment belongs to.
        * ``world_size`` (int): Total number of processes across the experiment.

    Args:
        gpu (int): Local process (GPU) number.
        args (~argparse.Namespace): CLI arguments from :mod:`argparse`.

    Returns:
        ~wandb.sdk.wandb_run.Run | ~wandb.sdk.lib.RunDisabled | None: The :mod:`wandb` run object
        for this process or ``None`` if ``log_all=False`` and ``rank!=0``.
    """
    run: Optional[Union[Run, RunDisabled]] = None
    if CONFIG.get("wandb_log", False) or CONFIG.get("project", None):
        try:
            if CONFIG.get("log_all", False) and args.world_size > 1:
                run = wandb.init(  # pragma: no cover
                    entity=CONFIG.get("entity", None),
                    project=CONFIG.get("project", None),
                    group=CONFIG.get("group", "DDP"),
                    dir=CONFIG.get("wandb_dir", None),
                    name=args.jobid,
                )
            else:
                if gpu == 0:
                    run = wandb.init(
                        entity=CONFIG.get("entity", None),
                        project=CONFIG.get("project", None),
                        dir=CONFIG.get("wandb_dir", None),
                        name=args.jobid,
                    )
            CONFIG["wandb_log"] = True
        except wandb.UsageError:  # type: ignore[attr-defined]  # pragma: no cover
            print(
                "wandb API Key has not been inited.",
                "\nEither call wandb.login(key=[your_api_key]) or use `wandb login` in the shell.",
                "\nOr if not using wandb, safely ignore this message.",
            )
            CONFIG["wandb_log"] = False
        except wandb.errors.Error as err:  # type: ignore[attr-defined]  # pragma: no cover
            print(err)
            CONFIG["wandb_log"] = False
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
        args (~argparse.Namespace): Arguments from the CLI ``parser`` from :mod:`argparse`.

    Returns:
        ~argparse.Namespace: Inputted arguments with the addition of ``rank``, ``dist_url``
        and ``world_sized`` attributes.
    """
    if "SLURM_JOB_ID" in os.environ:  # pragma: no cover
        # Single-node and multi-node distributed training on SLURM cluster.
        # Requeue job on SLURM preemption.
        signal.signal(signal.SIGUSR1, _handle_sigusr1)  # type: ignore[attr-defined]
        signal.signal(signal.SIGTERM, _handle_sigterm)

        # Get SLURM variables.
        slurm_job_nodelist: Optional[str] = os.getenv("SLURM_JOB_NODELIST")
        slurm_nodeid: Optional[str] = os.getenv("SLURM_NODEID")
        slurm_nnodes: Optional[str] = os.getenv("SLURM_NNODES")
        slurm_jobid: Optional[str] = os.getenv("SLURM_JOB_ID")

        # Check that SLURM variables have been found.
        assert slurm_job_nodelist is not None
        assert slurm_nodeid is not None
        assert slurm_nnodes is not None
        assert slurm_jobid is not None

        # Find a common host name on all nodes.
        # Assume scontrol returns hosts in the same order on all nodes.
        cmd = "scontrol show hostnames " + slurm_job_nodelist
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(slurm_nodeid) * args.ngpus_per_node
        args.world_size = int(slurm_nnodes) * args.ngpus_per_node
        args.dist_url = f"tcp://{host_name}:58472"
        args.jobid = slurm_jobid

    else:
        # Single-node distributed training.
        args.rank = 0
        args.dist_url = "tcp://localhost:58472"
        args.world_size = args.ngpus_per_node
        args.jobid = None

    return args


def config_args(args: Namespace) -> Namespace:
    """Prepare the arguments generated from the :mod:`argparse` CLI for the job run.

    * Finds and sets ``args.ngpus_per_node``;
    * updates the ``CONFIG`` with new arguments from the CLI;
    * sets the seeds from the seed found in ``CONFIG`` or from CLI;
    * uses :func:`config_env_vars` to determine the correct arguments for distributed computing jobs e.g. SLURM.

    Args:
        args (~argparse.Namespace): Arguments from the CLI ``parser`` from :mod:`argparse`.

    Returns:
        ~argparse.Namespace: Inputted arguments with the addition of ``rank``, ``dist_url``
        and ``world_sized`` attributes.
    """
    args.ngpus_per_node = torch.cuda.device_count()

    # Convert CLI arguments to dict.
    args_dict = vars(args)

    # Find which CLI arguments are not in the config.
    new_args = {key: args_dict[key] for key in args_dict if key not in CONFIG}

    # Updates the config with new arguments from the CLI.
    CONFIG.update(new_args)

    # Overrides the arguments from the config with those of the CLI where they overlap.
    # WARNING: This will include the use of the default CLI arguments.
    if args_dict.get("override"):  # pragma: no cover
        updated_args = {
            key: args_dict[key]
            for key in args_dict
            if args_dict[key] != CONFIG[key] and args_dict[key] is not None
        }
        CONFIG.update(updated_args)

    # Get seed from config.
    seed = CONFIG.get("seed", 42)

    # Set torch, numpy and inbuilt seeds for reproducibility.
    utils.set_seeds(seed)

    return config_env_vars(args)


def _run_preamble(
    gpu: int, run: Callable[[int, Namespace], Any], args: Namespace
) -> None:  # pragma: no cover
    # Calculates the global rank of this process.
    args.rank += gpu

    # Setups the `wandb` run for this process.
    args.wandb_run = setup_wandb_run(gpu, args)

    if args.world_size > 1:
        dist.init_process_group(  # type: ignore[attr-defined]
            backend="gloo",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        print(f"INITIALISED PROCESS ON {args.rank}")

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True  # type: ignore

    # Start this process run.
    run(gpu, args)


def distributed_run(run: Callable[[int, Namespace], Any], args: Namespace) -> None:
    """Runs the supplied function and arguments with distributed computing according to arguments.

    :func:`_run_preamble` adds some additional commands to initialise the process group for each run
    and allocating the GPU device number to use before running the supplied function.

    Note:
        ``args`` must contain the attributes ``rank``, ``world_size`` and ``dist_url``. These can be
        configured using :func:`config_env_vars` or :func:`config_args`.

    Args:
        run (~typing.Callable[[int, ~argparse.Namespace], ~typing.Any]): Function to run with distributed computing.
        args (~argparse.Namespace): Arguments for the run and to specify the variables for distributed computing.
    """
    if args.world_size <= 1:
        # Setups up the `wandb` run.
        args.wandb_run = setup_wandb_run(0, args)

        # Run the experiment.
        run(0, args)

    else:  # pragma: no cover
        try:
            mp.spawn(_run_preamble, (run, args), args.ngpus_per_node)  # type: ignore[attr-defined]
        except KeyboardInterrupt:
            dist.destroy_process_group()  # type: ignore[attr-defined]
