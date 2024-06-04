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
    "WandbConnectionManager",
    "setup_wandb_run",
    "config_env_vars",
    "config_args",
    "distributed_run",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import functools
import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import hydra
import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

import wandb
from minerva.trainer import Trainer
from minerva.utils import DEFAULT_CONF_DIR_PATH, DEFAULT_CONFIG_NAME, utils


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


def _config_load_resolver(path: str):
    with open(Path(path)) as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_wandb_run(
    gpu: int,
    cfg: DictConfig,
) -> Tuple[Optional[Union[Run, RunDisabled]], DictConfig]:
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
    if cfg.get("wandb_log", False) or cfg.get("project", None):
        try:
            if cfg.get("log_all", False) and cfg.world_size > 1:
                run = wandb.init(  # pragma: no cover
                    entity=cfg.get("entity", None),
                    project=cfg.get("project", None),
                    group=cfg.get("group", "DDP"),
                    dir=cfg.get("wandb_dir", None),
                    name=cfg.jobid,
                    settings=wandb.Settings(start_method="thread"),
                )
            else:
                if gpu == 0:
                    run = wandb.init(
                        entity=cfg.get("entity", None),
                        project=cfg.get("project", None),
                        dir=cfg.get("wandb_dir", None),
                        name=cfg.jobid,
                    )
            cfg["wandb_log"] = True
        except wandb.UsageError:  # type: ignore[attr-defined]  # pragma: no cover
            print(
                "wandb API Key has not been inited.",
                "\nEither call wandb.login(key=[your_api_key]) or use `wandb login` in the shell.",
                "\nOr if not using wandb, safely ignore this message.",
            )
            cfg["wandb_log"] = False
        except wandb.errors.Error as err:  # type: ignore[attr-defined]  # pragma: no cover
            print(err)
            cfg["wandb_log"] = False
    else:
        print("Weights and Biases logging OFF")

    return run, cfg


def config_env_vars(cfg: DictConfig) -> DictConfig:
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
        OmegaConf.update(
            cfg, "rank", int(slurm_nodeid) * cfg.ngpus_per_node, force_add=True
        )
        OmegaConf.update(
            cfg, "world_size", int(slurm_nnodes) * cfg.ngpus_per_node, force_add=True
        )
        OmegaConf.update(cfg, "dist_url", f"tcp://{host_name}:58472", force_add=True)
        OmegaConf.update(cfg, "jobid", slurm_jobid, force_add=True)

    else:
        # Single-node distributed training.
        OmegaConf.update(cfg, "rank", 0, force_add=True)
        OmegaConf.update(cfg, "dist_url", "tcp://localhost:58472", force_add=True)
        OmegaConf.update(cfg, "world_size", cfg.ngpus_per_node, force_add=True)
        OmegaConf.update(cfg, "jobid", None, force_add=True)

    return cfg


def config_args(cfg: DictConfig) -> DictConfig:
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
    cfg.ngpus_per_node = torch.cuda.device_count()

    # Get seed from config.
    seed = cfg.get("seed", 42)

    # Set torch, numpy and inbuilt seeds for reproducibility.
    utils.set_seeds(seed)

    return config_env_vars(cfg)


def _run_preamble(
    gpu: int,
    run: Callable[[int, Optional[Union[Run, RunDisabled]], DictConfig], Any],
    cfg: DictConfig,
) -> None:  # pragma: no cover
    # Calculates the global rank of this process.
    cfg.rank += gpu

    # Setups the `wandb` run for this process.
    wandb_run, cfg = setup_wandb_run(gpu, cfg)

    if cfg.world_size > 1:
        dist.init_process_group(  # type: ignore[attr-defined]
            backend="nccl",
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank,
        )
        print(f"INITIALISED PROCESS ON {cfg.rank}")

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True  # type: ignore

    # Start this process run.
    run(gpu, wandb_run, cfg)


def distributed_run(
    run: Callable[[int, Optional[Union[Run, RunDisabled]], DictConfig], Any]
) -> Callable[..., Any]:
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

    OmegaConf.register_new_resolver("cfg_load", _config_load_resolver, replace=True)

    @functools.wraps(run)
    def inner_decorator(cfg: DictConfig):
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)

        cfg = config_args(cfg)

        if cfg.world_size <= 1:
            # Setups up the `wandb` run.
            wandb_run, cfg = setup_wandb_run(0, cfg)

            # Run the experiment.
            run(0, wandb_run, cfg)

        else:  # pragma: no cover
            try:
                print("starting process...")
                mp.spawn(_run_preamble, (run, cfg), cfg.ngpus_per_node)  # type: ignore[attr-defined]
            except KeyboardInterrupt:
                dist.destroy_process_group()  # type: ignore[attr-defined]

    return inner_decorator


@hydra.main(
    version_base="1.3",
    config_path=str(DEFAULT_CONF_DIR_PATH),
    config_name=DEFAULT_CONFIG_NAME,
)
@distributed_run
def run_trainer(
    gpu: int, wandb_run: Optional[Union[Run, RunDisabled]], cfg: DictConfig
) -> None:

    trainer = Trainer(
        gpu=gpu,
        wandb_run=wandb_run,
        **cfg,
    )

    if not cfg.get("eval", False):
        trainer.fit()

    if cfg.get("pre_train", False) and gpu == 0:
        trainer.save_backbone()
        trainer.close()

    if not cfg.get("pre_train", False):
        trainer.test()
