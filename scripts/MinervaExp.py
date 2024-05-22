#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
# Copyright (C) 2024 Harry Baker
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
"""Script to execute the creation, fitting and testing of a computer vision neural network model.

Designed for use in SLURM clusters and with distributed computing support.

Some code derived from Barlow Twins implementation of distributed computing:
https://github.com/facebookresearch/barlowtwins
"""

# TODO: Add ability to conduct hyper-parameter iterative variation experimentation.
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"


# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import functools
from typing import Any, Callable, Optional, Union

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

# from minerva.trainer import Trainer
from minerva.utils import DEFAULT_CONF_DIR_PATH, DEFAULT_CONFIG_NAME, runner, utils


def run_preamble(
    gpu: int,
    # run: Callable[[int, Optional[Union[Run, RunDisabled]], DictConfig], Any],
    cfg: DictConfig,
) -> None:  # pragma: no cover
    # Calculates the global rank of this process.
    cfg.rank += gpu

    # Setups the `wandb` run for this process.
    wandb_run, cfg = runner.setup_wandb_run(gpu, cfg)

    if cfg.world_size > 1:
        dist.init_process_group(  # type: ignore[attr-defined]
            backend="gloo",
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank,
        )
        print(f"INITIALISED PROCESS ON {cfg.rank}")

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        # torch.backends.cudnn.benchmark = True  # type: ignore

    # Start this process run.
    main(gpu, wandb_run, cfg)


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

    OmegaConf.register_new_resolver(
        "cfg_load", runner._config_load_resolver, replace=True
    )

    @functools.wraps(run)
    def inner_decorator(cfg: DictConfig):
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        cfg = runner.config_args(cfg)

        print("Config setup complete")
        print(f"{cfg.dist_url=}")
        print(f"{cfg.world_size=}")

        if cfg.world_size <= 1:
            # Setups up the `wandb` run.
            wandb_run, cfg = runner.setup_wandb_run(0, cfg)

            # Run the experiment.
            run(0, wandb_run, cfg)

        else:  # pragma: no cover
            try:
                print("starting process...")
                mp.spawn(main, (cfg,), cfg.ngpus_per_node)  # type: ignore[attr-defined]
            except KeyboardInterrupt:
                dist.destroy_process_group()  # type: ignore[attr-defined]

    return inner_decorator


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
@hydra.main(
    version_base="1.3",
    config_path=str(DEFAULT_CONF_DIR_PATH),
    config_name=DEFAULT_CONFIG_NAME,
)
@runner.distributed_run
def main(gpu: int, wandb_run, cfg: DictConfig) -> None:
    # cfg.rank += gpu

    # # Setups the `wandb` run for this process.
    # wandb_run, cfg = runner.setup_wandb_run(gpu, cfg)

    # if cfg.world_size > 1:
    #     dist.init_process_group(  # type: ignore[attr-defined]
    #         backend="gloo",
    #         init_method=cfg.dist_url,
    #         world_size=cfg.world_size,
    #         rank=cfg.rank,
    #     )
    #     print(f"INITIALISED PROCESS ON {cfg.rank}")

    # if torch.cuda.is_available():
    #     torch.cuda.set_device(gpu)
    #     #torch.backends.cudnn.benchmark = True  # type: ignore

    # trainer = Trainer(
    #     gpu=gpu,
    #     wandb_run=wandb_run,
    #     **cfg,
    # )

    # if not cfg.get("eval", False):
    #     trainer.fit()

    # if cfg.get("pre_train", False) and gpu == 0:
    #     trainer.save_backbone()
    #     trainer.close()

    # if not cfg.get("pre_train", False):
    #     trainer.test()
    print("done!")


if __name__ == "__main__":
    # Print Minerva banner.
    utils._print_banner()

    # Run the specified main with distributed computing and the arguments provided.
    main()
