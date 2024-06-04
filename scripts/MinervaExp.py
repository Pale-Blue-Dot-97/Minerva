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
# from typing import Optional, Union

# import hydra
# from omegaconf import DictConfig
# from wandb.sdk.lib import RunDisabled
# from wandb.sdk.wandb_run import Run

# from minerva.trainer import Trainer
# from minerva.utils import DEFAULT_CONF_DIR_PATH, DEFAULT_CONFIG_NAME, runner, utils
from minerva.utils import utils
from minerva.utils.runner import WandbConnectionManager, run_trainer

# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
# @hydra.main(
#     version_base="1.3",
#     config_path=str(DEFAULT_CONF_DIR_PATH),
#     config_name=DEFAULT_CONFIG_NAME,
# )
# @runner.distributed_run
# def main(
#     gpu: int, wandb_run: Optional[Union[Run, RunDisabled]], cfg: DictConfig
# ) -> None:
#     # cfg.rank += gpu

#     # # Setups the `wandb` run for this process.
#     # wandb_run, cfg = runner.setup_wandb_run(gpu, cfg)

#     # if cfg.world_size > 1:
#     #     dist.init_process_group(  # type: ignore[attr-defined]
#     #         backend="gloo",
#     #         init_method=cfg.dist_url,
#     #         world_size=cfg.world_size,
#     #         rank=cfg.rank,
#     #     )
#     #     print(f"INITIALISED PROCESS ON {cfg.rank}")

#     # if torch.cuda.is_available():
#     #     torch.cuda.set_device(gpu)
#     #     #torch.backends.cudnn.benchmark = True  # type: ignore

#     trainer = Trainer(
#         gpu=gpu,
#         wandb_run=wandb_run,
#         **cfg,
#     )

#     if not cfg.get("eval", False):
#         trainer.fit()

#     if cfg.get("pre_train", False) and gpu == 0:
#         trainer.save_backbone()
#         trainer.close()

#     if not cfg.get("pre_train", False):
#         trainer.test()
#     print("done!")


def main():
    with WandbConnectionManager:
        run_trainer()


if __name__ == "__main__":
    # Print Minerva banner.
    utils._print_banner()

    # Run the specified main with distributed computing and the arguments provided.
    main()
