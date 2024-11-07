#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""Adaptation of ``MinervaExp.py`` for cluster visualisation of a model.
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
import hydra
from omegaconf import DictConfig, OmegaConf

from minerva.utils import DEFAULT_CONF_DIR_PATH, DEFAULT_CONFIG_NAME, utils
from minerva.trainer import Trainer
from minerva.utils.runner import _config_load_resolver, _construct_patch_size


OmegaConf.register_new_resolver("cfg_load", _config_load_resolver, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "to_patch_size", _construct_patch_size, replace=True
)


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
@hydra.main(
    version_base="1.3",
    config_path=str(DEFAULT_CONF_DIR_PATH),
    config_name=DEFAULT_CONFIG_NAME,
)
def main(cfg: DictConfig) -> None:
    trainer = Trainer(
        gpu=0,
        wandb_run=None,
        **cfg,  # type: ignore[misc]
    )
    trainer.tsne_cluster()
    trainer.close()


if __name__ == "__main__":
    # Print Minerva banner.
    utils._print_banner()

    main()
