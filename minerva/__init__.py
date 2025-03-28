# -*- coding: utf-8 -*-
# Copyright (C) 2024 Harry Baker
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
r""":mod:`minerva` is a package designed to facilitate the fitting and visualation of models for geo-spatial research.

To main entry point to :mod:`minerva` is via :class:`Trainer`.
    >>> import hydra
    >>> from omegaconf import DictConfig
    >>> from minerva.trainer import Trainer  # Class designed to handle fitting of model.

    >>> @hydra.main(version_base="1.3")
    >>> def main(cfg: DictConfig) -> None:
    >>>     # Initialise a Trainer. Also creates the model.
    >>>     trainer = Trainer(**cfg)
    >>>     # Run the fitting (train and validation epochs).
    >>>     trainer.fit()
    >>>     # Run the testing epoch and output results.
    >>>     trainer.test()

.. note::
    Includes two small ``.tiff`` exercpts from the ChesapeakeCVPR dataset used for testing.

    https://lila.science/datasets/chesapeakelandcover Credit for the data goes to:

    Robinson C, Hou L, Malkin K, Soobitsky R, Czawlytko J, Dilkina B, Jojic N.
    Large Scale High-Resolution Land Cover Mapping with Multi-Resolution Data.
    Proceedings of the 2019 Conference on Computer Vision and Pattern Recognition (CVPR 2019)
"""

__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"

try:
    from ._version import version as __version__
    from ._version import version_tuple as version_tuple
except ImportError:  # pragma: no cover
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")  # type: ignore[assignment]

import warnings

warnings.filterwarnings("ignore", module="torchvision")
