"""Module to handle various IO from `dataloaders` and to models.

    Copyright (C) 2022 Harry James Baker

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program in LICENSE.txt. If not,
    see <https://www.gnu.org/licenses/>.

Author: Harry James Baker

Email: hjb1d20@soton.ac.uk or hjbaker97@gmail.com

Institution: University of Southampton

Created under a project funded by the Ordnance Survey Ltd.

TODO:
    * Add a self-supervised IO.
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Tuple, Dict, Any
from torch import Tensor
from Minerva.models import MinervaModel
import numpy as np
import torch


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def sup_tg(sample: Dict[Any, Any], model, device, mode: str) -> Tuple[Any, Any, Any, Any]:
    x_batch = sample['image']
    y_batch = sample['mask']

    x_batch = x_batch.to(torch.float)
    y_batch = np.squeeze(y_batch, axis=1)
    y_batch = y_batch.type(torch.long)

    # Transfer to GPU.
    x, y = x_batch.to(device), y_batch.to(device)

    # Runs a training epoch.
    if mode == 'train':
        loss, z = model.training_step(x, y)

    # Runs a validation epoch.
    elif mode == 'val':
        loss, z = model.validation_step(x, y)

    # Runs a testing epoch.
    elif mode == 'test':
        loss, z = model.testing_step(x, y)

    return loss, z, y, sample['bbox']


def ssl_pair_tg(sample_pair: Tuple[Dict[Any, Any], Dict[Any, Any]], model: MinervaModel,
                device: torch.device, mode: str) -> Tuple[Any, Any, Any]:
    x_i_batch: Tensor = sample_pair[0]['image']
    x_j_batch: Tensor = sample_pair[1]['image']

    y_batch = torch.arange(len(x_i_batch))
    y_batch = torch.cat([y_batch, y_batch], dim=0)

    x_i_batch = x_i_batch.to(torch.float)
    x_j_batch = x_j_batch.to(torch.float)

    x_batch = torch.stack([x_i_batch, x_j_batch])

    # Transfer to GPU.
    x, y = x_batch.to(device), y_batch.to(device)

    # Runs a training epoch.
    if mode == 'train':
        loss, z = model.training_step(x, y)

    # Runs a validation epoch.
    elif mode == 'val':
        loss, z = model.validation_step(x, y)

    return loss, z, y, sample_pair[0]['bbox']
