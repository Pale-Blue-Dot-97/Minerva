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
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Sequence, Tuple, Dict, Any, Literal
from torch import Tensor
from torchgeo.datasets import GeoDataset
from minerva.models import MinervaModel
from torchgeo.datasets.utils import BoundingBox
from minerva.utils import utils
import numpy as np
import torch


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def sup_tg(batch: Dict[Any, Any], model: MinervaModel, device: torch.device,
           mode: Literal['train', 'val', 'test'], **kwargs) -> Tuple[Any, Tensor, Tensor, Sequence[BoundingBox]]:
    """Provides IO functionality for a supervised model using `torchgeo` datasets.

    Args:
        batch (Dict[Any, Any]): Batch of data in a dict. Must have 'image', 'mask' and 'bbox' keys.
        model (MinervaModel): Model being fitted.
        device (torch.device): `torch` device object to send data to (e.g. CUDA device).
        mode (Literal['train', 'val', 'test']): Mode of model fitting to use.

    Returns:
        Tuple[Any, Tensor, Tensor, Sequence[BoundingBox]]: The `loss`, the model output `z`, the `y` supplied
            and the bounding boxes of the input images supplied.
    """
    # Extracts the x and y batches from the dict.
    x_batch: Tensor = batch['image']
    y_batch: Tensor = batch['mask']

    # Re-arranges the x and y batches.
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

    return loss, z, y, batch['bbox']


def ssl_pair_tg(batch: Dict[Any, Any], model: MinervaModel, device: torch.device, mode: Literal['train', 'val'],
                dataset: GeoDataset, **kwargs) -> Tuple[Any, Tensor, Tensor, Sequence[BoundingBox]]:
    """Provides IO functionality for a self-supervised Siamese model using `torchgeo` datasets.

    Args:
        batch (Dict[Any, Any]): Batch of data in a dict. Must have 'image' and 'bbox' keys.
        model (MinervaModel): Model being fitted.
        device (torch.device): `torch` device object to send data to (e.g. CUDA device).
        mode (Literal['train', 'val']): Mode of model fitting to use.
        dataset (GeoDataset): The same dataset object the `batch` was sampled from,
            to be used to sample the geo-similar batch.

    Returns:
        Tuple[Any, Tensor, Tensor, Sequence[BoundingBox]]: The `loss`, the model output `z`, the `y` supplied
            and the bounding boxes of the original input images supplied.
    """
    # Extracts the x_i batch from the dict.
    x_i_batch: Tensor = batch['image']

    # The jth_batch (i.e. the other half of the pairs) are extracted by using the bounding boxes
    # of the original batch to find geo-similar samples.
    j_batch = utils.extract_geo_pairs(batch['bbox'], dataset, max_r=kwargs['max_r'])
    x_j_batch: Tensor = j_batch['image']

    # Creates an identity matrix to act as the y labels.
    y_batch = torch.arange(len(x_i_batch))
    y_batch = torch.cat([y_batch, y_batch], dim=0)

    # Ensures both batches of images are floats.
    x_i_batch = x_i_batch.to(torch.float)
    x_j_batch = x_j_batch.to(torch.float)

    # Stacks each side of the pair batches together.
    x_batch = torch.stack([x_i_batch, x_j_batch])

    # Transfer to GPU.
    x, y = x_batch.to(device), y_batch.to(device)

    # Runs a training epoch.
    if mode == 'train':
        loss, z = model.training_step(x, y)

    # Runs a validation epoch.
    elif mode == 'val':
        loss, z = model.validation_step(x, y)

    return loss, z, y, batch['bbox']
