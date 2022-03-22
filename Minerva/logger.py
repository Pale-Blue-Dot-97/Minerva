"""Module to handle the logging of results from various model types.

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
from typing import Dict, Any
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class STGLogger():
    def __init__(self, n_batches: int, batch_size: int, out_shape, n_classes: int, n_samples: int, 
                 record_int: bool = True, record_float: bool = False) -> None:
        self.record_int = record_int
        self.record_float = record_float
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_samples = n_samples

        self.logs: Dict[str, Any] = {'batch_num' : 0,
                                     'total_loss' : 0.0,
                                     'total_correct' : 0.0,
                                     'labels': None,
                                     'predictions': None,
                                     'probs': None,
                                     'ids': [],
                                     'bounds': None}

        if self.record_int:
            self.logs['labels'] = np.empty((n_batches, batch_size, *out_shape), dtype=np.uint8)
            self.logs['predictions'] = np.empty((n_batches, batch_size, *out_shape), dtype=np.uint8)

        if self.record_float:
            try:
                self.logs['probs'] = np.empty((n_batches, batch_size, n_classes, *out_shape), dtype=np.float16)
            except MemoryError:
                print('Dataset too large to record probabilities of predicted classes!')

            try:
                self.logs['bounds'] = np.empty((n_batches, batch_size), dtype=object)
            except MemoryError:
                print('Dataset too large to record bounding boxes of samples!')

    def log(self, loss, z: Tensor, y: Tensor, bbox, mode: str, step_num: int, writer: SummaryWriter) -> None:
        if self.record_int:
            # Arg max the estimated probabilities and add to predictions.
            self.logs['predictions'][self.logs['batch_num']] = torch.argmax(z, 1)

            # Add the labels and sample IDs to lists.
            self.logs['labels'][self.logs['batch_num']] = y.numpy()
            batch_ids = []
            for i in range(self.logs['batch_num'] * self.batch_size, (self.logs['batch_num'] + 1) * self.batch_size):
                batch_ids.append(str(i).zfill(len(str(self.n_samples))))
            self.logs['ids'].append(batch_ids)

        if self.record_float:
            # Add the estimated probabilities to probs.
            self.logs['probs'][self.logs['batch_num']] = z.numpy()
            self.logs['bounds'][self.logs['batch_num']] = bbox

        ls = loss.item()
        correct = (torch.argmax(z, 1) == y).sum().item()

        self.logs['total_loss'] += ls
        self.logs['total_correct'] += correct

        writer.add_scalar(tag=f'{mode}_loss', scalar_value=ls, global_step=step_num)
        writer.add_scalar(tag=f'{mode}_acc', scalar_value=correct / len(torch.flatten(y)), global_step=step_num)

    def get_logs(self) -> Dict[str, Any]:
        return self.logs
