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
import abc
from abc import ABC


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MinervaLogger(ABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_batches: int, batch_size: int, n_samples: int, record_int: bool = True,
                 record_float: bool = False) -> None:
        super(MinervaLogger, self).__init__()
        self.record_int = record_int
        self.record_float = record_float
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_samples = n_samples

        self.logs = {}
        self.results = {}

    @abc.abstractmethod
    def log(self, mode: str, step_num: int, writer: SummaryWriter, loss, *args) -> None:
        pass

    @property
    def get_logs(self) -> Dict[str, Any]:
        return self.logs

    @property
    def get_results(self) -> Dict[str,Any]:
        return self.results


class STG_Logger(MinervaLogger):
    def __init__(self, n_batches: int, batch_size: int, n_samples: int, out_shape, n_classes: int,
                 record_int: bool = True, record_float: bool = False) -> None:
        super(STG_Logger, self).__init__(n_batches, batch_size, n_samples, record_int, record_float)

        self.logs: Dict[str, Any] = {
            'batch_num' : 0,
            'total_loss' : 0.0,
            'total_correct' : 0.0}

        self.results: Dict[str, Any] = {
            'y': None,
            'z': None,
            'probs': None,
            'ids': [],
            'bounds': None}

        if self.record_int:
            self.results['y'] = np.empty((self.n_batches, self.batch_size, *out_shape), dtype=np.uint8)
            self.results['z'] = np.empty((self.n_batches, self.batch_size, *out_shape), dtype=np.uint8)

        if self.record_float:
            try:
                self.results['probs'] = np.empty((self.n_batches, self.batch_size, n_classes, *out_shape),
                                              dtype=np.float16)
            except MemoryError:
                print('Dataset too large to record probabilities of predicted classes!')

            try:
                self.results['bounds'] = np.empty((self.n_batches, self.batch_size), dtype=object)
            except MemoryError:
                print('Dataset too large to record bounding boxes of samples!')

    def log(self, mode: str, step_num: int, writer: SummaryWriter, loss, z: Tensor, y: Tensor, bbox) -> None:
        if self.record_int:
            # Arg max the estimated probabilities and add to predictions.
            self.results['z'][self.logs['batch_num']] = torch.argmax(z, 1).cpu().numpy()

            # Add the labels and sample IDs to lists.
            self.results['y'][self.logs['batch_num']] = y.cpu().numpy()
            batch_ids = []
            for i in range(self.logs['batch_num'] * self.batch_size, (self.logs['batch_num'] + 1) * self.batch_size):
                batch_ids.append(str(i).zfill(len(str(self.n_samples))))
            self.results['ids'].append(batch_ids)

        if self.record_float:
            # Add the estimated probabilities to probs.
            self.results['probs'][self.logs['batch_num']] = z.detach().cpu().numpy()
            self.results['bounds'][self.logs['batch_num']] = bbox

        ls = loss.item()
        correct = (torch.argmax(z, 1) == y).sum().item()

        self.logs['total_loss'] += ls
        self.logs['total_correct'] += correct

        writer.add_scalar(tag=f'{mode}_loss', scalar_value=ls, global_step=step_num)
        writer.add_scalar(tag=f'{mode}_acc', scalar_value=correct / len(torch.flatten(y)), global_step=step_num)

        self.logs['batch_num'] += 1


class SSL_Logger(MinervaLogger):
    def __init__(self, n_batches: int, batch_size: int, n_samples: int, out_shape=None, n_classes: int = None,
                 record_int: bool = True, record_float: bool = False) -> None:
        super(SSL_Logger, self).__init__(n_batches, batch_size, n_samples, record_int, record_float)

        self.logs: Dict[str, Any] = {'batch_num' : 0,
                                     'total_loss' : 0.0}

    def log(self, mode: str, step_num: int, writer: SummaryWriter, loss, z: Tensor, y: Tensor, bbox) -> None:
        ls = loss.item()
        self.logs['total_loss'] += ls

        writer.add_scalar(tag=f'{mode}_loss', scalar_value=ls, global_step=step_num)

        self.logs['batch_num'] += 1
