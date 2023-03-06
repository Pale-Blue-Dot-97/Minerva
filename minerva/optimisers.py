# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2018 Noah Golmant
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Custom :mod:`torch` optimisers."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Noah Golmant"
__license__ = "MIT License"
__copyright__ = "Copyright (c) 2018 Noah Golmant"

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch
from torch.optim.optimizer import Optimizer


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.

    Source: https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py

    Args:
        params (iterable | dict): Iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
        max_epoch (int): maximum training epoch to determine polynomial LR decay.

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks: https://arxiv.org/abs/1708.03888

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Union[Iterable[Any], Dict[Any, Any]],
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        eta: float = 0.001,
        max_epoch: int = 200,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            max_epoch=max_epoch,
        )
        super(LARS, self).__init__(params, defaults)

    def step(  # type: ignore[override]
        self, epoch: Optional[int] = None, closure: Optional[Callable[..., Any]] = None
    ):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch (int, optioanl): Current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            lr = group["lr"]
            max_epoch = group["max_epoch"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                weight_norm = torch.linalg.norm(p.data)
                grad_norm = torch.linalg.norm(d_p)

                # Global LR computed on polynomial decay schedule
                decay = (1 - float(epoch) / max_epoch) ** 2
                global_lr = lr * decay

                # Compute local learning rate for this layer
                local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm)

                # Update the momentum term
                actual_lr = local_lr * global_lr

                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)  # type: ignore[attr-defined]
                else:
                    buf = param_state["momentum_buffer"]
                buf.mul_(momentum).add_(d_p + weight_decay * p.data, alpha=actual_lr)
                p.data.add_(-buf)

        return loss
