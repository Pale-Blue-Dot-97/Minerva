# -*- coding: utf-8 -*-
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
#
"""Module containing core utility functions and abstract classes for :mod:`models`."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"

__all__ = [
    "FilterOutputs",
    "MinervaModel",
    "MinervaWrapper",
    "MinervaDataParallel",
    "MinervaBackbone",
    "MinervaOnnxModel",
    "get_model",
    "get_torch_weights",
    "get_output_shape",
    "bilinear_init",
    "is_minerva_model",
    "is_minerva_subtype",
    "extract_wrapped_model",
    "wrap_model",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
import os
import warnings
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Type, overload

import numpy as np
import torch
from nptyping import NDArray
from packaging.version import Version
from torch import Tensor
from torch._dynamo.eval_frame import OptimizedModule
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules import Module
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchvision.models._api import WeightsEnum

from minerva.utils.utils import func_by_str


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class FilterOutputs(Module):
    """Helper class for use in :class:~`torch.nn.Sequential` to filter previous layer's outputs by index.

    Attributes:
        indexes (int | list[int]): Index(es) of the inputs to pass forward.

    Args:
        indexes (int | list[int]): Index(es) of the inputs to pass forward.
    """

    def __init__(self, indexes: int | list[int]) -> None:
        self.indexes = indexes

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs[self.indexes]


class MinervaModel(Module, ABC):
    """Abstract class to act as a base for all Minerva Models.

    Designed to provide inter-compatability with :class:`~trainer.Trainer`.

    Attributes:
        criterion (~torch.nn.Module): :mod:`torch` loss function model will use.
        input_shape (tuple[int, ...]): Optional; Defines the shape of the input data. Typically in order of
            number of channels, image width, image height but may vary dependant on model specs.
        n_classes (int): Number of classes in input data.
        output_shape (tuple[int, ...]): The shape of the output of the network.
            Determined and set by :meth:`determine_output_dim`.
        optimiser: :mod:`torch` optimiser model will use, to be initialised with inherited model's parameters.

    Args:
        criterion (~torch.nn.Module): Optional; :mod:`torch` loss function model will use.
        input_shape (tuple[int, ...]): Optional; Defines the shape of the input data. Typically in order of
            number of channels, image width, image height but may vary dependant on model specs.
        n_classes (int): Optional; Number of classes in input data.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        criterion: Optional[Module] = None,
        input_size: Optional[tuple[int, ...]] = None,
        n_classes: Optional[int] = None,
        scaler: Optional[GradScaler] = None,
    ) -> None:
        super(MinervaModel, self).__init__()

        # Sets loss function
        self.criterion: Optional[Module] = criterion

        self.input_size = input_size
        self.n_classes = n_classes
        self.scaler = scaler

        # Output shape initialised as None. Should be set by calling determine_output_dim.
        self.output_shape: Optional[tuple[int, ...]] = None

        # Optimiser initialised as None as the model parameters created by its init is required to init a
        # torch optimiser. The optimiser MUST be set by calling set_optimiser before the model can be trained.
        self.optimiser: Optional[Optimizer] = None

        # Like the optimiser, the scheduler needs to be set after the model is inited as it needs to wrap
        # the optimiser. Use ``set_scheduler`` to add the scheduler to the model.
        self.scheduler: Optional[LRScheduler] = None

    def set_optimiser(self, optimiser: Optimizer) -> None:
        """Sets the optimiser used by the model.

        .. warning::
            *MUST* be called after initialising a model and supplied with a :class:`torch.optim.Optimizer`
            using this model's parameters.

        Args:
            optimiser (~torch.optim.Optimizer): :class:`torch.optim.Optimizer` model will use,
                initialised with this model's parameters.
        """
        self.optimiser = optimiser

    def set_scheduler(self, scheduler: LRScheduler) -> None:
        self.scheduler = scheduler

    def set_criterion(self, criterion: Module) -> None:
        """Set the internal criterion.

        Args:
            criterion (~torch.nn.Module): Criterion (loss function) to set.
        """
        self.criterion = criterion

    def determine_output_dim(
        self, sample_pairs: bool = False, change_detection: bool = False
    ) -> None:
        """Uses :func:`get_output_shape` to find the dimensions of the output of this model and sets to attribute."""

        assert self.input_size is not None

        self.output_shape = get_output_shape(
            self,
            self.input_size,
            sample_pairs=sample_pairs,
            change_detection=change_detection,
        )

    def _remake_classifier(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def update_n_classes(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self._remake_classifier()

    @overload
    def step(
        self, x: Tensor, y: Tensor, train: bool = False
    ) -> tuple[Tensor, Tensor | tuple[Tensor, ...]]: ...  # pragma: no cover

    @overload
    def step(
        self, x: Tensor, *, train: bool = False
    ) -> tuple[Tensor, Tensor | tuple[Tensor, ...]]: ...  # pragma: no cover

    def step(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        train: bool = False,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, ...]]:
        """Generic step of model fitting using a batch of data.

        Raises:
            NotImplementedError: If :attr:`~MinervaModel.optimiser` is ``None``.
            NotImplementedError: If :attr:`~MinervaModel.criterion` is ``None``.

        Args:
            x (~torch.Tensor): Batch of input data to network.
            y (~torch.Tensor): Either a batch of ground truth labels or generated labels/ pairs.
            train (bool): Sets whether this shall be a training step or not. ``True`` for training step
                which will then clear the :attr:`~MinervaModel.optimiser`, and perform a backward pass of the
                network then update the :attr:`~MinervaModel.optimiser`. If ``False`` for a validation or testing step,
                these actions are not taken.

        Returns:
            tuple[~torch.Tensor, ~torch.Tensor | tuple[~torch.Tensor, ...]]: :class:`tuple` of the loss computed
            by the loss function and the model outputs.
        """

        if self.optimiser is None:
            raise NotImplementedError("Optimiser has not been set!")

        if self.criterion is None:
            raise NotImplementedError("Criterion has not been set!")

        # Resets the optimiser's gradients if this is a training step.
        if train:
            self.optimiser.zero_grad()

        z: Tensor | tuple[Tensor, ...]
        loss: Tensor

        mix_precision: bool = True if self.scaler else False
        device_type = "cpu" if x.device.type == "cpu" else "cuda"

        # CUDA does not support ``torch.bfloat16`` while CPU does not support ``torch.float16`` for autocasting.
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        # Will enable mixed precision (if a Scaler has been set).
        with torch.amp.autocast_mode.autocast(
            device_type=device_type, dtype=autocast_dtype, enabled=mix_precision
        ):
            # Forward pass.
            z = self.forward(x)

            # Compute Loss.
            loss = self.criterion(z, y)

        # Performs a backward pass if this is a training step.
        if train:
            # Scales the gradients if using mixed precision training.
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()

            else:
                loss.backward()
                self.optimiser.step()

        return loss, z


class MinervaWrapper(MinervaModel):
    """Wraps a :mod:`torch` model class in :class:`MinervaModel` so it can be used in :mod:`minerva`.

    Attributes:
        model (~torch.nn.Module): The wrapped :mod:`torch` model that is now compatible with :mod:`minerva`.

    Args:
        model (~torch.nn.Module | ~typing.Callable[..., ~torch.nn.Module]): The :mod:`torch` model object or
            constructor to, initialise and then wrap in :attr:`~MinervaWrapper.model`.
        criterion (~torch.nn.Module): Optional; :mod:`torch` loss function model will use.
        input_shape (tuple[int, ...]): Optional; Defines the shape of the input data. Typically in order of
            number of channels, image width, image height but may vary dependant on model specs.
        n_classes (int): Optional; Number of classes in input data.

    """

    def __init__(
        self,
        model: Module | Callable[..., Module],
        criterion: Optional[Module] = None,
        input_size: Optional[tuple[int, ...]] = None,
        n_classes: Optional[int] = None,
        scaler: Optional[GradScaler] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(criterion, input_size, n_classes, scaler)

        if isinstance(model, Module):
            self.model = model
        else:
            self.model = model(*args, **kwargs)

    def __call__(self, *inputs) -> Any:
        return self.forward(*inputs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def __repr__(self) -> Any:
        return self.model.__repr__()

    def forward(self, *inputs) -> Any:
        return self.model.forward(*inputs)


class MinervaBackbone(MinervaModel):
    """Abstract class to mark a model for use as a backbone.

    Attributes:
        backbone (~torch.nn.Module): Backbone of the the model.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone: Module

    def get_backbone(self) -> Module:
        """Gets the :attr:`~MinervaBackbone.backbone` network of the model.

        Returns:
            ~torch.nn.Module: The :attr:`~MinervaModel.backbone` of the model.
        """
        return self.backbone

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze the backbone so that the weights do not change while the rest of the model trains.

        Args:
            freeze (bool): Whether to 'freeze' the backbone (Set :meth:`~torch.Tensor.requires_grad_` to `False`).
                Defaults to `True`.

        .. versionadded:: 0.28
        """
        self.backbone.requires_grad_(False if freeze else True)


class MinervaDataParallel(Module):  # pragma: no cover
    """Wrapper for :class:`~torch.nn.parallel.data_parallel.DataParallel` or
    :class:`~torch.nn.parallel.DistributedDataParallel` that automatically fetches the
    attributes of the wrapped model.

    Attributes:
        model (~torch.nn.Module): :mod:`torch` model to be wrapped by
            :class:`~torch.nn.parallel.data_parallel.DataParallel` or
            :class:`~torch.nn.parallel.DistributedDataParallel`.
        paralleliser (~torch.nn.parallel.data_parallel.DataParallel | ~torch.nn.parallel.DistributedDataParallel):
            The paralleliser to wrap the :attr:`~MinervaDataParallel.model` in.

    Args:
        model (~torch.nn.Module): :mod:`torch` model to be wrapped by
            :class:`~torch.nn.parallel.data_parallel.DataParallel` or
            :class:`~torch.nn.parallel.DistributedDataParallel`.
    """

    def __init__(
        self,
        model: Module,
        paralleliser: Type[DataParallel] | Type[DDP],  # type: ignore[type-arg]
        *args,
        **kwargs,
    ) -> None:
        super(MinervaDataParallel, self).__init__()
        self.model = paralleliser(model, *args, **kwargs).cuda()
        # Set these so that epoch logging will use the wrapped model's values
        self.output_shape = model.output_shape
        self.n_classes = model.n_classes

    def forward(self, *inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Ensures a forward call to the model goes to the actual wrapped model.

        Args:
            inputs (tuple[~torch.Tensor, ...]): Input of tensors to be parsed to the
                :attr:`~MinervaDataParallel.model` forward.

        Returns:
            tuple[~torch.Tensor, ...]: Output of :attr:`~MinervaDataParallel.model`.
        """
        z = self.model(*inputs)
        assert isinstance(z, tuple) and list(map(type, z)) == [Tensor] * len(z)
        return z

    def __call__(self, *inputs) -> tuple[Tensor, ...]:
        return self.forward(*inputs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

    def __repr__(self) -> Any:
        return self.model.__repr__()


class MinervaOnnxModel(MinervaModel):
    """Special model class for enabling :mod:`onnx` models to be used within :mod:`minerva`.

    Attributes:
        model (~torch.nn.Module): :mod:`onnx` model imported into :mod:`torch`.

    Args:
        model (~torch.nn.Module): :mod:`onnx` model imported into :mod:`torch`.
    """

    def __init__(self, model: Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = model

    def __call__(self, *inputs) -> Any:
        return self.model.forward(*inputs)

    def __getattr__(self, name) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def __repr__(self) -> Any:
        return self.model.__repr__()

    def forward(self, *inputs: Any) -> Any:
        """Performs a forward pass of the :attr:`~MinervaOnnxModel.model` within.

        Args:
            inputs (~typing.Any): Input to be parsed to the ``.forward`` method of :attr:`~MinervaOnnxModel.model`.

        Returns:
            ~typing.Any: Output of :attr:`~MinervaOnnxModel.model`.
        """
        return self.model.forward(*inputs)


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_model(model_name: str) -> Callable[..., MinervaModel]:
    """Returns the constructor of the ``model_name`` in :mod:`models`.

    Args:
        model_name (str): Name of the model to get.

    Returns:
        ~typing.Callable[..., MinervaModel]: Constructor of the model requested.
    """
    model: Callable[..., MinervaModel] = func_by_str("minerva.models", model_name)
    return model


def get_torch_weights(weights_name: str) -> Optional[WeightsEnum]:
    """Loads pre-trained model weights from :mod:`torchvision` via Torch Hub API.

    Args:
        weights_name (str): Name of model weights. See
            https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
            for a list of possible pre-trained weights.

    Returns:
        torchvision.models._api.WeightsEnum | None: API query for the specified weights.
        ``None`` if query cannot be found. See note on use:

    Note:
        This function only returns a query for the API of the weights. To actually use them, you need to call
        :meth:`~torchvision.models._api.WeightsEnum.get_state_dict` to download the weights (if not already in cache).
    """
    weights: Optional[WeightsEnum] = None
    try:
        weights = torch.hub.load("pytorch/vision", "get_weight", name=weights_name)
    except OSError:  # pragma: no cover
        th_dir = os.environ.get("TORCH_HUB", Path("~/.cache/torch/hub").expanduser())
        try:
            weights = torch.hub.load(
                f"{th_dir}/pytorch_vision_main",
                "get_weight",
                name=weights_name,
                source="local",
            )
        except FileNotFoundError as err:  # pragma: no cover
            print(err)
            weights = None

    return weights


def get_output_shape(
    model: Module,
    image_dim: Sequence[int] | int,
    sample_pairs: bool = False,
    change_detection: bool = False,
) -> tuple[int, ...]:
    """Gets the output shape of a model.

    Args:
        model (~torch.nn.Module): Model for which the shape of the output needs to be found.
        image_dim (~typing.Sequence[int] | int]): Expected shape of the input data to the model.
        sample_pairs (bool): Optional; Flag for if paired sampling is active.
            Will send a paired sample through the model.

    Returns:
        tuple[int, ...]: The shape of the output data from the model.
    """
    _image_dim: Sequence[int] | int = image_dim
    try:
        assert not isinstance(image_dim, int)
        if len(image_dim) == 1:
            _image_dim = image_dim[0]
    except (AssertionError, TypeError):
        if not hasattr(image_dim, "__len__"):
            pass

    if not hasattr(_image_dim, "__len__"):
        assert isinstance(_image_dim, int)
        random_input = torch.rand([4, _image_dim])
    elif sample_pairs or change_detection:
        assert isinstance(_image_dim, Iterable)
        random_input = torch.rand([2, 4, *_image_dim])
    else:
        assert isinstance(_image_dim, Iterable)
        random_input = torch.rand([4, *_image_dim])

    random_input = random_input.nan_to_num()
    with torch.no_grad():
        output: Tensor = model(random_input.to(next(model.parameters()).device))

    if len(output[0].data.shape) == 1:
        return (output[0].data.shape[0],)

    else:
        return tuple(output[0].data.shape[1:])


def bilinear_init(in_channels: int, out_channels: int, kernel_size: int) -> Tensor:
    """Constructs the weights for the bi-linear interpolation kernel for use in transpose convolutional layers.

    Source: https://github.com/haoran1062/FCN-pytorch/blob/master/FCN.py

    Args:
        in_channels (int): Number of input channels to the layer.
        out_channels (int): Number of output channels from the layer.
        kernel_size (int): Size of the (square) kernel.

    Returns:
        ~torch.Tensor: :class:`~torch.Tensor` of the initialised bi-linear interpolated weights for the
        transpose convolutional layer's kernels.
    """
    factor = (kernel_size + 1) // 2

    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = int(factor - 0.5)

    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight: NDArray[Any, Any] = np.zeros(
        (in_channels, out_channels, kernel_size, kernel_size), dtype="float32"
    )
    weight[range(in_channels), range(out_channels), :, :] = filt

    weights = torch.from_numpy(weight)  # type: ignore[attr-defined]
    assert isinstance(weights, Tensor)
    return weights


def is_minerva_model(model: Module) -> bool:
    """
    Checks if model is a :class:`MinervaModel` while accounting for models that are
    :class:~`torch._dynamo.eval_frame.OptimizedModule` from :meth:`torch.compile` usage.

    Args:
        model (~torch.module.Module): Torch model to be evaluated.

    Returns:
        bool: ``True`` if model (or model wrapped in a compiled model) is a :class:`MinervaModel`
        or :class:`MinervaDataParallel`, else ``False``.

    .. versionadded:: 0.27
    """
    if isinstance(model, OptimizedModule):
        return isinstance(model._orig_mod, (MinervaModel, MinervaDataParallel))
    else:
        return isinstance(model, (MinervaModel, MinervaDataParallel))


def is_minerva_subtype(model: Module, subtype: type) -> bool:
    """
    Checks if model is a specific type while accounting for models that are
    :class:~`torch._dynamo.eval_frame.OptimizedModule` from :meth:`torch.compile` usage.

    Args:
        model (Module): Torch model to be evaluated.
        subtype (type): Type to check model against.

    Returns:
        bool: ``True`` if model (or model wrapped in a compiled model) is ``subtype`` else ``False``.

    .. versionadded:: 0.27
    """
    if isinstance(model, OptimizedModule):
        return isinstance(model._orig_mod, subtype)
    else:
        return isinstance(model, subtype)


def extract_wrapped_model(
    model: MinervaModel | MinervaDataParallel | OptimizedModule,
) -> MinervaModel:
    """
    Extracts the actual model object from within :class:`MinervaDataParallel` or
    :class:~`torch._dynamo.eval_frame.OptimizedModule` and returns.

    Args:
        model (MinervaModel | MinervaDataParallel | OptimizedModule): Model that may or may not be wrapped.

    Returns:
        MinervaModel: Extracted model.

    .. versionadded:: 0.27
    """
    if isinstance(model, OptimizedModule):
        _model = model._orig_mod
        assert isinstance(_model, (MinervaModel, MinervaDataParallel))
        model = _model

    if isinstance(model, MinervaDataParallel):  # pragma: no cover
        model = model.model.module

    assert isinstance(model, MinervaModel)
    return model


def wrap_model(model, gpu: int, torch_compile: bool = False):
    # Checks if multiple GPUs detected. If so, wraps model in DistributedDataParallel for multi-GPU use.
    if torch.cuda.device_count() > 1:  # pragma: no cover
        print(f"{torch.cuda.device_count()} GPUs detected")
        model = MinervaDataParallel(
            torch.nn.modules.SyncBatchNorm.convert_sync_batchnorm(  # type: ignore
                model
            ),
            DDP,
            device_ids=[gpu],
        )

    # Wraps the model in `torch.compile` to speed up computation time.
    if (
        Version(torch.__version__) > Version("2.0.0")
        and torch_compile
        and os.name != "nt"
    ):
        try:
            _compiled_model: OptimizedModule = torch.compile(
                model
            )  # type:ignore[assignment]
            assert is_minerva_model(_compiled_model)
            assert isinstance(_compiled_model, OptimizedModule)
            model = _compiled_model
        except RuntimeError as err:  # pragma: no cover
            warnings.warn(str(err))

    return model
