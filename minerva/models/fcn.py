# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2023 Harry Baker

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
"""Module containing Fully Convolutional Network (FCN) models."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"

__all__ = [
    "FCN",
    "DCN",
    "FCN8ResNet18",
    "FCN8ResNet34",
    "FCN8ResNet50",
    "FCN8ResNet101",
    "FCN8ResNet152",
    "FCN16ResNet18",
    "FCN16ResNet34",
    "FCN16ResNet50",
    "FCN32ResNet18",
    "FCN32ResNet34",
    "FCN32ResNet50",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.modules as nn
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler

from .core import MinervaBackbone, MinervaModel, bilinear_init, get_model


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class FCN(MinervaBackbone):
    """Base Fully Convolutional Network (FCN) class to be subclassed by FCN variants described in the FCN paper.

    Based on the example found here: https://github.com/haoran1062/FCN-pytorch/blob/master/FCN.py

    Subclasses :class:`~models.MinervaModel`.

    Attributes:
        backbone_name (str): Optional; Name of the backbone within this module to use for the FCN.
        decoder_variant (str): Optional; Flag for which DCN variant to construct.
            Must be either ``'32'``, ``'16'`` or ``'8'``. See the FCN paper for details on these variants.
        backbone (~torch.nn.Module): Backbone of the FCN that takes the imagery input and
            extracts learned representations.
        decoder (~torch.nn.Module): Decoder that takes the learned representations from the backbone encoder
            and de-convolves to output a classification segmentation mask.

    Args:
        criterion: :mod:`torch` loss function model will use.
        input_size (tuple[int] | list[int]): Optional; Defines the shape of the input data in
            order of number of channels, image width, image height.
        n_classes (int): Optional; Number of classes in data to be classified.
        batch_size (int): Optional; Number of samples in each batch supplied to the network.
            Only needed for Decoder, not DCN.
        backbone_weight_path (str): Optional; Path to pre-trained weights for the backbone to be loaded.
        freeze_backbone (bool): Freezes the weights on the backbone to prevent end-to-end training
            if using a pre-trained backbone.
        backbone_kwargs (dict[str, ~typing.Any]): Optional; Keyword arguments for the backbone packed up into a dict.
    """

    backbone_name: str = "ResNet18"
    decoder_variant: Literal["32", "16", "8"] = "32"

    def __init__(
        self,
        criterion: Any,
        input_size: Tuple[int, ...] = (4, 256, 256),
        n_classes: int = 8,
        scaler: Optional[GradScaler] = None,
        backbone_weight_path: Optional[str] = None,
        freeze_backbone: bool = False,
        backbone_kwargs: Dict[str, Any] = {},
    ) -> None:
        super(FCN, self).__init__(
            criterion=criterion,
            input_size=input_size,
            n_classes=n_classes,
            scaler=scaler,
        )

        # Initialises the selected Minerva backbone.
        self.backbone: MinervaModel = get_model(self.backbone_name)(
            input_size=input_size, n_classes=n_classes, encoder=True, **backbone_kwargs  # type: ignore
        )

        # Loads and graphts the pre-trained weights ontop of the backbone if the path is provided.
        if backbone_weight_path is not None:  # pragma: no cover
            self.backbone.load_state_dict(torch.load(backbone_weight_path))

            # Freezes the weights of backbone to avoid end-to-end training.
            if freeze_backbone:
                self.backbone.requires_grad_(False)

        # Determines the output shape of the backbone so the correct input shape is known
        # for the proceeding layers of the network.
        self.backbone.determine_output_dim()

        self._make_dcn()

    def _make_dcn(self) -> None:
        backbone_out_shape = self.backbone.output_shape
        assert isinstance(backbone_out_shape, Sequence)
        self.decoder = DCN(
            in_channel=backbone_out_shape[0],
            n_classes=self.n_classes,
            variant=self.decoder_variant,
        )

    def _remake_classifier(self) -> None:
        self._make_dcn()

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass of the FCN by using the forward methods of the backbone and
        feeding its output into the forward for the decoder.

        Can be called directly as a method (e.g. ``model.forward()``)
        or when data is parsed to model (e.g. ``model()``).

        Args:
            x (~torch.Tensor): Input data to network.

        Returns:
            ~torch.Tensor: segmentation mask with a channel for each class of the likelihoods the network places on
            each pixel input ``x`` being of that class.
        """
        z = self.backbone(x)
        z = self.decoder(z)

        assert isinstance(z, Tensor)
        return z


class DCN(MinervaModel):
    """Generic DCN defined by the FCN paper. Can construct the DCN32, DCN16 or DCN8 variants defined in the paper.

    Based on the example found here: https://github.com/haoran1062/FCN-pytorch/blob/master/FCN.py

    Attributes:
        variant (~typing.Literal['32', '16', '8']): Defines which DCN variant this object is, altering the
            layers constructed and the computational graph. Will be either ``'32'``, ``'16'`` or ``'8'``.
            See the FCN paper for details on these variants.
        n_classes (int): Number of classes in dataset. Defines number of output classification channels.
        relu (~torch.nn.ReLU): Rectified Linear Unit (ReLU) activation layer to be used throughout the network.
        Conv1x1 (~torch.nn.Conv2d): First Conv1x1 layer acting as input to the network from the final output of
            the encoder and common to all variants.
        bn1 (~torch.nn.BatchNorm2d): First batch norm layer common to all variants that comes after Conv1x1.
        DC32 (~torch.nn.ConvTranspose2d): De-convolutional layer with stride 32 for DCN32 variant.
        dbn32 (~torch.nn.BatchNorm2d): Batch norm layer after DC32.
        Conv1x1_x3 (~torch.nn.Conv2d): Conv1x1 layer acting as input to the network taking the output from the
            third layer from the ResNet encoder.
        DC2 (~torch.nn.ConvTranspose2d): De-convolutional layer with stride 2 for DCN16 & DCN8 variants.
        dbn2 (~torch.nn.BatchNorm2d): Batch norm layer after DC2.
        DC16 (~torch.nn.ConvTranspose2d): De-convolutional layer with stride 16 for DCN16 variant.
        dbn16 (~torch.nn.BatchNorm2d): Batch norm layer after DC16.
        Conv1x1_x2 (~torch.nn.Conv2d): Conv1x1 layer acting as input to the network taking the output from the
            second layer from the ResNet encoder.
        DC4 (~torch.nn.ConvTranspose2d): De-convolutional layer with stride 2 for DCN8 variant.
        dbn4 (~torch.nn.BatchNorm2d): Batch norm layer after DC4.
        DC8 (~torch.nn.ConvTranspose2d): De-convolutional layer with stride 8 for DCN8 variant.
        dbn8 (~torch.nn.BatchNorm2d): Batch norm layer after DC8.

    Args:
        in_channel (int): Optional; Number of channels in the input layer of the network.
            Should match the number of output channels (likely feature maps) from the encoder.
        n_classes (int): Optional; Number of classes in dataset. Defines number of output classification channels.
        variant (~typing.Literal['32', '16', '8']): Optional; Flag for which DCN variant to construct.
            Must be either ``'32'``, ``'16'`` or ``'8'``. See the FCN paper for details on these variants.

    Raises:
        NotImplementedError: Raised if ``variant`` does not match known types.
    """

    def __init__(
        self,
        in_channel: int = 512,
        n_classes: int = 21,
        variant: Literal["32", "16", "8"] = "32",
    ) -> None:
        super(DCN, self).__init__(n_classes=n_classes)
        self.variant: Literal["32", "16", "8"] = variant

        assert type(self.n_classes) is int

        # Common to all variants.
        self.relu = nn.ReLU(inplace=True)
        self.Conv1x1 = nn.Conv2d(in_channel, self.n_classes, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(self.n_classes)

        if self.variant == "32":
            self.DC32 = nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(64, 64),
                stride=(32, 32),
                dilation=1,
                padding=(16, 16),
            )
            self.DC32.weight.data = bilinear_init(self.n_classes, self.n_classes, 64)
            self.dbn32 = nn.BatchNorm2d(self.n_classes)

        if self.variant in ("16", "8"):
            self.Conv1x1_x3 = nn.Conv2d(
                int(in_channel / 2), self.n_classes, kernel_size=(1, 1)
            )
            self.DC2 = nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(4, 4),
                stride=(2, 2),
                dilation=1,
                padding=(1, 1),
            )
            self.DC2.weight.data = bilinear_init(self.n_classes, self.n_classes, 4)
            self.dbn2 = nn.BatchNorm2d(self.n_classes)

        if self.variant == "16":
            self.DC16 = nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(32, 32),
                stride=(16, 16),
                dilation=1,
                padding=(8, 8),
            )
            self.DC16.weight.data = bilinear_init(self.n_classes, self.n_classes, 32)
            self.dbn16 = nn.BatchNorm2d(self.n_classes)

        if self.variant == "8":
            self.Conv1x1_x2 = nn.Conv2d(
                int(in_channel / 4), self.n_classes, kernel_size=(1, 1)
            )

            self.DC4 = nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(4, 4),
                stride=(2, 2),
                dilation=1,
                padding=(1, 1),
            )
            self.DC4.weight.data = bilinear_init(self.n_classes, self.n_classes, 4)
            self.dbn4 = nn.BatchNorm2d(self.n_classes)

            self.DC8 = nn.ConvTranspose2d(
                self.n_classes,
                self.n_classes,
                kernel_size=(16, 16),
                stride=(8, 8),
                dilation=1,
                padding=(4, 4),
            )
            self.DC8.weight.data = bilinear_init(self.n_classes, self.n_classes, 16)
            self.dbn8 = nn.BatchNorm2d(self.n_classes)

        if self.variant not in ("32", "16", "8"):
            raise NotImplementedError(
                f"Variant {self.variant} does not match known types"
            )

    def forward(self, x: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """Performs a forward pass of the decoder. Depending on DCN variant, will take multiple inputs
        throughout pass from the encoder.

        Can be called directly as a method (e.g. ``model.forward()``)
        or when data is parsed to model (e.g. ``model()``).

        Args:
            x (tuple[~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor, ~torch.Tensor]): Input data to network.
                Should be from a backbone that supports output at multiple points e.g ResNet.

        Returns:
            ~torch.Tensor:  Segmentation mask with a channel for each class of the likelihoods the network places on
            each pixel input ``x`` being of that class.

        Raises:
            NotImplementedError: Raised if ``variant`` does not match known types.
        """
        if self.variant not in ("32", "16", "8"):
            raise NotImplementedError(
                f"Variant {self.variant} does not match known types"
            )

        # Unpack outputs from the ResNet layers.
        x4, x3, x2, *_ = x

        # All DCNs have a common 1x1 Conv input block.
        z = self.bn1(self.relu(self.Conv1x1(x4)))

        # If DCN32, forward pass through DC32 and DBN32 and return output.
        if self.variant == "32":
            z = self.dbn32(self.relu(self.DC32(z)))
            assert isinstance(z, Tensor)
            return z

        # Common Conv1x1 layer to DCN16 & DCN8.
        x3 = self.bn1(self.relu(self.Conv1x1_x3(x3)))
        z = self.dbn2(self.relu(self.DC2(z)))

        z = z + x3

        # If DCN16, forward pass through DCN16 and DBN16 and return output.
        if self.variant == "16":
            z = self.dbn16(self.relu(self.DC16(z)))
            assert isinstance(z, Tensor)
            return z

        # If DCN8, continue through remaining layers to output.
        else:
            x2 = self.bn1(self.relu(self.Conv1x1_x2(x2)))
            z = self.dbn4(self.relu(self.DC4(z)))

            z = z + x2

            z = self.dbn8(self.relu(self.DC8(z)))

            assert isinstance(z, Tensor)
            return z


class FCN32ResNet18(FCN):
    """
    Fully Convolutional Network (FCN) using a :class:`~models.resnet.ResNet18` backbone
    with a ``DCN32`` decoder.
    """

    backbone_name = "ResNet18"
    decoder_variant = "32"


class FCN32ResNet34(FCN):
    """
    Fully Convolutional Network (FCN) using a :class:`~models.resnet.ResNet34` backbone
    with a ``DCN32`` decoder.
    """

    backbone_name = "ResNet34"
    decoder_variant = "32"


class FCN32ResNet50(FCN):
    """
    Fully Convolutional Network (FCN) using a :class:`~models.resnet.ResNet50` backbone
    with a ``DCN32`` decoder.
    """

    backbone_name = "ResNet50"
    decoder_variant = "32"


class FCN16ResNet18(FCN):
    """
    Fully Convolutional Network (FCN) using a :class:`~models.resnet.ResNet18` backbone
    with a ``DCN16`` decoder.
    """

    backbone_name = "ResNet18"
    decoder_variant = "16"


class FCN16ResNet34(FCN):
    """
    Fully Convolutional Network (FCN) using a :class:`~models.resnet.ResNet34` backbone
    with a ``DCN16`` decoder.
    """

    backbone_name = "ResNet34"
    decoder_variant = "16"


class FCN16ResNet50(FCN):
    """
    Fully Convolutional Network (FCN) using a :class:`~models.resnet.ResNet50` backbone
    with a ``DCN16`` decoder.
    """

    backbone_name = "ResNet50"
    decoder_variant = "16"


class FCN8ResNet18(FCN):
    """
    Fully Convolutional Network (FCN) using a :class:`~models.resnet.ResNet18` backbone
    with a ``DCN8`` decoder.
    """

    backbone_name = "ResNet18"
    decoder_variant = "8"


class FCN8ResNet34(FCN):
    """
    Fully Convolutional Network (FCN) using a :class:`~models.resnet.ResNet34` backbone
    with a ``DCN8`` decoder.
    """

    backbone_name = "ResNet34"
    decoder_variant = "8"


class FCN8ResNet50(FCN):
    """
    Fully Convolutional Network (FCN) using a :class:`~models.resnet.ResNet50` backbone
    with a ``DCN8`` decoder.
    """

    backbone_name = "ResNet50"
    decoder_variant = "8"


class FCN8ResNet101(FCN):
    """
    Fully Convolutional Network (FCN) using a :class:`~models.resnet.ResNet101` backbone
    with a ``DCN8`` decoder.
    """

    backbone_name = "ResNet101"
    decoder_variant = "8"


class FCN8ResNet152(FCN):
    """
    Fully Convolutional Network (FCN) using a :class:`~models.resnet.ResNet152` backbone
    with a ``DCN8`` decoder.
    """

    backbone_name = "ResNet152"
    decoder_variant = "8"
