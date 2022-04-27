"""Module containing custom transforms to be used with `torchvision.transforms`.

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
    * Document classes
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Dict, Tuple
from torch import Tensor
from minerva.utils import utils


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class ClassTransform:
    """Transform to be applied to a mask to convert from one labelling schema to another.

    Attributes:
        transform (Dict[int, int]): Mapping from one labelling schema to another.

    Args:
        transform (Dict[int, int]): Mapping from one labelling schema to another.
    """

    def __init__(self, transform: Dict[int, int]) -> None:
        self.transform = transform

    def __call__(self, mask: Tensor) -> Tensor:
        """Transforms the given mask from the original label schema to the new.

        Args:
            mask (Tensor): Mask in the original label schema.

        Returns:
            Tensor: Mask transformed into new label schema.
        """
        return utils.mask_transform(mask, self.transform)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transform={self.transform})"


class PairCreate:
    """Transform that takes a sample and returns a pair of the same sample."""

    def __init__(self) -> None:
        pass

    def __call__(self, sample: Any) -> Tuple[Any, Any]:
        """Takes a sample and returns it and a copy as a tuple pair.

        Args:
            sample (Any): Sample to duplicate.

        Returns:
            Tuple[Any, Any]: Tuple of two copies of the sample.
        """
        return sample, sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalise:
    """Transform that normalises an image tensor based on the bit size.

    Attributes:
        norm_value (int): Value to normalise image with.

    Args:
        norm_value (int): Value to normalise image with.
    """

    def __init__(self, norm_value: int) -> None:
        self.norm_value = norm_value

    def __call__(self, img: Tensor) -> Tensor:
        """Normalises inputted image using `norm_value`.

        Args:
            img (Tensor): Image tensor to be normalised. Should have a bit size that relates to `norm_value`.

        Returns:
            Tensor: Input image tensor normalised by `norm_value`.
        """
        return img / self.norm_value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(norm_value={self.norm_value})"
