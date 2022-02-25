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

Attributes:

TODO:
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Dict
from Minerva.utils import utils


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class ClassTransform:
    def __init__(self, transform: Dict[int, int]) -> None:
        self.transform = transform

    def __call__(self, sample: Dict[Any, Any]) -> Dict[Any, Any]:
        mask = sample.pop('mask')

        new_mask = utils.mask_transform(mask, self.transform)
        
        sample['mask'] = new_mask
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transform={self.transform})"
