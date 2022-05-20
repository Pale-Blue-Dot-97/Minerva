"""Module containing custom samplers for `torch` datasets.

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
from typing import Tuple, Optional, Callable, Dict, Any, Iterable
from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import BoundingBox


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class PairedDataset(RasterDataset):
    def __init__(
        self,
        dataset_cls: Callable,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dataset = dataset_cls(*args, **kwargs)

    def __getitem__(
        self, queries: Tuple[BoundingBox, BoundingBox]
    ) -> Tuple[Dict[str, Any], ...]:
        return self.dataset.__getitem__(queries[0]), self.dataset.__getitem__(
            queries[1]
        )

    def __getattr__(self, item):
        if item in self.dataset.__dict__:
            return getattr(self.dataset, item)
        elif item in self.__dict__:
            return getattr(self, item)
        else:
            raise AttributeError

    def __repr__(self) -> str:
        return self.dataset.__repr__()
