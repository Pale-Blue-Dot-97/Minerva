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
r"""Collation functions designed for :mod:`minerva`."""

# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"
__all__ = [
    "get_collator",
    "stack_sample_pairs",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Callable, Iterable

from hydra.utils import get_method
from torchgeo.datasets.utils import stack_samples


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_collator(
    collator_target: str = "torchgeo.datasets.stack_samples",
) -> Callable[..., Any]:
    """Gets the function defined in parameters to collate samples together to form a batch.

    Args:
        collator_target (str): Dot based import path for collator method.
            Defaults to :meth:`torchgeo.datasets.stack_samples`

    Returns:
        ~typing.Callable[..., ~typing.Any]: Collation function found from target path given.
    """
    collator: Callable[..., Any]
    collator = get_method(collator_target)
    assert callable(collator)
    return collator


def stack_sample_pairs(
    samples: Iterable[tuple[dict[Any, Any], dict[Any, Any]]],
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Takes a list of paired sample dicts and stacks them into a tuple of batches of sample dicts.

    Args:
        samples (~typing.Iterable[tuple[dict[~typing.Any, ~typing.Any], dict[~typing.Any, ~typing.Any]]]): List of
            paired sample dicts to be stacked.

    Returns:
        tuple[dict[~typing.Any, ~typing.Any], dict[~typing.Any, ~typing.Any]]: Tuple of batches within dicts.
    """
    a, b = tuple(zip(*samples))
    return stack_samples(a), stack_samples(b)
