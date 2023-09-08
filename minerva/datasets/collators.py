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
r"""Collation functions designed for :mod:`minerva`."""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"
__all__ = [
    "get_collator",
    "stack_sample_pairs",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from torchgeo.datasets.utils import stack_samples

from minerva.utils import utils


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_collator(
    collator_params: Optional[Dict[str, str]] = None
) -> Callable[..., Any]:
    """Gets the function defined in parameters to collate samples together to form a batch.

    Args:
        collator_params (dict[str, str]): Optional; Dictionary that must contain keys for
            ``'module'`` and ``'name'`` of the collation function. Defaults to ``config['collator']``.

    Returns:
        ~typing.Callable[..., ~typing.Any]: Collation function found from parameters given.
    """
    collator: Callable[..., Any]
    if collator_params is not None:
        module = collator_params.pop("module", "")
        if module == "":
            collator = globals()[collator_params["name"]]
        else:
            collator = utils.func_by_str(module, collator_params["name"])
    else:
        collator = stack_samples

    assert callable(collator)
    return collator


def stack_sample_pairs(
    samples: Iterable[Tuple[Dict[Any, Any], Dict[Any, Any]]]
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """Takes a list of paired sample dicts and stacks them into a tuple of batches of sample dicts.

    Args:
        samples (~typing.Iterable[tuple[dict[~typing.Any, ~typing.Any], dict[~typing.Any, ~typing.Any]]]): List of
            paired sample dicts to be stacked.

    Returns:
        tuple[dict[~typing.Any, ~typing.Any], dict[~typing.Any, ~typing.Any]]: Tuple of batches within dicts.
    """
    a, b = tuple(zip(*samples))
    return stack_samples(a), stack_samples(b)
