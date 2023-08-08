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
r"""Tests for :mod:`minerva.transforms`.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2023 Harry Baker"

# =====================================================================================================================
#                                                      IMPORTS
# =====================================================================================================================
from typing import Any, Dict, List, Optional

import pytest
import torch
from numpy.testing import assert_array_equal
from pytest_lazyfixture import lazy_fixture
from torch import FloatTensor, LongTensor
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, RandomVerticalFlip

from minerva.transforms import (
    ClassTransform,
    DetachedColorJitter,
    MinervaCompose,
    Normalise,
    PairCreate,
    SingleLabel,
    SwapKeys,
    ToRGB,
)
from minerva.utils import utils


# =====================================================================================================================
#                                                       TESTS
# =====================================================================================================================
@pytest.mark.parametrize(
    ["input_mask", "output"],
    [
        (lazy_fixture("simple_mask"), torch.tensor([[1, 3, 0], [2, 0, 1], [1, 1, 1]])),
        (
            torch.tensor([[5, 3, 5], [4, 5, 1], [1, 3, 1]], dtype=torch.long),
            torch.tensor([[0, 3, 0], [2, 0, 1], [1, 3, 1]]),
        ),
    ],
)
def test_class_transform(
    example_matrix: Dict[int, int], input_mask: LongTensor, output: LongTensor
) -> None:
    transform = ClassTransform(example_matrix)

    assert_array_equal(output.numpy(), transform(input_mask).numpy())

    assert repr(transform) == f"ClassTransform(transform={example_matrix})"


@pytest.mark.parametrize(
    "sample", (42, lazy_fixture("simple_mask"), lazy_fixture("example_matrix"))
)
def test_pair_create(sample: Any) -> None:
    transform = PairCreate()

    assert transform(sample) == (sample, sample)
    assert repr(transform) == "PairCreate()"


@pytest.mark.parametrize("normalisation", (255, 65535))
@pytest.mark.parametrize(
    "mask",
    (
        lazy_fixture("simple_mask"),
        torch.tensor(
            [[1023.0, 3.890, 557.0], [478.0, 5.788, 10009.0], [1.0, 10240.857, 1458.7]]
        ),
    ),
)
def test_normalise(normalisation: int, mask) -> None:
    transform = Normalise(normalisation)
    assert_array_equal(transform(mask), mask / normalisation)
    assert repr(transform) == f"Normalise(norm_value={normalisation})"


def test_compose(simple_mask: LongTensor, simple_rgb_img: FloatTensor) -> None:
    transform_1 = Normalise(255)
    compose_1 = MinervaCompose(transform_1)

    with pytest.raises(TypeError):
        _ = compose_1(42)  # type: ignore[arg-type, call-overload]

    compose_2 = MinervaCompose(
        [transform_1, RandomHorizontalFlip(1.0), RandomVerticalFlip(1.0)]
    )

    input_1 = simple_mask.type(torch.float)

    wrong_compose = MinervaCompose(transforms=42)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        _ = wrong_compose(input_1)

    with pytest.raises(TypeError):
        _ = str(wrong_compose)

    output_1 = input_1 / 255
    output_2 = torch.tensor([[0.8, 1.0, 0.7], [0.3, 0.5, 0.4], [0.5, 0.0, 1.0]])  # type: ignore[attr-defined]

    input_3 = {"image": input_1}
    input_4 = {"image": simple_rgb_img}

    compose_3 = MinervaCompose(transform_1, key="image")
    compose_4 = MinervaCompose(
        [transform_1, RandomHorizontalFlip(1.0), RandomVerticalFlip(1.0)], key="image"
    )

    output_3 = {"image": output_1}
    output_4 = {"image": output_2}

    assert_array_equal(compose_1(input_1), output_1)
    assert_array_equal(compose_2(simple_rgb_img), output_2)
    assert_array_equal(compose_3(input_3)["image"], output_3["image"])
    assert_array_equal(compose_4(input_4)["image"], output_4["image"])

    assert repr(compose_1) == "MinervaCompose(Normalise(norm_value=255))"
    assert (
        repr(compose_2)
        == "MinervaCompose("
        + "\n    Normalise(norm_value=255)"
        + "\n    {0}".format(RandomHorizontalFlip(1.0))
        + "\n    {0}".format(RandomVerticalFlip(1.0))
        + "\n)"
    )


def test_detachedcolorjitter() -> None:
    transform = DetachedColorJitter(0.8, 0.8, 0.8, 0.2)
    colorjitter = ColorJitter(0.8, 0.8, 0.8, 0.2)  # type: ignore[type]

    img = torch.rand(4, 224, 224)
    img_rgb = img[:3]
    img_nir = img[3:]

    err_img = torch.rand(2, 224, 224)

    manual_detach = torch.cat((colorjitter(img_rgb), img_nir), 0)  # type: ignore[attr-defined]

    assert_array_equal(transform(img).size(), manual_detach.size())
    assert_array_equal(transform(img_rgb).size(), img_rgb.size())
    assert_array_equal(transform(img_nir).size(), img_nir.size())

    with pytest.raises(ValueError, match=r"\d channel images are not supported!"):
        transform(err_img)

    assert repr(transform) == f"Detached{repr(colorjitter)}"


def test_dublicator(
    simple_mask: LongTensor,
    simple_rgb_img: FloatTensor,
    norm_simple_rgb_img: FloatTensor,
) -> None:
    transform = (utils.dublicator(Normalise))(255)

    input_1 = simple_mask.type(torch.float)
    output_1 = input_1 / 255

    result_1, result_2 = transform((input_1, simple_rgb_img))

    assert_array_equal(result_1, output_1)
    assert_array_equal(result_2, norm_simple_rgb_img)

    assert repr(transform) == f"dublicator({repr(Normalise(255))})"


@pytest.mark.parametrize(
    ["transform", "keys", "args", "in_img", "expected"],
    [
        (
            Normalise,
            None,
            255,
            lazy_fixture("simple_rgb_img"),
            lazy_fixture("norm_simple_rgb_img"),
        ),
        (
            Normalise,
            ["image"],
            255,
            lazy_fixture("simple_rgb_img"),
            lazy_fixture("norm_simple_rgb_img"),
        ),
        (
            RandomHorizontalFlip,
            ["image", "mask"],
            1.0,
            lazy_fixture("simple_sample"),
            lazy_fixture("flipped_simple_sample"),
        ),
    ],
)
def test_tg_to_torch(
    transform, keys: Optional[List[str]], args: Any, in_img, expected
) -> None:
    transformation = (utils.tg_to_torch(transform, keys=keys))(args)

    if keys and len(keys) > 1:
        img = in_img.copy()
        result = transformation(img)
        for key in keys:
            assert_array_equal(result[key], expected[key])
    else:
        assert_array_equal(transformation(in_img), expected)

    with pytest.raises(TypeError):
        transformation(["wrongisimo!"])  # type: ignore[type]

    assert repr(transformation) == repr(transform(args))


def test_to_rgb(random_rgbi_tensor) -> None:
    transform_1 = ToRGB()

    assert_array_equal(transform_1(random_rgbi_tensor), random_rgbi_tensor[:3])
    assert repr(transform_1) == "ToRGB(channels --> [0:3])"

    with pytest.raises(
        ValueError, match="Image has less than 3 channels! Cannot be RGB!"
    ):
        _ = transform_1(random_rgbi_tensor[:2])

    transform_2 = ToRGB((1, 2, 3))

    assert_array_equal(transform_2(random_rgbi_tensor), random_rgbi_tensor[1:])
    assert repr(transform_2) == "ToRGB(channels --> [(1, 2, 3)])"

    with pytest.raises(
        ValueError, match="Image has less channels that trying to reduce to!"
    ):
        _ = transform_2(random_rgbi_tensor[:2])


def test_single_label(random_tensor_mask) -> None:
    transform_1 = SingleLabel()

    assert_array_equal(
        transform_1(random_tensor_mask),
        LongTensor([utils.find_tensor_mode(random_tensor_mask)]),
    )

    assert repr(transform_1) == "SingleLabel(mode=modal)"

    with pytest.raises(
        NotImplementedError, match="wrong mode is not a recognised operating mode!"
    ):
        transform_2 = SingleLabel("wrong mode")
        _ = transform_2(random_tensor_mask)


def test_swap_keys(random_rgbi_tensor, random_tensor_mask) -> None:
    transform = SwapKeys("mask", "image")

    in_sample = {"image": random_rgbi_tensor, "mask": random_tensor_mask}

    correct_out_sample = {"image": random_tensor_mask, "mask": random_tensor_mask}

    out_sample = transform(in_sample)

    assert_array_equal(out_sample["image"], correct_out_sample["image"])
    assert_array_equal(out_sample["mask"], correct_out_sample["mask"])
    assert repr(transform) == "SwapKeys(mask -> image)"
