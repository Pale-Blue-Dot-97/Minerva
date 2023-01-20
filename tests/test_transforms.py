import pytest
import torch
from numpy.testing import assert_array_equal
from torch import LongTensor
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, RandomVerticalFlip

from minerva.transforms import (
    ClassTransform,
    DetachedColorJitter,
    MinervaCompose,
    Normalise,
    PairCreate,
)
from minerva.utils import utils


def test_class_transform(simple_mask, example_matrix) -> None:

    transform = ClassTransform(example_matrix)

    output_1 = torch.tensor([[1, 3, 0], [2, 0, 1], [1, 1, 1]])  # type: ignore[attr-defined]

    input_2: LongTensor = torch.tensor(  # type: ignore[attr-defined, assignment]
        [[5, 3, 5], [4, 5, 1], [1, 3, 1]], dtype=torch.long  # type: ignore[attr-defined]
    )

    output_2 = torch.tensor([[0, 3, 0], [2, 0, 1], [1, 3, 1]])  # type: ignore[attr-defined]

    assert_array_equal(output_1.numpy(), transform(simple_mask).numpy())
    assert_array_equal(output_2.numpy(), transform(input_2).numpy())

    assert repr(transform) == f"ClassTransform(transform={example_matrix})"


def test_pair_create(simple_mask, example_matrix) -> None:
    transform = PairCreate()
    sample_1 = 42

    assert transform(sample_1) == (sample_1, sample_1)
    assert transform(simple_mask) == (simple_mask, simple_mask)
    assert transform(example_matrix) == (example_matrix, example_matrix)

    assert repr(transform) == "PairCreate()"


def test_normalise(simple_mask) -> None:
    transform_1 = Normalise(255)
    transform_2 = Normalise(65535)

    input_1 = simple_mask.type(torch.FloatTensor)

    input_2 = torch.tensor(  # type: ignore[attr-defined]
        [[1023.0, 3.890, 557.0], [478.0, 5.788, 10009.0], [1.0, 10240.857, 1458.7]]
    )

    assert_array_equal(transform_1(input_1), input_1 / 255)
    assert_array_equal(transform_1(input_2), input_2 / 255)
    assert_array_equal(transform_2(input_1), input_1 / 65535)
    assert_array_equal(transform_2(input_2), input_2 / 65535)

    assert repr(transform_1) == "Normalise(norm_value=255)"
    assert repr(transform_2) == "Normalise(norm_value=65535)"


def test_compose(simple_mask) -> None:
    transform_1 = Normalise(255)
    compose_1 = MinervaCompose(transform_1)

    with pytest.raises(TypeError):
        _ = compose_1(42)  # type: ignore[arg-type, call-overload]

    compose_2 = MinervaCompose(
        [transform_1, RandomHorizontalFlip(1.0), RandomVerticalFlip(1.0)]
    )

    input_1 = simple_mask.type(torch.FloatTensor)

    wrong_compose = MinervaCompose(transforms=42)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        _ = wrong_compose(input_1)

    with pytest.raises(TypeError):
        _ = str(wrong_compose)

    input_2 = torch.tensor(  # type: ignore[attr-defined]
        [[255.0, 0.0, 127.5], [102.0, 127.5, 76.5], [178.5, 255.0, 204.0]]
    )

    output_1 = input_1 / 255
    output_2 = torch.tensor([[0.8, 1.0, 0.7], [0.3, 0.5, 0.4], [0.5, 0.0, 1.0]])  # type: ignore[attr-defined]

    input_3 = {"image": input_1}
    input_4 = {"image": input_2}

    compose_3 = MinervaCompose(transform_1, key="image")
    compose_4 = MinervaCompose(
        [transform_1, RandomHorizontalFlip(1.0), RandomVerticalFlip(1.0)], key="image"
    )

    output_3 = {"image": output_1}
    output_4 = {"image": output_2}

    assert_array_equal(compose_1(input_1), output_1)
    assert_array_equal(compose_2(input_2), output_2)
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
    transform_1 = DetachedColorJitter(0.8, 0.8, 0.8, 0.2)
    colorjitter_1 = ColorJitter(0.8, 0.8, 0.8, 0.2)  # type: ignore[type]

    img = torch.rand(4, 224, 224)
    img_rgb = img[:3]
    img_nir = img[3:]

    err_img = torch.rand(2, 224, 224)

    manual_detach = torch.cat((colorjitter_1(img_rgb), img_nir), 0)  # type: ignore[attr-defined]

    assert_array_equal(transform_1(img).size(), manual_detach.size())
    assert_array_equal(transform_1(img_rgb).size(), img_rgb.size())
    assert_array_equal(transform_1(img_nir).size(), img_nir.size())

    with pytest.raises(ValueError, match=r"\d channel images are not supported!"):
        transform_1(err_img)

    assert repr(transform_1) == f"Detached{repr(colorjitter_1)}"


def test_dublicator(simple_mask) -> None:
    transform_1 = (utils.dublicator(Normalise))(255)

    input_1 = simple_mask.type(torch.FloatTensor)

    input_2 = torch.tensor(  # type: ignore[attr-defined]
        [[255.0, 0.0, 127.5], [102.0, 127.5, 76.5], [178.5, 255.0, 204.0]]
    )

    output_1 = input_1 / 255
    output_2 = input_2 / 255

    result_1, result_2 = transform_1((input_1, input_2))

    assert_array_equal(result_1, output_1)
    assert_array_equal(result_2, output_2)

    assert repr(transform_1) == f"dublicator({repr(Normalise(255))})"


def test_tg_to_torch(simple_mask) -> None:
    transform_1 = (utils.tg_to_torch(Normalise))(255)

    transform_2 = (utils.tg_to_torch(Normalise, keys=["image"]))(255)

    transform_3 = (utils.tg_to_torch(RandomHorizontalFlip, keys=["image", "mask"]))(1.0)

    img = torch.tensor(  # type: ignore[attr-defined]
        [[255.0, 0.0, 127.5], [102.0, 127.5, 76.5], [178.5, 255.0, 204.0]]
    )

    out_img = torch.tensor(  # type: ignore[attr-defined]
        [[127.5, 0.0, 255.0], [76.5, 127.5, 102.0], [204.0, 255.0, 178.5]]
    )

    out_mask = torch.tensor([[5, 3, 1], [1, 5, 4], [1, 1, 1]])  # type: ignore[attr-defined]

    input_3 = {"image": img, "mask": simple_mask}
    output_3 = {"image": out_img, "mask": out_mask}

    result_2 = transform_2({"image": img})
    result_3 = transform_3(input_3)

    assert_array_equal(transform_1(img), img / 255)  # type: ignore[type]
    assert_array_equal(result_2["image"], img / 255)
    assert_array_equal(result_3["image"], output_3["image"])
    assert_array_equal(result_3["mask"], output_3["mask"])

    input_4 = ["wrongisimo!"]

    with pytest.raises(TypeError):
        transform_1(input_4)  # type: ignore[type]

    assert repr(transform_1) == repr(Normalise(255))
