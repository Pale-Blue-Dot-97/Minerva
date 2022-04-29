from minerva.transforms import ClassTransform, PairCreate, Normalise
import numpy as np
from numpy.testing import assert_array_equal
import torch


def test_class_transform() -> None:
    matrix = {1: 1, 3: 3, 4: 2, 5: 0}

    transform = ClassTransform(matrix)

    input_1 = torch.tensor([[1, 3, 5], [4, 5, 1], [1, 1, 1]])

    output_1 = torch.tensor([[1, 3, 0], [2, 0, 1], [1, 1, 1]])

    input_2 = torch.tensor([[5, 3, 5], [4, 5, 1], [1, 3, 1]])

    output_2 = torch.tensor([[0, 3, 0], [2, 0, 1], [1, 3, 1]])

    assert assert_array_equal(output_1.numpy(), transform(input_1).numpy()) is None
    assert assert_array_equal(output_2.numpy(), transform(input_2).numpy()) is None

    assert repr(transform) == f"ClassTransform(transform={matrix})"


def test_pair_create() -> None:
    transform = PairCreate()
    sample_1 = 42
    sample_2 = torch.tensor([[1, 3, 5], [4, 5, 1], [1, 1, 1]])
    sample_3 = {1: 1, 3: 3, 4: 2, 5: 0}

    assert transform(sample_1) == (sample_1, sample_1)
    assert transform(sample_2) == (sample_2, sample_2)
    assert transform(sample_3) == (sample_3, sample_3)

    assert repr(transform) == "PairCreate()"


def test_normalise() -> None:
    transform_1 = Normalise(255)
    transform_2 = Normalise(65535)

    input_1 = torch.tensor([[1.0, 3.0, 5.0], [4.0, 5.0, 1.0], [1.0, 1.0, 1.0]])

    input_2 = torch.tensor(
        [[1023.0, 3.890, 557.0], [478.0, 5.788, 10009.0], [1.0, 10240.857, 1458.7]]
    )

    assert assert_array_equal(transform_1(input_1), input_1 / 255) is None
    assert assert_array_equal(transform_1(input_2), input_2 / 255) is None
    assert assert_array_equal(transform_2(input_1), input_1 / 65535) is None
    assert assert_array_equal(transform_2(input_2), input_2 / 65535) is None

    assert repr(transform_1) == f"Normalise(norm_value=255)"
    assert repr(transform_2) == f"Normalise(norm_value=65535)"
