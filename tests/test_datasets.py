from torchgeo.datasets.utils import BoundingBox

from minerva.datasets import make_bounding_box


def test_make_bounding_box() -> None:
    assert make_bounding_box() is None
    assert make_bounding_box(False) is None

    bbox = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    assert make_bounding_box(bbox) == BoundingBox(*bbox)
