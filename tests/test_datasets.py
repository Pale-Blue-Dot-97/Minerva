from minerva.datasets import make_bounding_box
from torchgeo.datasets.utils import BoundingBox


def test_make_bounding_box() -> None:
    assert make_bounding_box() is None
    assert make_bounding_box(False) is None

    bbox = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    assert make_bounding_box(bbox) == BoundingBox(*bbox)
