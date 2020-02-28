import os


def ensure_folder(dir_path, exist_ok=True):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=exist_ok)


def to_standard_box(bbox):
    """
    Transforms a given bounding box into a standardized representation.
    :param bbox: Bounding box given as [(x0, y0), (x1, y1)] or [x0, y0, x1, y1]
    :return: Bounding box represented as [x0, y0, x1, y1]
    """
    from typing import Iterable
    if isinstance(bbox[0], Iterable):
        bbox = [xy for p in bbox for xy in p]
    return bbox
