from pathlib import Path
from datetime import datetime
import random
import glob
import pydash as py_
import pytz
import cv2
import imutils

from src.utils.config import get_configs


_CONFIG = get_configs()
_RAW_IMAGEPATH = Path(py_.get(_CONFIG, "raw_image_path"))
_PROCESSED_IMAGEPATH = Path(py_.get(_CONFIG, "processed_image_path"))
_TZ = pytz.timezone(py_.get(_CONFIG, "timezone"))


def read_image(file_path):
    img = cv2.imread(file_path)
    return img


def resize_image(image, factor=1.0, height=None, width=None, dim=None):
    """
    Some references
    1. More info on interpolation - https://chadrick-kwag.net/cv2-resize-interpolation-methods/


    :param image: incoming image
    :param factor: Scale factor if providedW
    :param height: Desired Height if provided
    :param width: Desired width if provided
    :param dim: Desired dimension input
    :return:
    """
    if dim is not None:
        return cv2.resize(image, dim)

    if height is not None and width is not None:
        # h, w is incoming image resolution
        # height and width is param input
        # Determine appro interpolation
        # Scenarios to consider
        # h >= height + w >= width   --> INTER_LINEAR or INTER_CUBIC
        # h <= height + w >= width   --> ?
        # h >= height + w <= width   --> ?
        # h <= height + w <= width   --> INTER_AREA
        return imutils.resize(image, height=height, width=width)

    elif height is not None:
        return imutils.resize(image, height=height)
    elif width is not None:
        return imutils.resize(image, width=width)
    else:
        h, w, _ = image.shape
        if factor >= 1.0:
            interpol = cv2.INTER_AREA
        else:
            interpol = cv2.INTER_CUBIC

        return imutils.resize(image, height=int(h//factor), width=int(w//factor), inter=interpol)


def resize_instaimage(image, dim=(1080, 1080)):
    return resize_image(image, dim=dim)


def output_image(image,
                 image_name=datetime.now(_TZ).__format__("%Y%m%d_%H%M%S"),
                 output_path=_PROCESSED_IMAGEPATH, ext=".jpg"):

    image_name = Path(image_name)
    if image_name.suffix == '':
        image_name = image_name.with_suffix(ext)
    else:
        pass

    output_path = Path(output_path).joinpath(image_name)
    cv2.imwrite(output_path.as_posix(), image)

    assert output_path.exists()


# Testing
def test():
    test_path = Path(random.choice(glob.glob(f"{_RAW_IMAGEPATH.as_posix()}/*")))

    # image = resize_instaimage(read_image(test_path))
    image = resize_image(read_image(test_path.as_posix()), factor=4)

    output_image(image)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()

