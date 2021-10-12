from pathlib import Path
from datetime import datetime
import random
import glob
import pytz
from copy import deepcopy

import pydash as py_
import cv2
import imutils
import numpy as np

from src.utils.config import get_configs
from src.computer_vision.yolov4_main import predict

_CONFIG = get_configs()
_RAW_IMAGEPATH = Path(py_.get(_CONFIG, "raw_image_path"))
_PROCESSED_IMAGEPATH = Path(py_.get(_CONFIG, "processed_image_path"))
_TZ = pytz.timezone(py_.get(_CONFIG, "timezone"))


def read_image(file_path):
    img = cv2.imread(file_path)
    return img


def resize_image(image, factor=1.0, height=None, width=None, dim=None, restrict=False):
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
    if restrict is True:
        h, w, _ = image.shape
        if h <= 1500 and w <= 1500:
            return image


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


def crop_image(image, factor, pad=False):
    image = deepcopy(image)
    h, w, _ = image_shape(image)
    print(type(factor))
    if isinstance(factor, float):
        assert 0. <= factor < 0.5
        _crop_h = int(h*factor)
        _crop_w = int(w*factor)
    elif isinstance(factor, tuple):
        assert 0. <= factor[0] < 0.5
        assert 0. <= factor[1] < 0.5
        _crop_h = int(h*factor[0])
        _crop_w = int(w*factor[1])
    else:
        raise

    _end_h = h - _crop_h
    _end_w = w - _crop_w

    print(_crop_h, _end_h, _crop_w, _end_w)
    if pad is True:
        image[0:_crop_h] = image[h - _crop_h:h] = 255
        image[:, 0:_crop_w] = image[:, w - _crop_w:w] = 255
        return image
    else:
        image[_crop_h:_end_h, _crop_w:_end_w]
        return image


# Image Info
def image_shape(image):
    return image.shape


# Testing
def main(output=False):
    test_path = Path(random.choice(glob.glob(f"{_RAW_IMAGEPATH.as_posix()}/*")))
    image = read_image(test_path.as_posix())
    print(image_shape(image))

    image = resize_image(image, factor=3, restrict=True)
    print(image_shape(image))
    image_cropped = crop_image(image, (0.1, 0.1), pad=False)

    response = predict(image_cropped)
    _object = 0
    for i in response.iterrows():
        xx = i[1]
        print(xx.class_name)
        if xx.class_name not in ["bowl", "cup", "sandwich", "banana", "cake"]:
            continue

        cv2.rectangle(image_cropped, (xx.x1, xx.y1), (xx.x2, xx.y2), color=(0, 255, 0), thickness=1)
        cv2.putText(image_cropped,
                    org=(xx.x1 + 15, xx.y1+ 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    text=f"{xx.class_name}: {xx.score : 2f}",
                    color=(0, 0, 255),
                    thickness=2)

        _object += 1

    cv2.imshow("original", image)
    cv2.imshow("cropped_image", image_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if _object > 0 and output:
        output_image(image_cropped)
    else:
        pass


if __name__ == "__main__":
    main(output=True)

