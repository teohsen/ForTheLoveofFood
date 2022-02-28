import hashlib
from datetime import datetime

from src.computer_vision.detect_food import *
from src.computer_vision.detect_food import _CONFIG, _CUSTOM_WEIGHTS, _CUSTOM_CONFIG, _CUSTOM_CLASSES, _TESTOUT

shibasushi = YoloV4Custom(_CUSTOM_WEIGHTS.as_posix(), _CUSTOM_CONFIG.as_posix(), _CUSTOM_CLASSES)
output_path = Path(_TESTOUT)
# Unit Test


def test_it(image_path, output=False):
    image = cv2.imread(image_path)
    classes, confi, bbox = shibasushi.predict(image)
    output_image = shibasushi.draw(image, classes, confi, bbox)
    cv2.imshow("image", image)
    cv2.imshow("output_image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output:
        tmp_name = hashlib.sha256(datetime.now().__str__().encode("utf-8")).hexdigest()
        cv2.imwrite(f"{output_path.joinpath(tmp_name).with_suffix('.jpg')}", output_image)

    return output_image, classes, confi, bbox


test_list = [
    r"test\photo_2021-10-25_18-39-01.jpg"
]

processed_image, predictions = test_it(test_list[0])
