import hashlib
from datetime import datetime

from src.computer_vision.detect_food import *
from src.computer_vision.detect_food import _CUSTOM_WEIGHTS, _CUSTOM_CONFIG, _CUSTOM_CLASSES, _TESTOUT

shibasushi = YoloV4Custom(_CUSTOM_WEIGHTS.as_posix(), _CUSTOM_CONFIG.as_posix(), _CUSTOM_CLASSES)
output_path = Path(_TESTOUT)
# Unit Test


def test_it(image_path, output=False):
    image = cv2.imread(image_path.as_posix())
    classes, confi, bbox = shibasushi.predict(image, nmsthreshold=0.1)
    output_image = shibasushi.draw(image, classes, confi, bbox)

    cv2.imshow("image", cv2.resize(image, (600, 400)))
    cv2.imshow("output_image", cv2.resize(output_image, (600, 400)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output:
        tmp_name = hashlib.sha256(datetime.now().__str__().encode("utf-8")).hexdigest()
        cv2.imwrite(f"{output_path.joinpath(tmp_name).with_suffix('.jpg').as_posix()}", output_image)


filepath = Path("data/test").glob("*")
for file in filepath:
    test_it(file, output=True)
