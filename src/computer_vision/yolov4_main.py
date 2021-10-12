from pathlib import Path

from models.yolo.yolov4 import Yolov4
from src.utils.config import get_configs

config = get_configs()

_YOLOV4_WEIGHTS = Path(config.get("v4weights"))
_YOLOV4_COCO = Path(config.get("v4coco"))

model_yolov4 = Yolov4(weight_path=_YOLOV4_WEIGHTS.as_posix(),
                      class_name_path=_YOLOV4_COCO.as_posix())


def predict(image):
    return model_yolov4.predict(image)
