from pathlib import Path
import cv2
from src.utils.config import get_configs

_CONFIG = get_configs()

_CUSTOM_WEIGHTS = Path(_CONFIG.get("shibasushi_weights"))
_CUSTOM_CONFIG = Path(_CONFIG.get("shibasushi_cfg"))
_CUSTOM_CLASSES = _CONFIG.get("shibasushi_class")
_DIM = int(_CONFIG.get("shibasushi_dim"))
_TESTOUT = _CONFIG.get("cv_output")


class YoloV4Custom:

    def __init__(self, config, weight, classnames):
        self.class_list = self.load_class(classnames)
        self.model = self.load_model(config, weight)

    def load_class(self, classnames):
        with open(classnames, "rt") as f:
            classes = f.read().rstrip('\n').split('\n')

        return classes

    def load_model(self, config, weights):
        net = cv2.dnn.readNetFromDarknet(weights, config)
        model = cv2.dnn_DetectionModel(net)
        model.setInputSize(416, 416)
        model.setInputScale(1.0 / 255)
        model.setInputSwapRB(True)
        return model

    def predict(self, image, threshold=0.3, nmsthreshold=0.35):
        image = image.copy()
        classes, confidence, boxes = self.model.detect(image,
                                                     confThreshold=threshold,
                                                     nmsThreshold=nmsthreshold)

        return classes, confidence, boxes

    def draw(self, image, classes, confidence, boxes):
        classes = classes.flatten()
        confidence = confidence.flatten()

        image = image.copy()

        for i in range(len(classes)):
            _class = int(classes[i])
            _label = self.class_list[_class].upper()
            _confidence = confidence[i]
            label_text = f"{_label} - {_confidence:.2f}"
            x, y, w, h = boxes[i]
            print(label_text, x, y, h, w)
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=4)
            if y >= 50:
                cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, thickness=7,
                            color=(255, 255, 0))
            else:
                cv2.putText(image, label_text, (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, thickness=7,
                            color=(255, 255, 0))

        return image

