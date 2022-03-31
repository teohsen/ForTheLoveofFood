import boto3
import cv2

from util.config import load_configuration

config = load_configuration()

try:
    rekog_client = boto3.client('rekognition',
                                aws_access_key_id=config.get("AWS_ACCESS_KEY_ID"),
                                aws_secret_access_key=config.get("AWS_SECRET_ACCESS_KEY"),
                                region_name=config.get("AWS_REGION"))
except Exception:
    raise
    
def detect_ppe(image):
    confidence = 50

    data = {
        "Image": {'Bytes': cv2.imencode('.png', image)[1].tobytes()},
        "SummarizationAttributes": {
            "MinConfidence": confidence,
            "RequiredEquipmentTypes": ["FACE_COVER", "HAND_COVER", "HEAD_COVER"]
        }
    }
    response = rekog_client.detect_protective_equipment(**data)
    
