# ===== sklearn backward-compat patch =====
import sys
import sklearn.preprocessing
sys.modules['sklearn.preprocessing.data'] = sklearn.preprocessing
# ========================================

import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

CLASS_DICT_PATH = os.path.join(ARTIFACTS_DIR, "class_dictionary.json")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "saved_model.pkl")

# Use OpenCV built-in haarcascades (Render safe)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

__class_name_to_number = {}
__class_number_to_name = {}
__model = None


def load_saved_artifacts():
    print("loading saved artifacts...start")

    global __class_name_to_number
    global __class_number_to_name
    global __model

    with open(CLASS_DICT_PATH, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    if __model is None:
        __model = joblib.load(MODEL_PATH)

    print("loading saved artifacts...done")


def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    if len(imgs) == 0:
        return []

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))

        combined_img = np.vstack((
            scalled_raw_img.reshape(32 * 32 * 3, 1),
            scalled_img_har.reshape(32 * 32, 1)
        ))

        final = combined_img.reshape(1, -1).astype(float)

        result.append({
            "class": __class_number_to_name[__model.p]()_
