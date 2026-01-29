# ================== CRITICAL PATCH (LEGACY SKLEARN PICKLE) ==================
import sys
import sklearn.preprocessing
import sklearn.svm._classes

sys.modules['sklearn.preprocessing.data'] = sklearn.preprocessing
sys.modules['sklearn.svm.classes'] = sklearn.svm._classes
# ===========================================================================

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

# Use OpenCV built-in haarcascades (DEPLOY-SAFE)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

__class_name_to_number = {}
__class_number_to_name = {}
__model = None


def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
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
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


def load_saved_artifacts():
    global __class_name_to_number, __class_number_to_name, __model

    print("loading saved artifacts...start")

    with open(CLASS_DICT_PATH, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    if __model is None:
        __model = joblib.load(MODEL_PATH)

    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces
