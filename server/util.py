# ===== sklearn backward compatibility patches =====
import sys
import sklearn.preprocessing
import sklearn.svm

# old sklearn paths used when the model was trained
sys.modules['sklearn.preprocessing.data'] = sklearn.preprocessing
sys.modules['sklearn.svm.classes'] = sklearn.svm
# ================================================

import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import os

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

CLASS_DICT_PATH = os.path.join(ARTIFACTS_DIR, "class_dictionary.json")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "saved_model.pkl")

# ---------- OPENCV CASCADES (Render safe) ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

__class_name_to_number = {}
__class_number_to_name = {}
__model = None


# ---------- LOAD MODEL ----------
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

        # 🔧 PATCH: fix old SVC model incompatibility
        try:
            svc = __model.steps[-1][1]  # last step in Pipeline
            if not hasattr(svc, "break_ties"):
                svc.break_ties = False
        except Exception:
            pass

    print("loading saved artifacts...done")


# ---------- CLASSIFICATION ----------
def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image(file_path, image_base64_data)

    if len(imgs) == 0:
        return []

    result = []
    for img in imgs:
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, "db1", 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))

        combined_img = np.vstack((
            scaled_raw_img.reshape(32 * 32 * 3, 1),
            scaled_img_har.reshape(32 * 32, 1)
        ))

        final = combined_img.reshape(1, -1).astype(float)

        prediction = __model.predict(final)[0]
        probabilities = __model.predict_proba(final)[0]

        result.append({
            "class": __class_number_to_name[prediction],
            "class_probability": np.around(probabilities * 100, 2).tolist(),
            "class_dictionary": __class_name_to_number
        })

    return result


# ---------- IMAGE HELPERS ----------
def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def get_cropped_image(image_path, image_base64_data):
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_color = img[y:y + h, x:x + w]
        cropped_faces.append(roi_color)

    return cropped_faces
