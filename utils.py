import cv2
import numpy as np
import tensorflow as tf
import json

# Load labels
with open("labels.json") as f:
    CLASS_NAMES = json.load(f)

# Load model once
@tf.keras.utils.register_keras_serializable()
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

IMG_SIZE = 224

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict(image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    class_id = np.argmax(preds)
    confidence = float(preds[class_id])
    return CLASS_NAMES[str(class_id)], confidence, preds
