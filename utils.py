import cv2
import os
import uuid
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "derma_model.h5"
CLASS_DICT = {
    0: "clear_skin",
    1: "dark_spots",
    2: "puffy_eyes",
    3: "wrinkles"
}

# Load model once
model = load_model(MODEL_PATH)

def predict_skin(face_img):
    img = cv2.resize(face_img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    cls = int(np.argmax(preds))
    conf = float(preds[cls])
    return CLASS_DICT[cls], conf, preds

def annotate_and_save(image, label, confidence):
    os.makedirs("outputs/annotated", exist_ok=True)
    annotated = image.copy()
    h, w, _ = annotated.shape

    # Rectangle around image (or later face)
    cv2.rectangle(annotated, (10,10), (w-10,h-10), (0,255,0), 2)
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(annotated, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Save annotated image
    filename = f"{uuid.uuid4()}.jpg"
    save_path = os.path.join("outputs/annotated", filename)
    cv2.imwrite(save_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    return annotated, save_path
