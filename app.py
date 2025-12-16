# app.py
import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import zipfile

# ----------------------------
# Model & class setup
# ----------------------------
MODEL_PATH = "derma_model.h5"  # Your uploaded model
model = load_model(MODEL_PATH)

class_dict = {
    0: "clear_skin",
    1: "dark_spots",
    2: "puffy_eyes",
    3: "wrinkles"
}

# Haarcascade for face detection
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------
# Helper functions
# ----------------------------
def predict_skin(face_img):
    img = cv2.resize(face_img, (224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    cls = np.argmax(preds)
    conf = preds[cls] * 100
    return class_dict[cls], float(conf)

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.05, 4, minSize=(24,24))
    return faces

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("DermaAI - Skin Prediction")
st.markdown("Upload images and get predicted labels with annotated outputs.")

uploaded_files = st.file_uploader(
    "Upload one or multiple images", type=["jpg","jpeg","png"], accept_multiple_files=True
)

if uploaded_files:
    annotated_folder = "annotated_faces"
    cropped_folder = "cropped_faces"
    os.makedirs(annotated_folder, exist_ok=True)
    os.makedirs(cropped_folder, exist_ok=True)
    
    csv_records = []

    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        orig = img.copy()

        faces = detect_faces(img)

        for idx, (x,y,w,h) in enumerate(faces):
            x, y = max(0,x), max(0,y)
            face_crop = orig[y:y+h, x:x+w]

            if face_crop.size == 0:
                continue

            label, conf = predict_skin(face_crop)

            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, f"{label} {conf:.1f}%", (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            crop_path = os.path.join(cropped_folder, f"{file.name}_face{idx}.jpg")
            cv2.imwrite(crop_path, face_crop)

            csv_records.append([file.name, label, round(conf,2), "haar"])

        save_path = os.path.join(annotated_folder, file.name)
        cv2.imwrite(save_path, img)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Annotated: {file.name}")

    # Save CSV
    df = pd.DataFrame(csv_records, columns=["image","predicted_label","confidence","detected_by"])
    csv_path = "predictions.csv"
    df.to_csv(csv_path, index=False)
    st.success("CSV generated!")

    # Show chart
    st.subheader("Prediction Distribution")
    chart_data = df['predicted_label'].value_counts()
    st.bar_chart(chart_data)

    # Zip annotated images
    zip_path = "annotated_images.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(annotated_folder):
            for f in files:
                zipf.write(os.path.join(root,f), arcname=f)

    st.download_button("Download CSV", data=open(csv_path, "rb"), file_name="predictions.csv")
    st.download_button("Download Annotated Images", data=open(zip_path, "rb"), file_name="annotated_images.zip")
