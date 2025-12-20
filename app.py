# app.py
import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd
import zipfile
import datetime

# ----------------------------
# PAGE CONFIG + STYLING
# ----------------------------
st.set_page_config(
    page_title="DermaAI",
    page_icon="🧬",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fdfbfb, #e3f2fd);
}
h1, h2, h3 {
    color: #2C3E50;
}
.stButton>button {
    background-color: #1ABC9C;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}
.pred-card {
    background: white;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.major {
    color: #E74C3C;
}
.minor {
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# MODEL & CLASS SETUP (UNCHANGED)
# ----------------------------
MODEL_PATH = "derma_model.h5"
model = load_model(MODEL_PATH)

class_dict = {
    0: "clear_skin",
    1: "dark_spots",
    2: "puffy_eyes",
    3: "wrinkles"
}

condition_info = {
    "clear_skin": "Healthy skin with no visible dermatological issues.",
    "dark_spots": "Hyperpigmentation caused by sun exposure or acne scars.",
    "puffy_eyes": "Swelling around eyes due to fatigue or fluid retention.",
    "wrinkles": "Fine lines caused by aging and reduced skin elasticity."
}

haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# HELPER FUNCTIONS (UNCHANGED LOGIC)
# ----------------------------
def predict_skin(face_img):
    img = cv2.resize(face_img, (224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    cls = np.argmax(preds)
    conf = preds[cls] * 100
    return class_dict[cls], float(conf), preds

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.05, 4, minSize=(24,24))
    return faces

# ----------------------------
# UI HEADER
# ----------------------------
st.title("🧬 DermaAI – Skin Analysis")
st.markdown("AI-powered facial skin condition detection.")
st.warning("⚠ Educational use only. Not a medical diagnosis.")

# ----------------------------
# INPUT MODE
# ----------------------------
mode = st.radio("Choose image source:", ["Upload Image(s)", "Use Webcam"])

images = []

if mode == "Upload Image(s)":
    uploaded_files = st.file_uploader(
        "Upload one or multiple images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for f in uploaded_files:
            img = cv2.imdecode(
                np.frombuffer(f.read(), np.uint8),
                cv2.IMREAD_COLOR
            )
            images.append((f.name, img))

elif mode == "Use Webcam":
    cam = st.camera_input("Capture image")
    if cam:
        img = cv2.imdecode(
            np.frombuffer(cam.getvalue(), np.uint8),
            cv2.IMREAD_COLOR
        )
        images.append(("webcam.jpg", img))

# ----------------------------
# PROCESS IMAGES
# ----------------------------
if images:
    annotated_folder = "annotated_faces"
    os.makedirs(annotated_folder, exist_ok=True)

    csv_records = []

    for name, img in images:
        orig = img.copy()
        faces = detect_faces(img)

        for idx, (x,y,w,h) in enumerate(faces):
            face_crop = orig[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            label, conf, raw_preds = predict_skin(face_crop)

            # -------- MAJOR & MINOR LOGIC --------
            sorted_preds = sorted(
                [(class_dict[i], raw_preds[i]*100) for i in range(len(raw_preds))],
                key=lambda x: x[1],
                reverse=True
            )

            major_label, major_conf = sorted_preds[0]
            minor_preds = sorted_preds[1:3]

            # -------- SEVERITY --------
            severity = (
                "Mild" if major_conf < 60 else
                "Moderate" if major_conf < 85 else
                "Severe"
            )

            # -------- ANNOTATION --------
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(
                img, f"{major_label} {major_conf:.1f}%",
                (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2
            )

            csv_records.append([name, major_label, round(major_conf,2), severity])

            # -------- UI CARD --------
            st.markdown(f"""
            <div class="pred-card">
                <h3 class="major">🩺 Major Prediction</h3>
                <h2 class="major">{major_label}</h2>
                <p><b>Confidence:</b> {major_conf:.2f}%</p>
                <p><b>Severity:</b> {severity}</p>
                <p>{condition_info[major_label]}</p>
                <hr>
                <h4>🔍 Minor Predictions</h4>
                {"".join([f"<p class='minor'>{lbl}: {cnf:.2f}%</p>" for lbl, cnf in minor_preds])}
            </div>
            """, unsafe_allow_html=True)

            # -------- CONFIDENCE BARS --------
            st.subheader("Confidence Breakdown")
            st.markdown(f"**{major_label} (Major)**")
            st.progress(major_conf/100)

            for lbl, cnf in minor_preds:
                st.markdown(lbl)
                st.progress(cnf/100)

        save_path = os.path.join(annotated_folder, name)
        cv2.imwrite(save_path, img)

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption=f"Annotated: {name}",
            use_container_width=True
        )

    # ----------------------------
    # SUMMARY + DOWNLOADS
    # ----------------------------
    df = pd.DataFrame(
        csv_records,
        columns=["image","predicted_label","confidence","severity"]
    )

    st.subheader("📊 Prediction Distribution")
    st.bar_chart(df["predicted_label"].value_counts())

    csv_path = "predictions.csv"
    df.to_csv(csv_path, index=False)

    zip_path = "annotated_images.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for f in os.listdir(annotated_folder):
            zipf.write(os.path.join(annotated_folder, f), arcname=f)

    report = f"""
DermaAI Diagnosis Report
-----------------------
Date: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}
Images Processed: {len(images)}

Disclaimer:
This report is AI-generated and not a medical diagnosis.
"""

    st.download_button("📄 Download CSV", open(csv_path,"rb"), "predictions.csv")
    st.download_button("🖼 Download Annotated Images", open(zip_path,"rb"), "annotated_images.zip")
    st.download_button("📝 Download Report", report, "DermaAI_Report.txt")
