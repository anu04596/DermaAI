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
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="DermaAI",
    page_icon="🧬",
    layout="centered"
)

# ----------------------------
# PASTEL + SOFT THEME CSS
# ----------------------------
st.markdown("""
<style>
html, body { color: #1f2937; font-family: 'Segoe UI', sans-serif; }
.stApp { background: linear-gradient(135deg, #fde2e4, #e0f4f1, #e8eaf6); }
header[data-testid="stHeader"] { background: linear-gradient(135deg, #fbc2eb, #a6c1ee); }
header[data-testid="stHeader"] * { color: #1f2937 !important; }
h1,h2,h3,h4 { color: #1f2937 !important; text-align: center; }
p, label, span { color: #6b7280 !important; }
.stButton > button,
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #fbc2eb, #a6c1ee) !important;
    color: #1f2937 !important;
    border-radius: 14px;
    font-weight: 600;
    padding: 10px 22px;
    border: none;
    box-shadow: 0px 6px 14px rgba(0,0,0,0.12);
}
div[role="radiogroup"] { justify-content: center; }
.pred-card {
    background: rgba(255,255,255,0.92);
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.1);
    margin: 25px auto;
    max-width: 650px;
    text-align: center;
}
.major { color: #1f2937 !important; font-weight: 700; }
.stProgress > div > div { background-color: #a6c1ee; }
div[data-testid="stFileUploader"] {
    background: #f8fafc;
    border-radius: 16px;
    padding: 18px;
    border: 1.5px dashed #c7d2fe;
    max-width: 650px;
    margin: 25px auto;
}
div[data-testid="stFileUploader"] * { color: #6b7280 !important; }
div[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #fbc2eb, #a6c1ee) !important;
    color: #1f2937 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: none !important;
}
.workflow-card {
    background: rgba(255,255,255,0.9);
    padding: 22px;
    border-radius: 18px;
    width: 220px;
    aspect-ratio: 1/1;
    text-align: center;
    box-shadow: 0px 8px 22px rgba(0,0,0,0.08);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
MODEL_PATH = "derma_model.h5"
model = load_model(MODEL_PATH, compile=False)

# ----------------------------
# FACE DETECTOR
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    st.error("❌ Haar cascade failed to load")
    st.stop()

# ----------------------------
# CLASSES
# ----------------------------
class_dict = {0: "clear_skin", 1: "dark_spots", 2: "puffy_eyes", 3: "wrinkles"}
condition_info = {
    "clear_skin": "Healthy skin with no visible dermatological issues.",
    "dark_spots": "Hyperpigmentation caused by sun exposure or acne scars.",
    "puffy_eyes": "Swelling around eyes due to fatigue or fluid retention.",
    "wrinkles": "Fine lines caused by aging and reduced skin elasticity."
}

# ----------------------------
# IMPROVED FACE DETECTION
# ----------------------------
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    raw_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40)
    )

    img_area = image.shape[0] * image.shape[1]
    valid_faces = []

    for (x, y, w, h) in raw_faces:
        face_area = w * h
        aspect_ratio = w / h

        if (
            face_area > 0.03 * img_area and    # must be ≥3% of image
            0.75 <= aspect_ratio <= 1.33       # realistic face shape
        ):
            valid_faces.append((x, y, w, h))

    return valid_faces

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_skin(face):
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = preprocess_input(face)
    face = np.expand_dims(face, 0)
    preds = model.predict(face, verbose=0)[0]
    idx = np.argmax(preds)
    return class_dict[idx], preds[idx] * 100

# ----------------------------
# HEADER
# ----------------------------
st.markdown("<h1>DermaAI – Facial Skin Analysis Detection Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered facial skin condition assessment</p>", unsafe_allow_html=True)

# ----------------------------
# WORKFLOW CARDS
# ----------------------------
st.markdown("""
<div style="display:flex; justify-content:space-around; gap:24px; flex-wrap:nowrap; margin:40px 0;">
  <div class="workflow-card">
      <h2>1️⃣</h2>
      <h4>Upload Image</h4>
      <p>Upload your facial image(s)</p>
  </div>
  <div class="workflow-card">
      <h2>2️⃣</h2>
      <h4>AI Analysis</h4>
      <p>Deep learning skin analysis</p>
  </div>
  <div class="workflow-card">
      <h2>3️⃣</h2>
      <h4>View Results</h4>
      <p>Confidence-based results</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# INPUT MODE
# ----------------------------
mode = st.radio("Select Image Source", ["Upload Image(s)", "Use Webcam"], horizontal=True)
images = []

if mode == "Upload Image(s)":
    files = st.file_uploader("Upload facial images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if files:
        for f in files:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                images.append((f.name, img))
else:
    cam = st.camera_input("Capture image")
    if cam:
        img = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(("webcam.jpg", img))

# ----------------------------
# PROCESS IMAGES
# ----------------------------
if images:
    records = []
    os.makedirs("annotated_faces", exist_ok=True)

    for name, img in images:
        faces = detect_faces(img)

        if len(faces) == 0:
            st.warning(f"❌ No human face detected in `{name}`. Prediction skipped.")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            continue

        for (x, y, w, h) in faces:
            face_crop = img[y:y+h, x:x+w]
            label, conf = predict_skin(face_crop)

            records.append([
                name, label, round(conf, 2),
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

            st.markdown(f"""
            <div class="pred-card">
                <h3 class="major">🩺 Detection Result</h3>
                <h2 class="major">{label.replace('_',' ').title()}</h2>
                <p><b>Confidence:</b> {conf:.2f}%</p>
                <p>{condition_info[label]}</p>
            </div>
            """, unsafe_allow_html=True)

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, f"{label} {conf:.1f}%", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        save_path = os.path.join("annotated_faces", name)
        cv2.imwrite(save_path, img)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption=f"Annotated Output: {name}",
                 use_container_width=True)

    if records:
        df = pd.DataFrame(records, columns=["image","prediction","confidence","timestamp"])
        st.subheader("📊 Prediction Logs")
        st.dataframe(df, use_container_width=True)

        df.to_csv("prediction_logs.csv", index=False)

        with zipfile.ZipFile("annotated_images.zip", "w") as zipf:
            for f in os.listdir("annotated_faces"):
                zipf.write(os.path.join("annotated_faces", f), arcname=f)

        st.download_button("📄 Download Logs CSV", open("prediction_logs.csv","rb"))
        st.download_button("🖼 Download Annotated Images", open("annotated_images.zip","rb"))
