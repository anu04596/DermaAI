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

st.markdown("""
<style>

/* ---------- Global Text ---------- */
html, body {
    color: #1f2937;
}

/* ---------- App Background ---------- */
.stApp {
    background: linear-gradient(135deg, #fde2e4, #e0f4f1, #e8eaf6);
    font-family: 'Segoe UI', sans-serif;
}

/* ---------- Top Header ---------- */
header[data-testid="stHeader"] {
    background: linear-gradient(135deg, #fbc2eb, #a6c1ee);
}
header[data-testid="stHeader"] * {
    color: #1f2937 !important;
}

/* ---------- Headings ---------- */
h1, h2, h3, h4 {
    color: #1f2937 !important;
    text-align: center;
}

/* ---------- Body Text ---------- */
p, label, span {
    color: #6b7280 !important;
}
/* ---------- Buttons ---------- */
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

/* ---------- Radio ---------- */
div[role="radiogroup"] {
    justify-content: center;
}

/* ---------- Prediction Card ---------- */
.pred-card {
    background: rgba(255,255,255,0.92);
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.1);
    margin: 25px auto;
    max-width: 650px;
    text-align: center;
}

/* ---------- Highlight ---------- */
.major {
    color: #1f2937 !important;
    font-weight: 700;
}

/* ---------- Progress ---------- */
.stProgress > div > div {
    background-color: #a6c1ee;
}

/* ---------- FILE UPLOADER (CLEAN) ---------- */
div[data-testid="stFileUploader"] {
    background: #f8fafc;
    border-radius: 16px;
    padding: 18px;
    border: 1.5px dashed #c7d2fe;
    max-width: 650px;
    margin: 25px auto;
}

div[data-testid="stFileUploader"] * {
    color: #6b7280 !important;
}

div[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #fbc2eb, #a6c1ee) !important;
    color: #1f2937 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: none !important;
}

/* ---------- DataFrame ---------- */
div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.95);
    border-radius: 16px;
    padding: 10px;
}

div[data-testid="stDataFrame"] thead th {
    background-color: #e8eaf6 !important;
    color: #1f2937 !important;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------
# LOAD MODEL
# ----------------------------
MODEL_PATH = "derma_model.h5"
model = load_model(MODEL_PATH, compile=False)

# ----------------------------
# FACE DETECTOR (HAAR)
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

# ----------------------------
# FACE DETECTION
# ----------------------------
def detect_faces(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_dim = max(30, int(min(h, w) * 0.1))
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(min_dim, min_dim)
    )
    return faces

# ----------------------------
# SKIN PREDICTION
# ----------------------------
def predict_skin(face):
    face = cv2.resize(face, (224,224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = preprocess_input(face)
    face = np.expand_dims(face, 0)
    preds = model.predict(face, verbose=0)[0]
    idx = np.argmax(preds)
    return class_dict[idx], preds[idx]*100, preds

# ----------------------------
# HEADER
# ----------------------------
st.markdown(
    "<h1>DermaAI - Facial Skin Analysis Detection Platform</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; font-size:16px;'>AI-powered facial skin condition assessment</p>",
    unsafe_allow_html=True
)

st.info("⚠️ Educational use only. Not a medical diagnosis.")
st.markdown("""
<div style="display:flex; justify-content:center; gap:24px; flex-wrap:wrap; margin:40px 0;">

  <div style="
      background: rgba(255,255,255,0.9);
      padding: 22px;
      border-radius: 18px;
      width: 220px;
      text-align: center;
      box-shadow: 0px 8px 22px rgba(0,0,0,0.08);
  ">
      <h3>🔍 AI Skin Analysis</h3>
      <p>Deep learning–based facial skin condition detection</p>
  </div>

  <div style="
      background: rgba(255,255,255,0.9);
      padding: 22px;
      border-radius: 18px;
      width: 220px;
      text-align: center;
      box-shadow: 0px 8px 22px rgba(0,0,0,0.08);
  ">
      <h3>📊 Confidence Scores</h3>
      <p>Clear probability breakdown for each prediction</p>
  </div>

  <div style="
      background: rgba(255,255,255,0.9);
      padding: 22px;
      border-radius: 18px;
      width: 220px;
      text-align: center;
      box-shadow: 0px 8px 22px rgba(0,0,0,0.08);
  ">
      <h3>🔐 Privacy First</h3>
      <p>No permanent image storage, secure processing</p>
  </div>

</div>
""", unsafe_allow_html=True)

# ----------------------------
# INPUT MODE
# ----------------------------
mode = st.radio("Select Image Source", ["Upload Image(s)", "Use Webcam"], horizontal=True)
images = []

if mode == "Upload Image(s)":
    # Custom upload card
    st.markdown("""
    <div class="upload-card">
        <h3>📷 Upload Facial Images</h3>
        <p>Drag & drop or browse JPG, JPEG, PNG files</p>
    </div>
    """, unsafe_allow_html=True)

    files = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if files:
        for f in files:
            img = cv2.imdecode(
                np.frombuffer(f.read(), np.uint8),
                cv2.IMREAD_COLOR
            )
            images.append((f.name, img))
else:
    cam = st.camera_input("Capture image")
    if cam:
        img = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        images.append(("webcam.jpg", img))

# ----------------------------
# PROCESS IMAGES
# ----------------------------
if images:
    records = []
    annotated_dir = "annotated_faces"
    os.makedirs(annotated_dir, exist_ok=True)

    for name, img in images:
        faces = detect_faces(img)

        if len(faces) == 0:
            faces = [(0, 0, img.shape[1], img.shape[0])]

        for (x, y, w, h) in faces:
            face_crop = img[y:y+h, x:x+w]
            label, conf, raw = predict_skin(face_crop)

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(
                img,f"{label} {conf:.1f}%",
                (x,y-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(0,255,0),2
            )

            records.append([
                name, label, round(conf,2),
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

            st.markdown(f"""
            <div class="pred-card">
                <h3 class="major">🩺 Primary Detection</h3>
                <h2 class="major">{label.replace('_',' ').title()}</h2>
                <p><b>Confidence:</b> {conf:.2f}%</p>
                <p>{condition_info[label]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Confidence Breakdown")

            conf_dict = {class_dict[i]: raw[i]*100 for i in range(len(raw))}
            conf_dict = dict(sorted(conf_dict.items(), key=lambda x: x[1], reverse=True))

            for lbl, score in conf_dict.items():
                highlight = "🟢 " if lbl == label else ""
                st.markdown(f"**{highlight}{lbl.replace('_',' ').title()}** — {score:.2f}%")
                st.progress(int(score))

        save_path = os.path.join(annotated_dir, name)
        cv2.imwrite(save_path, img)

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption=f"Annotated Output: {name}",
            use_container_width=True
        )

    df = pd.DataFrame(
        records,
        columns=["image","prediction","confidence","timestamp"]
    )

    st.subheader("📊 Prediction Logs")
    st.dataframe(df, use_container_width=True)

    csv_path = "prediction_logs.csv"
    df.to_csv(csv_path, index=False)

    zip_path = "annotated_images.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for f in os.listdir(annotated_dir):
            zipf.write(os.path.join(annotated_dir, f), arcname=f)

    st.download_button("📄 Download Logs CSV", open(csv_path,"rb"), "prediction_logs.csv")
    st.download_button("🖼 Download Annotated Images", open(zip_path,"rb"), "annotated_images.zip")
