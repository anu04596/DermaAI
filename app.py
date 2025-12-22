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
# DARK UI
# ----------------------------
st.markdown("""
<style>
.stApp { background:#0b0b0b; color:#eaeaea; }
h1,h2,h3,h4 { color:#ffffff; }
.stButton>button {
    background:linear-gradient(90deg,#16a085,#1abc9c);
    color:white;border-radius:10px;font-weight:600;
}
.pred-card {
    background:#111827;
    padding:18px;
    border-radius:16px;
    border-left:6px solid #22c55e;
    margin-bottom:18px;
}
.major { color:#22c55e;font-weight:700 }
.minor { color:#9ca3af }
hr { border:1px solid #374151 }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
MODEL_PATH = "derma_model.h5"
model = load_model(MODEL_PATH, compile=False)

# ----------------------------
# FACE DETECTORS (SAFE)
# ----------------------------
# Built-in Haar (NO XML download needed)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("❌ Haar cascade failed to load")
    st.stop()

# DNN files (must exist)
DNN_PROTO = "deploy.prototxt.txt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(DNN_PROTO) or not os.path.exists(DNN_MODEL):
    st.error("❌ Missing DNN files (deploy.prototxt / caffemodel)")
    st.stop()

net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)

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
# HAAR + DNN FACE DETECTION
# ----------------------------
def detect_faces(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Haar candidates
    candidates = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(100, 100)
    )

    if len(candidates) == 0:
        return []

    # DNN validation
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (300,300),
        (104.0,177.0,123.0),
        swapRB=False, crop=False
    )

    net.setInput(blob)
    detections = net.forward()

    verified = []

    for (x,y,bw,bh) in candidates:
        cx, cy = x + bw//2, y + bh//2

        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf < 0.35:
                continue

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)

            if x1 < cx < x2 and y1 < cy < y2:
                verified.append((x,y,bw,bh,conf))
                break

    # Fallback to Haar if DNN fails
    if not verified:
        verified = [(x,y,bw,bh,0.0) for (x,y,bw,bh) in candidates]

    return verified

# ----------------------------
# SELECT BEST FACE
# ----------------------------
def select_best_face(faces, shape):
    if not faces:
        return None

    h, w = shape[:2]
    center = np.array([w//2, h//2])

    best, best_score = None, -1
    for (x,y,fw,fh,conf) in faces:
        area = fw * fh
        face_center = np.array([x+fw//2, y+fh//2])
        dist = np.linalg.norm(face_center-center)
        score = area - dist*2 + conf*500

        if score > best_score:
            best_score = score
            best = (x,y,fw,fh)

    return best

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
st.title("🧬 DermaAI – Skin Analysis Platform")
st.markdown("AI-powered facial skin condition assessment.")
st.warning("⚠️ Educational use only. Not a medical diagnosis.")

# ----------------------------
# INPUT
# ----------------------------
mode = st.radio("Select Image Source", ["Upload Image(s)", "Use Webcam"], horizontal=True)
images = []

if mode == "Upload Image(s)":
    files = st.file_uploader(
        "Upload face images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )
    if files:
        for f in files:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            images.append((f.name, img))
else:
    cam = st.camera_input("Capture image")
    if cam:
        img = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        images.append(("webcam.jpg", img))

# ----------------------------
# PROCESS
# ----------------------------
if images:
    records = []
    annotated_dir = "annotated_faces"
    os.makedirs(annotated_dir, exist_ok=True)

    for name, img in images:
        faces = detect_faces(img)
        best = select_best_face(faces, img.shape)

        if not best:
            st.warning(f"⚠️ No face detected in {name}")
            continue

        x,y,w,h = best
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
            <h2 class="major">{label}</h2>
            <p><b>Confidence:</b> {conf:.2f}%</p>
            <p>{condition_info[label]}</p>
        </div>
        """, unsafe_allow_html=True)

        save_path = os.path.join(annotated_dir, name)
        cv2.imwrite(save_path, img)

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption=f"Annotated Output: {name}",
            use_container_width=True
        )

    # ----------------------------
    # LOGS & DOWNLOADS
    # ----------------------------
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
