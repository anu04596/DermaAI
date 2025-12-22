# app.py
import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
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
# DARK PROFESSIONAL UI
# ----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0b0b0b;
    color: #eaeaea;
}
h1, h2, h3, h4 {
    color: #ffffff;
}
.stButton>button {
    background: linear-gradient(90deg, #16a085, #1abc9c);
    color: white;
    border-radius: 10px;
    font-weight: 600;
}
.pred-card {
    background: linear-gradient(145deg, #111827, #1f2933);
    padding: 18px;
    border-radius: 16px;
    border-left: 6px solid #22c55e;
    box-shadow: 0 10px 25px rgba(0,0,0,0.6);
    margin-bottom: 18px;
}
.major {
    color: #22c55e;
    font-weight: 700;
}
.minor {
    color: #9ca3af;
}
hr {
    border: 1px solid #374151;
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

# ----------------------------
# MEDIAPIPE FACE DETECTOR
# ----------------------------
mp_face_detection = mp.solutions.face_detection

def detect_faces(image, min_conf=0.7):
    h, w, _ = image.shape
    boxes = []

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_conf
    ) as detector:

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if not results.detections:
            return boxes

        for det in results.detections:
            box = det.location_data.relative_bounding_box

            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            x, y = max(0, x), max(0, y)

            # Reject tiny / false detections (arms, walls)
            if bw < 80 or bh < 80:
                continue

            boxes.append((x, y, bw, bh))

    return boxes

# ----------------------------
# HELPER FUNCTIONS (UNCHANGED)
# ----------------------------
def predict_skin(face_img):
    img = cv2.resize(face_img, (224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)[0]
    cls = np.argmax(preds)
    conf = preds[cls] * 100
    return class_dict[cls], float(conf), preds

# ----------------------------
# HEADER
# ----------------------------
st.title("🧬 DermaAI – Skin Analysis Platform")
st.markdown("AI-powered facial skin condition assessment with confidence insights.")
st.warning("⚠️ For educational and research purposes only. Not a medical diagnosis.")

# ----------------------------
# INPUT MODE
# ----------------------------
mode = st.radio(
    "Select Image Source",
    ["Upload Image(s)", "Use Webcam"],
    horizontal=True
)

images = []

if mode == "Upload Image(s)":
    uploaded_files = st.file_uploader(
        "Upload one or multiple facial images",
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

else:
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

        if not faces:
            st.warning(f"⚠️ No valid face detected in {name}")
            continue

        for (x,y,w,h) in faces:
            face_crop = orig[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            label, conf, raw_preds = predict_skin(face_crop)

            sorted_preds = sorted(
                [(class_dict[i], raw_preds[i]*100) for i in range(len(raw_preds))],
                key=lambda x: x[1],
                reverse=True
            )

            major_label, major_conf = sorted_preds[0]
            minor_preds = sorted_preds[1:3]

            severity = (
                "Mild" if major_conf < 60 else
                "Moderate" if major_conf < 85 else
                "Severe"
            )

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(
                img,
                f"{major_label} {major_conf:.1f}%",
                (x,y-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            csv_records.append([
                name,
                major_label,
                round(major_conf,2),
                severity,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

            st.markdown(f"""
            <div class="pred-card">
                <h3 class="major">🩺 Primary Detection</h3>
                <h2 class="major">{major_label}</h2>
                <p><b>Confidence:</b> {major_conf:.2f}%</p>
                <p><b>Severity:</b> {severity}</p>
                <p>{condition_info[major_label]}</p>
                <hr>
                <h4>Secondary Indicators</h4>
                {"".join([f"<p class='minor'>{lbl}: {cnf:.2f}%</p>" for lbl, cnf in minor_preds])}
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Confidence Breakdown")
            st.progress(float(major_conf) / 100)
            for lbl, cnf in minor_preds:
                st.markdown(lbl)
                st.progress(float(cnf) / 100)

        save_path = os.path.join(annotated_folder, name)
        cv2.imwrite(save_path, img)

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption=f"Annotated Output: {name}",
            use_container_width=True
        )

    # ----------------------------
    # LOGS + DOWNLOADS
    # ----------------------------
    df = pd.DataFrame(
        csv_records,
        columns=["image","predicted_label","confidence","severity","timestamp"]
    )

    st.subheader("📊 Prediction Logs")
    st.dataframe(df, use_container_width=True)

    csv_path = "prediction_logs.csv"
    df.to_csv(csv_path, index=False)

    zip_path = "annotated_images.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for f in os.listdir(annotated_folder):
            zipf.write(os.path.join(annotated_folder, f), arcname=f)

    st.download_button("📄 Download Prediction Logs", open(csv_path,"rb"), "prediction_logs.csv")
    st.download_button("🖼 Download Annotated Images", open(zip_path,"rb"), "annotated_images.zip")
