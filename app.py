import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import zipfile
from datetime import datetime
from utils import predict_skin, annotate_and_save, CLASS_DICT

# -------- PAGE SETUP --------
st.set_page_config(page_title="DermaAI", layout="wide")
st.title("🧴 DermaAI – Skin Condition Detection")

# -------- UPLOAD IMAGE --------
uploaded_file = st.file_uploader(
    "Upload Image(s)",
    type=["jpg","jpeg","png"]
)

if uploaded_file:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict
    label, confidence, raw_preds = predict_skin(image)

    # Annotate
    annotated_img, saved_path = annotate_and_save(image, label, confidence)

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Annotated Image")
        st.image(annotated_img, use_container_width=True)
    with col2:
        st.subheader("🔍 Prediction")
        st.success(f"**{label.upper()}**")
        st.write(f"Confidence: **{confidence*100:.2f}%**")

    # Confidence Chart
    st.subheader("📊 Class Confidence")
    chart_df = pd.DataFrame({
        "Class": list(CLASS_DICT.values()),
        "Confidence": raw_preds
    })
    st.bar_chart(chart_df.set_index("Class"))

    # -------- HISTORY CSV --------
    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/history.csv"
    new_row = {
        "image_name": uploaded_file.name,
        "prediction": label,
        "confidence": round(confidence, 4),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "annotated_path": saved_path
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(csv_path, index=False)

    st.subheader("🧾 Prediction History")
    st.dataframe(df)

    # CSV download
    st.download_button(
        label="⬇ Download CSV",
        data=df.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    # -------- ZIP Annotated Images --------
    def zip_annotated_images():
        zip_path = "outputs/annotated_images.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in os.listdir("outputs/annotated"):
                zipf.write(
                    os.path.join("outputs/annotated", file),
                    arcname=file
                )
        return zip_path

    if st.button("📦 Download All Annotated Images"):
        zip_file = zip_annotated_images()
        with open(zip_file, "rb") as f:
            st.download_button(
                "⬇ Download ZIP",
                f,
                file_name="annotated_images.zip",
                mime="application/zip"
            )

else:
    st.info("Upload an image to start prediction")
