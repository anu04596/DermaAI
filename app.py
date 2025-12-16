import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import predict

st.set_page_config(
    page_title="DermaAI",
    page_icon="🧴",
    layout="centered"
)

st.title("🧴 DermaAI – Skin Disease Detection")
st.write("Upload a skin image to predict the condition using Deep Learning.")

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        label, confidence, probs = predict(image)

    st.success(f"### 🧠 Prediction: **{label}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

    # 📊 Confidence Chart
    st.subheader("📊 Prediction Confidence")

    fig, ax = plt.subplots()
    ax.bar(range(len(probs)), probs)
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels(
        ["clear_skin", "darks_spots", "puffy_eyes", "wrinkles"],
        rotation=30
    )
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)

    st.pyplot(fig)

st.markdown("---")
st.caption("Built with TensorFlow • OpenCV • Streamlit")
