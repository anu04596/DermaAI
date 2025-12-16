# DermaScan 🩺  
**AI Facial Skin Aging Detection App**

🔗 **Live App:** https://dermaai-fhyknap4uo3l6bluyb2vki.streamlit.app/

---

## 📌 Overview
DermaScan is a deep learning–based web application that detects and classifies skin conditions from uploaded images.  
The system uses **computer vision** and **transfer learning** to deliver fast and reliable predictions through an interactive Streamlit interface.

---

## 🚀 Features
- Image upload and real-time prediction
- Automatic face/skin region detection
- Deep learning–based classification
- Confidence score for each prediction
- Deployed on Streamlit Cloud

---

## 🧠 Model Details
- **Architecture:** EfficientNetB0 (Transfer Learning)
- **Framework:** TensorFlow / Keras
- **Input Size:** 224 × 224 × 3
- **Output:** 4 skin condition classes (Softmax)
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy

---

## 🛠️ Technology Stack
- **Frontend:** Streamlit  
- **Backend:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Computer Vision:** OpenCV (Haar Cascade)  
- **Deployment:** Streamlit Cloud  

---

## 🔄 Prediction Workflow
1. User uploads a skin image  
2. Face/skin region detected using Haar Cascade  
3. Image resized and normalized  
4. CNN model performs classification  
5. Predicted condition and confidence displayed  

---
