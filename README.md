# DermaScan 🧬

**DermaScan** is an AI-powered web application for **skin condition classification**. It uses deep learning to analyze skin images and predict the type of skin issue in real-time with high accuracy.  

---

## Dataset 🗂️

The model is trained on a curated dataset with **4 skin classes**:

| Class Name     | Initial Images | After Augmentation |
|----------------|----------------|------------------|
| Clear Skin     | 100            | 500              |
| Puffy Eyes     | 100            | 500              |
| Wrinkles       | 100            | 500              |
| Dark Spots     | 100            | 500              |

- **Total images after augmentation:** 2000  
- Images resized to **224x224** and normalized using **EfficientNet preprocessing**.  
- Data augmentation techniques included **rotation, flipping, brightness adjustment, zooming**.  

### Dataset Flow Diagram

    A[Original Images (100 per class)] --> B[Data Augmentation] --> C[Augmented Dataset (500 per class)]
## Model Architecture

- The core model is based on **EfficientNet** with transfer learning.  
- **Input shape:** `(224, 224, 3)`  
- **Output shape:** `(4)` (4 classes)  
- **Total parameters:** 4,378,535  
  - Trainable params: 328,964  
  - Non-trainable params: 4,049,571  
- Model uses `softmax` activation for multi-class classification.  
- Optimized for **accuracy and real-time inference** on web applications.

   ` I[Input Image 224x224x3] --> E[EfficientNetB0 Base Model] --> G[Global Average Pooling] --> F[Fully Connected Layer]--> O[Softmax Output (4 classes)]`


---
## Features

- **Real-time skin disease prediction** from uploaded images.  
- **High-performance inference** with EfficientNet-based model.  
- **User-friendly web interface** accessible online.  
- **Categorical output** with class probabilities for better interpretability.  

---

## Live Application

You can access the live web application here:  
[DermaAI Live App](https://dermaai-fhyknap4uo3l6bluyb2vki.streamlit.app/)

---

## Notes

- The model is **not intended to replace professional medical advice**. Always consult a dermatologist for medical diagnosis.  
- Accuracy depends on the quality and clarity of the uploaded images.  
