# 🥭 Fresh vs Rotten Mango Classifier

An end-to-end Computer Vision web application built with **Streamlit** and **TensorFlow (MobileNetV2)** that classifies uploaded mango images as either **Fresh** or **Rotten**.

Developed by **Group EE7** for GET324.

---

## Features
* **Deep Learning Powered:** Built on a fine-tuned MobileNetV2 architecture trained on 29,000+ images (~93% validation accuracy).
* **Instant Inference:** Lightweight image preprocessing pipeline (`224x224` resolution).
* **Interactive UI:** Built using Streamlit for clean image upload and real-time confidence scoring.

---

## Project Structure
```text
├── app.py                  # Streamlit frontend & inference logic
├── mango_classifier.h5     # Trained Keras model weights
├── requirements.txt        # Production dependencies
└── README.md               # Project documentation
