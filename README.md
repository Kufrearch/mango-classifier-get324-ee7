# 🥭 Fresh vs Rotten Mango Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mango-classifier-get324-ee7-7hebjh7dsst3dftfqjgn3r.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)

An end-to-end deep learning web application built for automated mango quality assessment. Developed as part of the **GET324 Mini-Project (Group EE7)** course deliverables.

---

## Live Web Application

Access the operational Streamlit Cloud application directly here:  
**[Mango Classifier Web App Link](https://mango-classifier-get324-ee7-7hebjh7dsst3dftfqjgn3r.streamlit.app/)**

---

## Project Overview

This repository houses a fine-tuned binary vision classifier utilizing **MobileNetV2** transfer learning. The system evaluates input mango images and classifies them into two quality states:
* **Fresh:** Healthy, ripe, or undamaged mangoes.
* **Rotten:** Degraded, damaged, or decaying mangoes.

The web app presents real-time predictions alongside visual diagnostic reports and confidence metrics.

## Group Members

1. Archibong, Kufre Ini — 22/EG/EE/2103

2. Usoro, Ubongabasi Isonguyo — 22/EG/EE/1963

3. Udom, Victor Israel — 22/EG/EE/2043

4. Ekwebelem, Chibueze Princewill — 22/EG/EE/2053

5. David, Israel Akpanumoh — 22/EG/EE/2073

6. Sampson, Abasifreke Sampson — 22/EG/EE/1983

7. Ogunbiyi, Praise Ayodele — 22/EG/EE/2033

8. Okon, Samuel Ita — 22/EG/EE/1973

9. Etim, David Emmanuel — 22/EG/EE/1993

10. Inyang, Obongama Ekpong — 22/EG/EE/2083

11. Wisdom Ndueso Akpan — 22/EG/EE/2023

12. Nsikan, Saviour Ebenezer — 22/EG/EE/2003

---

## Repository Structure

```text
├── .devcontainer/       # Container configurations for isolated development environments
├── CONTRIBUTORS.md     # Group members, and technical contributions
├── README.md           # Master project documentation and application guide
├── app.py              # Streamlit application entry point and user interface code
├── mango_classifier.h5 # Trained MobileNetV2 model binary (HDF5 format)
└── requirements.txt    # Python runtime environment dependencies
