# 🧠 Face Emotion Detector

A real-time face emotion detection web app powered by a pre-trained FER-2013 deep learning model.

![Tech Stack](https://img.shields.io/badge/stack-Flask%20%7C%20OpenCV%20%7C%20TensorFlow-7c5cfc?style=flat-square)
![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square)

---

## ✨ Features

- 📷 **Live Webcam Stream** — Real-time emotion detection from your webcam
- 🖼️ **Image Upload** — Upload any photo to detect emotions (works on cloud too)
- 😄 😠 😢 **7 Emotion Classes** — Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- 📊 **Confidence Scores** — Each prediction shows % confidence
- 📋 **Emotion Log** — Real-time log of detected emotions with timestamps
- 📸 **Screenshot Button** — Capture and download any frame
- 🌙 **Dark Glassmorphism UI** — Premium modern design with animated background

---

## 🛠️ Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| Backend   | Python, Flask                       |
| AI/ML     | TensorFlow / Keras, FER-2013 model  |
| CV        | OpenCV (Haar Cascade face detector) |
| Frontend  | HTML5, Vanilla CSS, JavaScript      |

---

## 📁 Project Structure

```
emotion-detector/
├── app.py                               # Flask server
├── download_weights.py                  # Setup: downloads model & cascade
├── model/
│   └── emotion_model.h5                 # Pre-trained FER-2013 Keras model
├── haarcascade_frontalface_default.xml  # OpenCV face detector
├── static/
│   └── style.css                        # Dark glassmorphism UI
├── templates/
│   └── index.html                       # Main web page
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

```bash
# 1. Clone / navigate to the project
cd emotion-detector

# 2. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate       # Linux / Mac
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the model and Haar Cascade
python download_weights.py

# 5. Run the app
python app.py
```

Then open **http://localhost:5000** in your browser.

> **Camera permission**: Your browser will ask for webcam access — click Allow.

---

## 🧠 Model Details

- **Dataset**: FER-2013 (~35,000 grayscale face images)
- **Input**: 48×48 grayscale face crops
- **Architecture**: CNN
- **Output**: 7 classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

---

## ☁️ Deployment Notes (Hybrid Approach)

| Mode           | Method           | Notes                                |
|----------------|------------------|--------------------------------------|
| **Local**      | `python app.py`  | Full webcam support                  |
| **Cloud Demo** | Render / Railway | Use the **Upload Image** tab instead |

> Webcams don't work on cloud servers (no hardware access). The Image Upload tab is designed to work seamlessly in cloud deployments.

---

## 🚀 Deployment Guide (Step-by-Step)

### Option 1: Render (Recommended - 100% Reliable)
Render is the best choice because it supports **Docker**, which handles the complex OpenCV and TensorFlow dependencies for you.

1.  **GitHub Setup**:
    -   Create a new repository on GitHub.
    -   Upload all your project files (including the `Dockerfile`).
2.  **Render Setup**:
    -   Log in to [Render.com](https://render.com).
    -   Click **New +** and select **Web Service**.
    -   Connect your GitHub repository.
3.  **Settings**:
    -   **Name**: `emotion-detector` (or anything you like).
    -   **Runtime**: Select **Docker**. (Render will automatically use the `Dockerfile` I created).
    -   **Instance Type**: `Free` or `Starter`.
4.  **Done!**:
    -   Click **Deploy Web Service**.
    -   Wait 2–3 minutes for the build to finish. Your app will be live at `https://your-app-name.onrender.com`.

### ⚠️ Note on Vercel
**Vercel is NOT recommended** for this specific project. 
Vercel is designed for "Serverless Functions" which have strict limits:
-   **No System Libraries**: You cannot install `libgl1` (required by OpenCV).
-   **Size Limits**: TensorFlow + OpenCV dependencies exceed Vercel's free tier deployment size. 
-   **Performance**: AI models can be slow to "warm up" on serverless, causing 504 errors.

---

## 📦 requirements.txt

```
flask
opencv-python-headless
tensorflow
numpy
werkzeug
requests
```
