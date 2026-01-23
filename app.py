import os
import cv2
import numpy as np
import base64
import datetime
import json
from flask import Flask, render_template, Response, request, jsonify

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import tf_keras as keras

app = Flask(__name__)

# ─── Constants ───────────────────────────────────────────────
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_EMOJIS = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😄',
    'Sad':      '😢',
    'Surprise': '😲',
    'Neutral':  '😐',
}
EMOTION_COLORS = {
    'Angry':    (0, 0, 220),
    'Disgust':  (0, 140, 0),
    'Fear':     (128, 0, 128),
    'Happy':    (0, 200, 100),
    'Sad':      (200, 100, 0),
    'Surprise': (0, 165, 255),
    'Neutral':  (150, 150, 150),
}

emotion_log = []

# ─── Model & Cascade Loading ─────────────────────────────────
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
MODEL_PATH   = os.path.join('model', 'emotion_model.h5')

face_cascade = None
model = None

def load_resources():
    global face_cascade, model
    if os.path.exists(CASCADE_PATH):
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        print("✅ Loaded Haar Cascade.")
    else:
        print(f"⚠️  Cascade not found at '{CASCADE_PATH}'. Run download_weights.py first.")

    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Loaded emotion model.")
    else:
        print(f"⚠️  Model not found at '{MODEL_PATH}'. Run download_weights.py first.")

load_resources()

# ─── Detection Logic ─────────────────────────────────────────
def process_single_frame(frame, annotate=True):
    """Processes a single frame and returns detections. Optionally annotates."""
    if face_cascade is None or model is None:
        return frame, []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    detections = []
    for (x, y, w, h) in faces:
        # Preprocessing for AI: Crop -> 48x48 -> Grayscale -> Normalize
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_normalized = roi_resized.astype('float32') / 255.0
        roi_input = np.expand_dims(np.expand_dims(roi_normalized, -1), 0)

        preds = model.predict(roi_input, verbose=0)[0]
        emotion_idx = int(np.argmax(preds))
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = float(preds[emotion_idx]) * 100
        emoji = EMOTION_EMOJIS.get(emotion, '')
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        if annotate:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            # Label background pill
            label = f"{emoji} {emotion}  {confidence:.1f}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - lh - 12), (x + lw + 8, y), color, -1)
            cv2.putText(frame, label, (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detections.append({
            'emotion': emotion, 
            'confidence': round(confidence, 1), 
            'emoji': emoji,
            'box': [int(x), int(y), int(w), int(h)],
            'color': f"rgb({color[2]},{color[1]},{color[0]})" # BGR to RGB for CSS
        })

    return frame, detections


# ─── Routes ──────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
            
        header, encoded = data['image'].split(",", 1)
        decoded = base64.b64decode(encoded)
        npimg = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Decode failed'}), 400

        _, detections = process_single_frame(frame, annotate=False)
        
        # Log detections if any
        if detections:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            for d in detections:
                emotion_log.append({
                    'emotion': d['emotion'],
                    'emoji':   d['emoji'],
                    'confidence': d['confidence'],
                    'time': current_time,
                })
            if len(emotion_log) > 50:
                del emotion_log[:-50]

        return jsonify({'detections': detections})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/emotion_log')
def get_emotion_log():
    return jsonify(list(reversed(emotion_log[-20:])))


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Could not decode image'}), 400

    annotated, detections = process_single_frame(frame, annotate=True)

    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': f'data:image/jpeg;base64,{encoded}',
        'detections': detections,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
