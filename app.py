from flask import Flask, request, render_template, jsonify
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import io

app = Flask(__name__)

# Load model and labels
MODEL_PATH = "emotion_recognition_lstm.h5"
LABELS_PATH = "label_classes.npy"
model = tf.keras.models.load_model(MODEL_PATH)
label_classes = np.load(LABELS_PATH, allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# MFCC extraction from in-memory bytes
def extract_mfcc_from_bytes(audio_bytes, n_mfcc=13, max_len=100):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    return np.expand_dims(mfcc, axis=0)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    features = extract_mfcc_from_bytes(audio_bytes)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return jsonify({
        "predicted_emotion": predicted_label,
        "confidence": float(np.max(prediction))
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

