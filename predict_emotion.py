import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

# ===== Load Trained Model =====
model = tf.keras.models.load_model("emotion_recognition_lstm.h5")

# ===== Load Label Encoder Classes (from prepare_data.py) =====
# You must use the same label mapping as in training
# If you still have the encoder, save its classes when preparing data
label_classes = np.load("label_classes.npy", allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# ===== Function to Extract MFCC (same as in training) =====
def extract_mfcc(file_path, n_mfcc=13, max_len=100):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    return np.expand_dims(mfcc, axis=0)  # add batch dimension

# ===== Predict Emotion for a New Audio File =====
def predict_emotion(audio_path):
    features = extract_mfcc(audio_path)
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# ===== Example Usage =====
if __name__ == "__main__":
    test_audio = "YAF_white_neutral.wav"  # change this to your test file
    emotion = predict_emotion(test_audio)
    print(f"Predicted Emotion: {emotion}")
