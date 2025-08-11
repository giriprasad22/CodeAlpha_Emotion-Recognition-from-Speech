import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# ====== Step 1: Set dataset path ======
# Change this to your dataset location
DATA_DIR = 'TESS Toronto emotional speech set data'

# ====== Step 2: Load file paths & labels ======
audio_files = []
labels = []

for emotion_folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, emotion_folder)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                audio_files.append(os.path.join(folder_path, file_name))
                labels.append(emotion_folder)

print(f"Total audio files found: {len(audio_files)}")

# ====== Step 3: Extract MFCC features ======
def extract_mfcc(file_path, n_mfcc=13, max_len=100):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
    
    # Pad or truncate to fixed length
    if len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]
    
    return mfcc

features = [extract_mfcc(f) for f in audio_files]
features = np.array(features)

# ====== Step 4: Encode Labels ======
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 💾 Save label classes for use in prediction
np.save("label_classes.npy", label_encoder.classes_)

# One-hot encoding for deep learning
labels_onehot = tf.keras.utils.to_categorical(labels_encoded)

print("Feature shape:", features.shape)
print("Label shape:", labels_onehot.shape)
print("Classes (emotions):", label_encoder.classes_)

# ====== Step 5: Split into Train/Test ======
X_train, X_test, y_train, y_test = train_test_split(
    features, labels_onehot,
    test_size=0.2,
    random_state=42,
    stratify=labels_onehot
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ====== Step 6: Save Preprocessed Data ======
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Data preprocessing complete. Saved X_train, X_test, y_train, y_test, and label_classes.npy.")
