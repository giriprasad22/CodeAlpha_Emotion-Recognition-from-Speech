# CodeAlpha_Emotion-Recognition-from-Speech

# 🎤 Speech Emotion Recognition Web App

An interactive **Speech Emotion Recognition** system built with **Python, Deep Learning, and Flask** that can detect human emotions from speech audio (`.wav` / `.mp3`) using an LSTM model trained on the **TESS dataset**.  
The app provides a **modern web interface** with drag-and-drop upload for instant predictions.

---

## 📌 Features
- 🎯 **Emotion Detection** from speech using deep learning (MFCC features + LSTM model)
- 🎨 **Modern responsive UI** with drag & drop or file selection
- 📊 **Confidence meter** showing model prediction probability
- ⚡ **No file storage** — audio is processed in-memory
- 🔗 **Flask backend** with API endpoint for predictions
- 🌈 **Emotion icons & styling** for better visual feedback

---

## 🛠 Tech Stack
**Backend:**
- Python 3.x
- Flask
- Librosa (audio feature extraction)
- TensorFlow / Keras (LSTM deep learning model)
- NumPy, scikit-learn

**Frontend:**
- HTML5, CSS3
- Vanilla JavaScript (fetch API for AJAX calls)
- Responsive design

---

## 📂 Dataset
This project uses the **[TESS Toronto Emotional Speech Set](https://tspace.library.utoronto.ca/handle/1807/24487)** dataset for training.
> *Note: Dataset is not included in the repo. Please download it separately.*

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository
- git clone https://github.com/yourusername/speech-emotion-recognition.git
- cd speech-emotion-recognition


### 2️⃣ Create & activate a virtual environment
- python -m venv venv
- Activate:
- Windows
- venv\Scripts\activate

- macOS/Linux
- source venv/bin/activate

### 3️⃣ Install dependencies
- pip install -r requirements.txt

### 4️⃣ Prepare the dataset
- Download and extract the TESS dataset.
- Place the dataset folder in your project directory.
- Update `DATA_DIR` in `prepare_data.py` to point to your dataset path.

### 5️⃣ Preprocess data & train the model
- python prepare_data.py
- python train_model.py

### This will generate:
- `emotion_recognition_lstm.h5` → trained LSTM model
- `label_classes.npy` → encoded emotion labels
- Prepared `.npy` feature files

---

## 🚀 Run the Web App
-  python app.py
-  Open your browser and go to:
-  Open your browser and go to:
- Drag & drop or browse for an audio file.
- Instantly get predicted emotion with confidence score.

---

## 📁 Project Structure
speech-emotion-recognition/

│

├── app.py # Flask backend (runs the web app)

├── prepare_data.py # Preprocess dataset, extract MFCCs, encode labels

├── train_model.py # Train the LSTM model

├── requirements.txt # Python dependencies

│

├── emotion_recognition_lstm.h5 # Saved trained model

├── label_classes.npy # Saved encoded classes

├── predict_emotion # A Simple file to predict with web interface

│

├── templates/

│ └── index.html # Frontend HTML

│

├── static/

│ └── style.css # Frontend CSS styling

│

└── README.md # Project documentation





---

## 🔮 Future Improvements
- Real-time microphone recording & prediction
- More advanced deep learning models (CNN-LSTM hybrids)
- Multi-language emotion recognition
- Deployment to cloud (Heroku, Render, etc.)

---


