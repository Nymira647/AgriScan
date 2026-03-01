<<<<<<< HEAD
# 🌾 AgriScan - Rice Leaf Disease Detection

AI-powered web application for detecting rice leaf diseases using deep learning.

## 📋 Features

- Detects 6 types of rice leaf conditions
- MobileNetV2 Transfer Learning
- Flask REST API backend
- Streamlit web interface
- Real-time disease prediction with confidence scores

## 🎯 Detected Classes

1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy Rice Leaf
4. Leaf Blast
5. Leaf Scald
6. Sheath Blight

## 📁 Project Structure

```
AgriScan/
├── train.py              # Model training script
├── app.py                # Flask API backend
├── streamlit_app.py      # Streamlit web interface
├── requirements.txt      # Python dependencies
├── agriscan_model.h5     # Trained model (generated after training)
└── Rice_Leaf_AUG/        # Dataset folder (you need to add this)
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your `Rice_Leaf_AUG` dataset folder in the project root directory with the following structure:

```
Rice_Leaf_AUG/
├── Bacterial Leaf Blight/
├── Brown Spot/
├── Healthy Rice Leaf/
├── Leaf Blast/
├── Leaf Scald/
└── Sheath Blight/
```

### 3. Train the Model

```bash
python train.py
```

This will:
- Load the dataset
- Apply data augmentation
- Train MobileNetV2 model for 15 epochs
- Save the model as `agriscan_model.h5`

### 4. Run Flask API

```bash
python app.py
```

The API will start on `http://localhost:5000`

### 5. Run Streamlit App

Open a new terminal and run:

```bash
streamlit run streamlit_app.py
```

The web app will open in your browser at `http://localhost:8501`

## 💻 Usage

1. Make sure both Flask API and Streamlit app are running
2. Open the Streamlit web interface
3. Upload a rice leaf image
4. Click "Check Disease"
5. View the prediction and confidence score

## 🔧 Technical Details

- **Model**: MobileNetV2 (Transfer Learning)
- **Input Size**: 224x224 pixels
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Training Epochs**: 15
- **Validation Split**: 20%
- **Data Augmentation**: Rotation, Zoom, Horizontal Flip

## 📊 Model Training

The model uses:
- ImageDataGenerator with rescaling (1./255)
- Rotation range: 20 degrees
- Zoom range: 0.2
- Horizontal flip: True
- Frozen MobileNetV2 base layers
- Custom Dense layer with 6 outputs (softmax)

## 🌐 API Endpoints

### POST /predict
- Upload image file
- Returns JSON with prediction and confidence

### GET /
- Health check endpoint

## 📝 Notes

- Ensure the dataset is properly organized before training
- Training time depends on dataset size and hardware
- GPU acceleration recommended for faster training
- Model file (agriscan_model.h5) must exist before running the API
=======
# AgriScan
AI-powered rice leaf disease detection system using CNN, TensorFlow, Flask and Streamlit.
>>>>>>> 15d4c5d2963570aaf4eaf3aca66cfd3acbeb11c2
