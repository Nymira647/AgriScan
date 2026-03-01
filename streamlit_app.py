import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="AgriScan - Rice Leaf Disease Detector",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #F4F6F9;
    }
    .header-container {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        border-radius: 15px;
        margin-bottom: 3rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 2px;
    }
    .header-subtitle {
        color: #f0f0f0;
        font-size: 1.3rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    .upload-instruction {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    .healthy-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
    }
    .disease-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #EE5A6F 100%);
        color: white;
    }
    .disease-name {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        margin-top: 1rem;
        opacity: 0.95;
    }
    .disease-list {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .disease-list-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .disease-item {
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0;
        background: #F8F9FA;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        color: #333;
        font-size: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        margin-top: 1.5rem;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">AgriScan</h1>
        <p class="header-subtitle">AI-Powered Rice Leaf Disease Detection</p>
    </div>
""", unsafe_allow_html=True)

# Main Layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    # Upload Section
    st.markdown('<p class="upload-instruction">Upload a rice leaf image to analyze disease.</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        # Analyze Button
        if st.button('Analyze Disease'):
            with st.spinner('Analyzing...'):
                try:
                    # Automatically load class names from dataset folder
                    dataset_path = 'Rice_Leaf_AUG'
                    class_names = sorted([d for d in os.listdir(dataset_path) 
                                        if os.path.isdir(os.path.join(dataset_path, d))])
                    
                    # Prepare and send image to backend
                    uploaded_file.seek(0)
                    img_bytes = uploaded_file.read()
                    
                    # Load and preprocess image locally for prediction
                    from tensorflow.keras.models import load_model
                    image = Image.open(io.BytesIO(img_bytes))
                    image = image.convert('RGB')
                    image = image.resize((224, 224))
                    
                    # Convert to array and preprocess
                    img_array = np.array(image)
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Load model and predict
                    model = load_model('agriscan_model.h5')
                    predictions = model.predict(img_array, verbose=0)
                    
                    # Get probabilities
                    probabilities = predictions[0]
                    max_confidence = float(np.max(probabilities)) * 100
                    predicted_class = np.argmax(probabilities)
                    predicted_name = class_names[predicted_class]
                    
                    # Check if predicted class is "Unknown"
                    if predicted_name == 'Unknown':
                        st.session_state['prediction'] = 'Unknown'
                        st.session_state['confidence'] = f'{max_confidence:.2f}%'
                        st.session_state['is_unknown'] = True
                    else:
                        st.session_state['prediction'] = predicted_name
                        st.session_state['confidence'] = f'{max_confidence:.2f}%'
                        st.session_state['is_unknown'] = False
                    
                    st.rerun()
                
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Please run: python app.py")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Results Section
    if 'prediction' in st.session_state and 'confidence' in st.session_state:
        prediction = st.session_state['prediction']
        confidence = st.session_state['confidence']
        confidence_value = float(confidence.replace('%', ''))
        is_unknown = st.session_state.get('is_unknown', False)
        
        if is_unknown:
            # Unknown detection
            st.markdown(f"""
                <div class="result-card disease-card">
                    <p class="disease-name">Unknown</p>
                    <p class="confidence-text" style="font-size: 1.2rem; margin-top: 1.5rem;">
                        The uploaded image does not match known rice leaf disease classes.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Normal prediction
            card_class = "healthy-card" if prediction == 'Healthy Rice Leaf' else "disease-card"
            
            st.markdown(f"""
                <div class="result-card {card_class}">
                    <p class="disease-name">{prediction}</p>
                    <p class="confidence-text">{confidence}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.progress(confidence_value / 100)

with col2:
    # Right Side Panel - Disease List (automatically loaded from dataset)
    try:
        dataset_path = 'Rice_Leaf_AUG'
        if os.path.exists(dataset_path):
            class_names = sorted([d for d in os.listdir(dataset_path) 
                                if os.path.isdir(os.path.join(dataset_path, d))])
            
            disease_list_html = '<div class="disease-list"><div class="disease-list-title">Supported Classes</div>'
            for class_name in class_names:
                disease_list_html += f'<div class="disease-item">{class_name}</div>'
            disease_list_html += '</div>'
            
            st.markdown(disease_list_html, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="disease-list">
                    <div class="disease-list-title">Supported Classes</div>
                    <div class="disease-item">Dataset folder not found</div>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown("""
            <div class="disease-list">
                <div class="disease-list-title">Supported Classes</div>
                <div class="disease-item">Error loading classes</div>
            </div>
        """, unsafe_allow_html=True)
