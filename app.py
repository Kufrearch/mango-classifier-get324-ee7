import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Mango Vision — Fresh vs Rotten Classifier",
    page_icon="🥭",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
    <style>
    /* Main container background & typography */
    .main {
        padding-top: 1.5rem;
    }
    
    /* Card Container Styling */
    .card-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    /* Result Badge Styles */
    .result-badge-fresh {
        background-color: #e6f4ea;
        color: #137333;
        border: 1px solid #ceead6;
        padding: 16px;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin-top: 15px;
    }
    .result-badge-rotten {
        background-color: #fce8e6;
        color: #c5221f;
        border: 1px solid #fad2cf;
        padding: 16px;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin-top: 15px;
    }

    /* Metric visual enhancement */
    .metric-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        background: #ffffff;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
        border: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar contents
with st.sidebar:
    st.image("https://img.icons8.com/color/96/mango.png", width=64)
    st.title("About the System")
    st.markdown("""
    **Course Project:** GET324 (Group EE7)  
    **Architecture:** MobileNetV2 Transfer Learning  
    **Target Classes:** Binary (Fresh vs. Rotten)
    
    ---
    ### Model Performance
    - **Optimization:** TensorFlow / Keras 3
    - **Input Shape:** 224 × 224 × 3
    - **Cloud Engine:** Streamlit Cloud
    """)
    st.divider()
    st.caption("© GET324 Group EE7. All Rights Reserved.")

# Main Header UI
st.title("🥭 Mango Quality Classifier")
st.markdown("Automated binary classification of mango fruit quality using deep neural networks.")

# Usage & OOD Warning Banner
st.notice = st.warning(
    "💡 **Usage Notice:** This deep learning model is strictly trained for **Mango Image Classification**. "
    "Uploading non-mango objects (e.g., books, furniture, other fruits) will yield non-meaningful predictions due to model boundary constraints."
)

# Model Caching function
@st.cache_resource
def load_mango_model():
    return tf.keras.models.load_model('mango_classifier.h5', compile=False)

try:
    model = load_mango_model()
except Exception as e:
    st.error(f"❌ Failed to initialize inference engine: {e}")
    st.stop()

# File Uploader Section
st.subheader("1. Upload Image")
uploaded_file = st.file_uploader(
    "Choose a crisp image of a mango (.jpg, .jpeg, .png)", 
    type=["jpg", "jpeg", "png"],
    help="For optimal results, ensure the mango is well-lit and centered in the frame."
)

if uploaded_file is not None:
    st.subheader("2. Analysis & Inference")
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Source Input", use_container_width=True)
    
    with col2:
        with st.spinner("Processing feature extraction..."):
            # Preprocessing to match MobileNetV2 input tensors
            target_size = (224, 224)
            image_resized = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Predict raw logits
            prediction = model.predict(img_batch)
            confidence_score = float(prediction[0][0])
            
        st.markdown("#### Diagnostic Report")
        
        # Binary Threshold evaluation (0 -> Fresh, 1 -> Rotten)
        if confidence_score > 0.5:
            percent = confidence_score * 100
            st.markdown(
                f'<div class="result-badge-rotten">🛑 Classification: Rotten</div>', 
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="metric-container">
                    <div><strong>Quality Index:</strong> Defective / Degraded</div>
                    <div><strong>Confidence:</strong> {percent:.2f}%</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            percent = (1.0 - confidence_score) * 100
            st.markdown(
                f'<div class="result-badge-fresh">✅ Classification: Fresh</div>', 
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="metric-container">
                    <div><strong>Quality Index:</strong> High Grade / Healthy</div>
                    <div><strong>Confidence:</strong> {percent:.2f}%</div>
                </div>
                """, 
                unsafe_allow_html=True
            )

