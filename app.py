import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Fresh vs Rotten Mango Classifier",
    page_icon="🥭",
    layout="centered"
)

st.title("🥭 Fresh vs Rotten Mango Classifier")
st.markdown("Upload an image of a mango to check whether it is **Fresh** or **Rotten**.")

# Cache the model to optimize performance on Streamlit Cloud
@st.cache_resource
def load_mango_model():
    # Keras load_model handles Keras 3 format natively
    return tf.keras.models.load_model('mango_classifier.h5', compile=False)

try:
    model = load_mango_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Image File Uploader
uploaded_file = st.file_uploader("Choose a mango image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.info("Analyzing image quality...")
    
    # Image Preprocessing (Matches MobileNetV2 input shape 224x224)
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized) / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    
    # Model Prediction
    prediction = model.predict(img_reshape)
    confidence = float(prediction[0][0])  # Output value between 0.0 and 1.0
    
    # Class Mapping: 0 -> Fresh, 1 -> Rotten
    if confidence > 0.5:
        score = confidence * 100
        st.error("### Prediction: Rotten Mango 🛑")
        st.write(f"**Confidence Score:** {score:.2f}%")
    else:
        score = (1.0 - confidence) * 100
        st.success("### Prediction: Fresh Mango ✅")
        st.write(f"**Confidence Score:** {score:.2f}%")
    
