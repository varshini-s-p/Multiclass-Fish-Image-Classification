import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load custom CNN model
MODEL_PATH = 'models/outputs/efficientnetb0_model.h5'  
model = tf.keras.models.load_model(MODEL_PATH)

# Class names — ensure this matches your training order
class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# App title
st.title("🐟 Fish Species Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

# Only run prediction after image is uploaded
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
 
    # Preprocess image
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize((224, 224))  # Resize to match training input size
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    top_prediction_idx = np.argmax(predictions)
    predicted_class = class_names[top_prediction_idx]
    confidence_score = predictions[top_prediction_idx] * 100

    # Display prediction
    st.markdown(f"### 🐠 Predicted Species: `{predicted_class}`")
    st.markdown(f"**Confidence Score:** `{confidence_score:.2f}%`")

    # Show top 3 predictions
    st.markdown("### Top 3 Predictions:")
    top_3 = predictions.argsort()[-3:][::-1]
    for i in top_3:
        st.markdown(f"`{class_names[i]}` - **{predictions[i]*100:.2f}%**")
