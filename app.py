import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="🌾 Crop Infection Detection", layout="centered")

st.title("🌾 Crop Infection Detection")
st.write("Upload a leaf image to detect crop infection using a CNN model.")

MODEL_PATH = r"C:\Users\pavan\Documents\cropDetection_cnn\models\corn_cnn_v2.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))
    st.image(img, caption="Uploaded Image", width=300)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]
    st.write("Raw Prediction Output:", prediction)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    # 🧠 Define your class labels (adjust if needed)
    class_labels = ["Healthy", "Blight", "Gray Leaf Spot", "Common Rust"]

    predicted_label = class_labels[class_index]

    st.markdown("---")
    st.subheader("🌿 Prediction Result:")
    st.write(f"**Predicted Disease:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.write(f"**Predicted Class Index:** {class_index}")
