import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = r"C:\Users\pavan\Downloads\cropDetection_cnn\crop_disease_app\mobile_corn_model.h5"  # trained MobileNet model path
IMG_SIZE = (128, 128)
CLASS_NAMES = ['Healthy', 'Blight', 'Common Rust', 'Gray Leaf Spot']  # Update with your classes

# Load Model
@st.cache_resource
def load_cnn_model():
    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# ============================================================
# STREAMLIT PAGE SETTINGS
# ============================================================
st.set_page_config(page_title="🌾 Crop Disease Detector", layout="centered")
st.title("🌱 Crop Disease Detection (Mobile + Camera Ready)")
st.write("Upload or capture a crop leaf image (or video) to detect disease using MobileNetV2 model.")

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_disease(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    label = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)
    return label, conf

# ============================================================
# USER INPUT SELECTION
# ============================================================
option = st.radio("📸 Select Input Type:", 
                  ["Capture from Camera", "Upload Image", "Upload Video (MP4)"])

# ============================================================
# 1️⃣ CAMERA INPUT
# ============================================================
if option == "Capture from Camera":
    st.info("Use your mobile or webcam to capture an image.")
    img_file = st.camera_input("Take a photo")

    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Captured Leaf", use_container_width=True)
        frame = np.array(img)
        label, conf = predict_disease(frame)
        st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")

# ============================================================
# 2️⃣ IMAGE UPLOAD
# ============================================================
elif option == "Upload Image":
    uploaded = st.file_uploader("Upload leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
        frame = np.array(image)
        label, conf = predict_disease(frame)
        st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")

# ============================================================
# 3️⃣ VIDEO UPLOAD
# ============================================================
elif option == "Upload Video (MP4)":
    video_file = st.file_uploader("Upload a short video", type=["mp4", "avi", "mov"])
    if video_file:
        # Save temporary file
        tfile = open("temp_video.mp4", "wb")
        tfile.write(video_file.read())

        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label, conf = predict_disease(frame_rgb)
            cv2.putText(frame_rgb, f"{label} ({conf*100:.1f}%)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            stframe.image(frame_rgb, channels="RGB")
        cap.release()
        st.success("✅ Video processing complete!")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "📱 **Tip:** Works on mobile browsers. Open this app via local Wi-Fi IP to test live capture."
)
