import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# ============================================================
# 🌾 Crop Disease Detection Streamlit App
# ============================================================

st.set_page_config(page_title="🌾 Crop Disease Detection", layout="centered")

st.title("🌾 Crop Disease Detection using CNN")
st.markdown("Upload an image or video of a leaf to detect the disease.")

# ============================================================
# 🧠 Load Model
# ============================================================

@st.cache_resource
def load_cnn_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "mobile_corn_model_colab1.h5")  # or .keras
    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()
st.success("✅ Model loaded successfully!")

# ============================================================
# 🔮 Prediction Function
# ============================================================

IMG_SIZE = (224, 224)
CLASS_NAMES = ['Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy']  # match your training order

def predict_disease(frame):
    """
    Preprocess image and predict disease.
    """
    try:
        # Ensure RGB format
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize & normalize
        img = cv2.resize(frame, IMG_SIZE)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        label = CLASS_NAMES[np.argmax(preds)]
        conf = float(np.max(preds))
        return label, conf

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return "Unknown", 0.0

# ============================================================
# 📸 User Input Section
# ============================================================

option = st.radio(
    "📷 Select Input Type:",
    ["Capture from Camera", "Upload Image", "Upload Video (MP4)"]
)

# 1️⃣ Capture from Camera
if option == "Capture from Camera":
    st.info("Use your mobile or webcam to capture a leaf image.")
    img_file = st.camera_input("Take a photo", key="camera_input_1")

    if img_file is not None:
        try:
            image = Image.open(img_file).convert("RGB")
            frame = np.array(image)
            label, conf = predict_disease(frame)
            st.image(image, caption=f"Prediction: {label} ({conf*100:.2f}%)", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error processing captured image: {e}")

# 2️⃣ Upload Image
elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            label, conf = predict_disease(frame)
            st.image(image, caption=f"Prediction: {label} ({conf*100:.2f}%)", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error processing uploaded image: {e}")

# 3️⃣ Upload Video
elif option == "Upload Video (MP4)":
    video_file = st.file_uploader("Upload a short video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        try:
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.read())

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("❌ Could not open the uploaded video.")
            else:
                st.info("🎥 Processing video... please wait.")
                stframe = st.empty()
                success_frames = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    success_frames += 1
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    label, conf = predict_disease(frame_rgb)
                    cv2.putText(frame_rgb, f"{label} ({conf*100:.1f}%)", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    stframe.image(frame_rgb, channels="RGB", caption=f"{label} ({conf*100:.1f}%)")

                cap.release()
                st.success(f"✅ Processed {success_frames} frames successfully!")

        except Exception as e:
            st.error(f"❌ Error processing video: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# ============================================================
# 🧾 Footer
# ============================================================
st.markdown("---")
st.markdown("👨‍💻 Developed by **Pavan Kalyan** | Model: CNN (MobileNetV2 trained in Colab)")


