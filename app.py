import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
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
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "mobile_corn_model_colab1.h5")

    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()

    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()
st.success("✅ Model loaded successfully!")

# ============================================================
# 🔮 Prediction Function
# ============================================================

IMG_SIZE = (224, 224)
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

def predict_disease(frame):
    try:
        # 🔥 IMPORTANT: No color conversion here

        img = cv2.resize(frame, IMG_SIZE)
        img = preprocess_input(img)   # MobileNetV2 preprocessing
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img, verbose=0)
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

# ============================================================
# 📷 CAMERA INPUT
# ============================================================

if option == "Capture from Camera":
    st.info("Use your mobile or webcam to capture a leaf image.")
    img_file = st.camera_input("Take a photo")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        frame = np.array(image)

        label, conf = predict_disease(frame)

        st.image(image, caption=f"{label} ({conf*100:.2f}%)", use_container_width=True)

# ============================================================
# 🖼 IMAGE UPLOAD
# ============================================================

elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)

        label, conf = predict_disease(frame)

        st.image(image, caption=f"{label} ({conf*100:.2f}%)", use_container_width=True)

# ============================================================
# 🎥 VIDEO UPLOAD (FIXED VERSION)
# ============================================================

elif option == "Upload Video (MP4)":
    video_file = st.file_uploader("Upload a short video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        temp_path = "temp_video.mp4"

        try:
            # Save uploaded video
            with open(temp_path, "wb") as f:
                f.write(video_file.read())

            cap = cv2.VideoCapture(temp_path)

            if not cap.isOpened():
                st.error("❌ Could not open video.")
            else:
                st.info("🎥 Processing video frame-by-frame...")

                stframe = st.empty()
                frame_count = 0
                processed = 0
                skip_frames = 5  # 🔥 optimize performance

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # 🔥 Skip frames to reduce load
                    if frame_count % skip_frames != 0:
                        continue

                    processed += 1

                    # Convert BGR → RGB (ONLY here)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    label, conf = predict_disease(frame_rgb)

                    # Draw prediction
                    cv2.putText(frame_rgb,
                                f"{label} ({conf*100:.1f}%)",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2)

                    stframe.image(frame_rgb, channels="RGB")

                cap.release()

                st.success(f"✅ Processed {processed} frames!")

        except Exception as e:
            st.error(f"❌ Error: {e}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# ============================================================
# 🧾 Footer
# ============================================================

st.markdown("---")
st.markdown("👨‍💻 Developed by **Pavan Kalyan** | Model: CNN (MobileNetV2)")
