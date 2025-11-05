# ============================================================
#  STREAMLIT CONFIG - MUST BE FIRST
# ============================================================
import streamlit as st
st.set_page_config(page_title="🌾 Crop Disease Detector", layout="centered")

# ============================================================
#  LIBRARY IMPORTS
# ============================================================
import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ============================================================
# 🧠 LOAD MODEL (NO STREAMLIT DECORATORS ABOVE CONFIG)
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mobile_corn_model.h5")

def load_cnn_model():
    """Loads the trained CNN model."""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

#  use Streamlit’s caching *after* config has been set
model = st.cache_resource(load_cnn_model)()

# ============================================================
#  STREAMLIT UI START
# ============================================================
st.title(" Crop Disease Detection (Mobile + Camera Ready)")
st.write("Upload or capture an image to identify crop diseases using MobileNetV2 model.")

# PREDICTION FUNCTION

def predict_disease(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    label = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)
    return label, conf

# USER INPUT SELECTION

option = st.radio(
    "📸 Select Input Type:",
    ["Capture from Camera", "Upload Image", "Upload Video (MP4)"]
)

# 1. CAMERA INPUT

if option == "Capture from Camera":
    st.info("Use your mobile or webcam to capture an image.")
    img_file = st.camera_input("Take a photo", key="camera_input_1")
 

    if img_file is not None:
        try:
            image = Image.open(img_file).convert("RGB")
            st.image(image, caption="Captured Leaf", use_container_width=True)
            frame = np.array(image)

            label, conf = predict_disease(frame)
            st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")

        except Exception as e:
            st.error(f"❌ Error processing captured image: {e}")


# 2. IMAGE UPLOAD

if option == "Capture from Camera":
    st.info("Use your mobile or webcam to capture an image.")
    img_file = st.camera_input("Take a photo")

    if img_file is not None:
        try:
            image = Image.open(img_file).convert("RGB")
            st.image(image, caption="Captured Leaf", use_container_width=True)
            frame = np.array(image)

            label, conf = predict_disease(frame)
            st.success(f"Prediction: **{label}** ({conf*100:.2f}%)")

        except Exception as e:
            st.error(f"❌ Error processing captured image: {e}")


# 3. VIDEO UPLOAD

elif option == "Upload Video (MP4)":
    video_file = st.file_uploader("Upload a short video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        try:
            # Save the uploaded video temporarily
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.read())

            # Try to open with OpenCV
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("❌ Could not open the uploaded video. Please check the format.")
            else:
                st.info("🎥 Processing video... please wait.")
                stframe = st.empty()
                frame_count = 0
                success_frames = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    try:
                        # Convert frame to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        label, conf = predict_disease(frame_rgb)
                        success_frames += 1

                        # Overlay label
                        cv2.putText(frame_rgb,
                                    f"{label} ({conf*100:.1f}%)",
                                    (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2)

                        stframe.image(frame_rgb, channels="RGB")

                    except Exception as frame_err:
                        st.warning(f"⚠️ Error processing frame {frame_count}: {frame_err}")
                        continue

                cap.release()
                if success_frames == 0:
                    st.error("⚠️ No valid frames were processed.")
                else:
                    st.success(f"✅ Finished processing {success_frames} frames.")

        except Exception as e:
            st.error(f"❌ Error reading or processing the uploaded video: {e}")
        finally:
            # Cleanup temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

# FOOTER

st.markdown("---")
st.markdown(
    "📱 **Tip:** Works on mobile browsers. Open this app via local Wi-Fi IP to test live capture."
)







