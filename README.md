Crop Disease Detection using Deep Learning (Mobile-Ready)
1.Overview :

This project detects crop leaf diseases using deep learning (CNN / MobileNetV2).
It provides a web-based interface built with Streamlit that works seamlessly on mobile devices and desktops.
Users can upload images, record directly from their camera, or upload videos to get real-time predictions of crop health.

2.Features:

Trained using MobileNetV2 / VGG16 / ResNet50 architectures.
Upload or capture leaf images directly from your mobile camera.
Supports video uploads for continuous leaf health analysis.
Deployable on Streamlit Cloud for global access.
Optimized for fast inference on mobile and web.

3.Technologies Used:

Python 3.x
TensorFlow / Keras
OpenCV
NumPy & Pillow
Streamlit

4.Dataset:
The model was trained on a Corn Leaf Disease dataset, containing images of:
Healthy Leaves
Blight
Common Rust
Gray Leaf Spot
NOTE:(You can replace with your own dataset path or Kaggle source.)

STEPS TO IMPLEMENT:
1.Clone this repository:
git clone https://github.com/pavankalyan-127/Crop_Disease_Detection-.git
cd Crop_Disease_Detection-
2.nstall dependencies:
pip install -r requirements.txt
3.Run the Streamlit app:
streamlit run app.py
4.Open in your browser:
http://localhost:8501

