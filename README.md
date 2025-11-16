#  Corn Crop Disease Detection Using CNN & Transfer Learning

This project focuses on building a deep learning model to classify corn (maize) leaf diseases using image-based analysis. Early detection of crop diseases is crucial for preventing yield loss, improving productivity, and supporting farmers with automated diagnostic tools.

The model classifies **4 corn leaf categories**:

- **Gray Leaf Spot**
- **Healthy**
- **Blight**
- **Common Rust**

This project includes a full pipeline: dataset preparation, CNN & Transfer Learning training, model evaluation, and a Streamlit-based interactive UI for real-time predictions.

## Features

- ✔️ Classifies 4 major corn leaf diseases  
- ✔️ High accuracy using **CNN + Transfer Learning (MobileNet/VGG16)**  
- ✔️ Real-time image upload & webcam support  
- ✔️ Streamlit UI  
- ✔️ Preprocessing pipeline for agricultural datasets  
- ✔️ Lightweight model suitable for edge deployment 

## Model Architecture

The model was trained using:

- **Custom CNN layers**
- **Transfer Learning (VGG16 / MobileNetV2)**
- **Fine-tuning last layers**
- **Softmax classification (4 classes)**
Key steps:
1. Image resizing & normalization  
2. Augmentation (rotation, flip, zoom)  
3. Feature extraction using pretrained CNN  
4. Fully connected classifier  
5. Cross-entropy training
    
deployment link 
https://corncropdiseasedetection.streamlit.app/

![image alt](https://github.com/pavankalyan-127/Crop_Disease_Detection-/blob/main/corn_1.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/Crop_Disease_Detection-/blob/main/corn_2.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/Crop_Disease_Detection-/blob/main/corn_3.jpg?raw=true)
![image alt](https://github.com/pavankalyan-127/Crop_Disease_Detection-/blob/main/corn_4.jpg?raw=true)

