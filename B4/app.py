#Cài trước khi chạy streamlit
# pip


import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps

# Configure page
st.set_page_config(
   page_title="ASL Alphabet Recognition",
   page_icon="🐧",
   layout="centered",
   initial_sidebar_state="expanded",
)

#------------------------------------------
# Load model
@st.cache_resource
def load_model():
    model = keras.models.load_model("asl_alphabet_model.h5")
    return model
model = load_model()

#-------------------------------------------
# Khai báo lớp + img_size
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space',
    'nothing'
]
IMG_SIZE = 64

# -----------------------------------
# proprecess function
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)
    img_array = img_array / 255.0  # scale anh
    img_array = np.expand_dims(img_array, axis=0)  # them kich thuoc
    return img_array

#------------------
# streamlit app UI
st.title("ASL Alphabet Recognition")

input_type = st.radio(
    "Choose input type:",
    ("Upload Image", "Use Webcam"),
    index = 0
)

uploaded_file = st.file_uploader(
    "Upload an image of hand sign:",
    type=["jpg", "jpeg", "png"]
)


if input_type == "Use Webcam":
    st.warning("Webcam input is not supported in this version. Please uploade an image instead")
if input_type == "Upload Image" and uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Upload Image", use_container_width= True)
    
    # % đự đoán thấp nhất
    THRESHOLD = 0.7
    
    if st.button("Predict"):
        #Tải phần loading cho dự đoán
        with st.spinner("Predicting..."):
            img = preprocess_image(img)
            predictions = model.predict(img)
            print(predictions)
            confidence = np.max(predictions)
            if confidence < THRESHOLD:
                ppredicted_class = "Uncertain Prediction"
            else:
                
                predicted_class =class_names[np.argmax(predictions)]
        st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
        
        