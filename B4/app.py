#Cài trước khi chạy streamlit



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
            img_input = preprocess_image(img)

        # du doan hinh
        prediction = model.predict(img_input)
        predicted_probs = tf.nn.softmax(prediction[0])

        # Get top 2 predictions
        top_2_indices = np.argsort(predicted_probs)[-2:]
        top_1_prob = predicted_probs[top_2_indices[1]]
        top_2_prob = predicted_probs[top_2_indices[0]]

        PROBABILITY_GAP = 0.3  # top prediction should be at least 30% higher

        if top_1_prob < 0.6 or (top_1_prob - top_2_prob) < PROBABILITY_GAP:
            # Use st.error instead of print for Streamlit
            st.error("❌ ERROR: Ambiguous or invalid image!")
            st.warning(f"Top prediction: {top_1_prob:.2%}, Second: {top_2_prob:.2%}")
            st.info("Please provide a clear ASL hand sign image.")
        else:
            predicted_class = class_names[top_2_indices[1]]
            confidence = top_1_prob.numpy()  # Convert to Python float
            
            st.success(
                f"✅ Prediction: **{predicted_class}** "
                f"(Confidence: {confidence * 100:.2f}%)"
            )
            
            # Optional: Show top 3 predictions
            st.write("### Top 3 Predictions:")
            top_3_indices = np.argsort(predicted_probs)[-3:][::-1]
            for idx in top_3_indices:
                st.write(f"- {class_names[idx]}: {predicted_probs[idx]:.2%}")
