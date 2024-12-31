# import libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# page title
st.set_page_config(page_title="MelanomaScan", initial_sidebar_state="auto")

# title
st.markdown("<h1 style='text-align: center;'>Melanoma Detection</h1>", unsafe_allow_html=True)

# side bar
with st.sidebar:
    st.write("What is Melanoma?")
    st.image("abbcde melanoma.jpg")
    st.caption("Image from www.freepik.com")
    styled_artikel = """
    <p style="text-align: justify;">Melanoma is a skin cancer that develops in melanocytes, the pigment cells. 
    It is the most malignant form of skin cancer due to its high risk of spreading to other parts of the body. 
    Early signs of melanoma include the appearance of new moles or changes in existing moles. 
    To differentiate melanoma from regular moles, we can identify the following signs:</p>
    <ul>
        <li><b>A</b>symmetric: A mole that is not symmetrical, meaning one side is different from the other.</li>
        <li><b>B</b>order: A mole with irregular, jagged, or blurred edges.</li>
        <li><b>C</b>olor: A mole that has multiple colors in one area, such as black, brown, red, white, or blue.</li>
        <li><b>D</b>iameter: A mole with a diameter larger than 6 mm.</li>
        <li><b>E</b>volving: A mole that changes in shape, size, or color, or begins to show symptoms like itching or bleeding.</li>
    </ul>"""
    st.write(styled_artikel, unsafe_allow_html=True)

# data preprocessing
def predict(data, model):
    size = (224, 224)    
    image = ImageOps.fit(data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# import model
model = tf.keras.models.load_model('FourthModel.h5') 

# tabs navigation
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Upload an Image"

tab1, tab2 = st.tabs(["Upload an Image", "Take a Picture"])

with tab1:
    st.session_state.active_tab = "Upload an Image"
    uploaded_file = st.file_uploader("Click the button to upload your image", type=["jpg", "jpeg"], label_visibility="hidden")
    image_placeholder = st.empty()

    # classify
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_placeholder.image(image, use_container_width=True)
        predictions = predict(image, model)
        class_names = {0: "Melanoma", 1: "NotMelanoma"}
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class] * 100
        
        if predicted_class == 0:
            st.error("Melanoma Cancer")
            st.error(f"Probability: {confidence:.2f}%")
            st.info("Immediately consult your specialist doctor")
        elif predicted_class == 1:
            st.success("Not Melanoma")
            st.success(f"Probability: {confidence:.2f}%")
    else:
        st.error("Start the diagnosis by uploading an image or taking a picture")

with tab2:
    st.session_state.active_tab = "Take a Picture"
    camera_file = st.camera_input("Activate your camera")
    image_placeholder = st.empty()

    # classify
    if camera_file:
        image = Image.open(camera_file)
        image_placeholder.image(image, use_container_width=True)
        predictions = predict(image, model)
        class_names = {0: "Melanoma", 1: "NotMelanoma"}
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class] * 100
        
        if predicted_class == 0:
            st.error("Melanoma Cancer")
            st.error(f"Probability: {confidence:.2f}%")
            st.info("Immediately consult your specialist doctor")
        elif predicted_class == 1:
            st.success("Not Melanoma")
            st.success(f"Probability: {confidence:.2f}%")
    else:
        st.error("Start the diagnosis by uploading an image or taking a picture")

# hide content if no active input on the respective tab
if st.session_state.active_tab == "Upload an Image" and not uploaded_file:
    image_placeholder.empty()
elif st.session_state.active_tab == "Take a Picture" and not camera_file:
    image_placeholder.empty()
