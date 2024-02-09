# Breast_ultrasound_Classification


Breast ultrasound classification with Deep Learning involves training CNN models using Keras on labeled datasets to distinguish between benign and malignant abnormalities. After achieving high accuracy, a Streamlit application is developed for easy deployment, providing clinicians with accurate predictions on ultrasound images for clinical validation.

# What you have to do?
You gotta run the following code "ResNet50v2" and the at the end , it'll automatically saved the model with name "breast.h5".
So download the following save model , and then open the streamlit code, so you'll get the following code ...

import numpy as np
import cv2 
import streamlit as st
from keras.utils import load_img, img_to_array
from keras.models import load_model

model=load_model('Breast_ultrosound.h5')
st.title("BREAST_ULTRASOUND")
class_labels=[
    'benign',
    'malignant',
    'normal'
]
uploader = st.file_uploader('Select image', type=('jpg', 'png'))
if uploader is not None:
    image = load_img(uploader,target_size=(224,224),color_mode='rgb')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image.resize((224, 224)))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    st.subheader("Prediction:")
    d_predicted=class_labels[class_index]
    st.write(f' {d_predicted}')
Do a little bit changes , just change the "Breast_ultrasound.h5" with "breast.h5" in streamlit code , as it downloads automatically when we run the following Google Colab code.
