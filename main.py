import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

# create a streamlit app
st.set_page_config(page_title="Deteksi Nominal Uang Rupiah")
st.title("Deteksi Nominal Uang Rupiah Indonesia")
st.header("Aplikasi web untuk mendeteksi Nominal Uang Rupiah Indonesia Emisi 2016")

model = load_model('detectRP.h5')
labels = ['1000', '10000', '100000', '2000', '20000', '5000', '50000']

image = st.file_uploader("Silahkan Unggah Gambar Uang Anda", type=["jpg", "png", "jpeg"])
if image is None:
    st.text('Menunggu Gambar Diunggah....')
else:
    slot = st.empty()
    slot.text('Sedang Memprediksi....')

    test_image = Image.open(image)
    st.image(image, channels="BGR", caption="Input Image", width=400)
    # preprocess the image
    cv_image = np.array(test_image)
    image = cv2.resize(cv_image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    # predict the label
    prediction = model.predict(image)
    label = labels[np.argmax(prediction)]
    score = format(np.max(prediction)*100, '.2f')
    output = 'Label yang diprediksi adalah ' + label + ' dengan Skor kepercayaan : ' + score + '%'
    slot.text('Prediksi Selesai!')
    st.success(output)
    
