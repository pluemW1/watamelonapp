import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

# โหลดโมเดล
model_path = 'model/watermelon_model.h5'

if not os.path.exists(model_path):
    st.error(f"Error: Model file not found at {model_path}")
else:
    model = tf.keras.models.load_model(model_path)

# ฟังก์ชัน preprocess
def preprocess_audio_file(file_path, target_length=862):
    data, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)

    if mfccs.shape[1] < target_length:
        pad_width = target_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :target_length]

    mfccs_processed = np.expand_dims(mfccs, axis=-1)
    return mfccs_processed

# ส่วนต่อประสานผู้ใช้
st.title('แอพจำแนกความสุกของแตงโม')

uploaded_file = st.file_uploader("อัปโหลดไฟล์เสียง", type=["wav"])

if uploaded_file is not None:
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format='audio/wav')
    
    processed_data = preprocess_audio_file(file_path)
    prediction = model.predict(np.expand_dims(processed_data, axis=0))
    predicted_class = np.argmax(prediction)
    result = 'สุก' if predicted_class == 0 else 'ไม่สุก'
    
    st.write(f"ผลการวิเคราะห์: {result}")
