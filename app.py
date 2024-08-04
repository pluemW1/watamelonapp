import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import boto3
import os
import tarfile

# กำหนดค่า AWS S3
bucket_name = 'my-watermelon-models'
model_file_name = 'model_keras.tar.gz'
model_file_path = 'model/model.keras'

# กำหนด AWS credentials และ Region
aws_access_key_id = 'AKIAQKGGXRGHVXFZREWH'
aws_secret_access_key = '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'
region_name = 'ap-southeast-1'

# ตรวจสอบว่าโฟลเดอร์ model มีอยู่หรือไม่ ถ้าไม่มีให้สร้าง
if not os.path.exists('model'):
    os.makedirs('model')

# ดาวน์โหลดโมเดลจาก S3 พร้อมจัดการข้อผิดพลาด
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

try:
    if not os.path.exists(model_file_path):
        st.info(f"Downloading {model_file_name} from S3 bucket {bucket_name}...")
        s3.download_file(bucket_name, model_file_name, 'model/model_keras.tar.gz')
        st.success("Model downloaded successfully.")

        # แยกไฟล์ที่บีบอัด
        with tarfile.open('model/model_keras.tar.gz', 'r:gz') as tar:
            tar.extractall(path='model/')
except s3.exceptions.NoSuchBucket:
    st.error(f"The specified bucket does not exist: {bucket_name}")
except s3.exceptions.NoSuchKey:
    st.error(f"The specified key does not exist: {model_file_name}")
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# เพิ่มการจัดการข้อผิดพลาดเมื่อโหลดโมเดล
try:
    model = tf.keras.models.load_model(model_file_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

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
    
    st.success(f"ผลการวิเคราะห์: {result}")

    confidence = np.max(prediction)
    st.write(f"ความมั่นใจของการทำนาย: {confidence:.2f}")
