import cv2
import numpy as np
import streamlit as st

prototxt_path = r"C:\Users\ArnettaNT\Documents\thonny latihan\deploy.prototxt"
caffemodel_path = r"C:\Users\ArnettaNT\Documents\thonny latihan\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def detect_and_count_faces_dnn(image):
    (h, w) = image.shape[:2]
    image_resized = cv2.resize(image, (300, 300))  
    blob = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    num_faces = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.14: 
            num_faces += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 7)  
    return image, num_faces

st.title("Deteksi Wajah dengan DNN dan OpenCV")
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_with_faces, num_faces = detect_and_count_faces_dnn(image)
    image_rgb = cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption=f"Jumlah wajah yang terdeteksi: {num_faces}", use_column_width=True)
    st.write(f"Jumlah wajah yang terdeteksi: {num_faces}")
