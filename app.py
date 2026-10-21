import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="YOLOv11 Object Detection App",
    page_icon="🔍",
    layout="wide"
)

st.title("YOLOv11 End-to-End Object Detection")
st.write("SMIT AI Project - Object Detection Web App")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    if st.button("Run Detection"):
        with st.spinner("Detecting objects..."):
            results = model.predict(source=image, conf=confidence)
            res_plotted = results[0].plot()
            res_image = Image.fromarray(res_plotted[:, :, ::-1])
            
            with col2:
                st.subheader("Detected Objects")
                st.image(res_image, use_column_width=True)
                
            st.markdown("---")
            st.subheader("Detection Details")
            boxes = results[0].boxes
            if len(boxes) > 0:
                detected_names = [model.names[int(cls)] for cls in boxes.cls]
                unique_counts = {name: detected_names.count(name) for name in set(detected_names)}
                st.success(f"Total objects detected: {len(boxes)}")
                st.json(unique_counts)
            else:
                st.warning("No objects detected. Try lowering the confidence threshold.")