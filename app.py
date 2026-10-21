import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("YOLOv11 Object Detection Web App")

# Model Loader
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Sidebar Threshold
confidence = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25)

# File Uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)
    
    if st.button("Detect Objects"):
        with st.spinner("Detecting objects..."):
            results = model.predict(source=image, conf=confidence)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Detected Objects", use_container_width=True)
            
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.subheader("Detection Details")
                detected_names = [model.names[int(cls)] for cls in boxes.cls]
                unique_counts = {name: detected_names.count(name) for name in set(detected_names)}
                
                st.write(f"Total objects detected: {len(boxes)}")
                st.write(unique_counts)
            else:
                st.write("No objects detected. Try lowering the confidence threshold.")
