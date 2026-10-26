import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="YOLOv8 Visual Analytics Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Load YOLOv8 Model (cached so it loads once)
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")  # lightweight, fast nano model

model = load_yolo_model()

# Header
st.title("🔍 Real-Time Object Detection & Visual Analytics")
st.markdown(
    "Upload an image to detect, localize, and count multi-class objects in real time using **YOLOv8**."
)
st.write("---")

# Sidebar Controls
st.sidebar.header("Model Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.4, 
    step=0.05
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Image", 
    type=["jpg", "jpeg", "png"]
)

# Main Detection Logic
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    # Run Inference
    with st.spinner("Processing image with YOLOv8..."):
        results = model.predict(input_image, conf=confidence_threshold)
        
        # Render bounding boxes onto image (OpenCV BGR format)
        annotated_image = results[0].plot()

    # Layout: Side-by-side comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
    with col2:
        st.subheader("YOLOv8 Detections")
        # channels="BGR" ensures accurate colors from results[0].plot()
        st.image(annotated_image, channels="BGR", use_container_width=True)

    # Analytics Section
    st.write("---")
    st.subheader("📊 Detection Analytics")
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        class_ids = boxes.cls.tolist()
        names = model.names
        detected_names = [names[int(cls)] for cls in class_ids]
        counts = Counter(detected_names)

        # Summary Metrics
        st.metric("Total Objects Detected", len(boxes))
        
        # Breakdown by Class
        st.markdown("**Object Counts Breakdown:**")
        cols = st.columns(min(len(counts), 4))
        for idx, (label, count) in enumerate(counts.items()):
            cols[idx % 4].info(f"**{label.capitalize()}**: {count}")
    else:
        st.warning("No objects detected at this confidence threshold. Try lowering the slider on the sidebar.")

else:
    st.info("👈 Please upload an image from the sidebar to begin detection.")