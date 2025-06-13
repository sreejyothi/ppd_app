import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import torch
from torch.nn import Sequential
from ultralytics.nn.tasks import ClassificationModel



class_names = ['high', 'low', 'md', 'medium', 'zero']  # update if needed

@st.cache_resource
def load_model():
    try:
        torch.serialization.add_safe_globals([ClassificationModel, Sequential])
        model = YOLO("cassava_ppd_yolov8.pt")  
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

st.markdown("<h1 style='color:#198754;'>🧪 PPD Score Prediction from Tuber Images of Cassava (YOLOv8)</h1>", unsafe_allow_html=True)
st.subheader("Upload a cassava tuber image to predict the Postharvest Physiological Deterioration (PPD) score.")

uploaded_file = st.file_uploader("📤 Choose a cassava tuber image...", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("### 📁 Class Examples")
example_folder = "examples"
for class_name in class_names:
    class_path = os.path.join(example_folder, class_name)
    if os.path.isdir(class_path):
        st.sidebar.markdown(f"**{class_name.upper()}**")
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        for img_file in image_files[:2]:
            img_path = os.path.join(class_path, img_file)
            st.sidebar.image(img_path, use_container_width=False, width=160)
    else:
        st.sidebar.warning(f"No folder for '{class_name}'")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", width=300)

    temp_path = "temp_uploaded_image.jpg"
    image.save(temp_path)

    try:
        results = model(temp_path)
        top1_class_index = results[0].probs.top1
        predicted_class = class_names[top1_class_index]
        st.success(f"✅ Predicted Class: **{predicted_class.upper()}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
