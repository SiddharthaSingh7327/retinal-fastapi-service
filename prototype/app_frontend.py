import streamlit as st
import requests
from PIL import Image
import io
import os

st.set_page_config(
    page_title="Retinal DR Diagnostics",
    page_icon="eye",
    layout="centered"
)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
st.title("Retinal Diagnostic Assistant")
st.markdown("Upload a fundus image below to perform automated **Diabetic Retinopathy (DR)** assessment powered by PyTorch ResNet18.")
st.divider()
uploaded_file=st.file_uploader("Choose a retinal fundus image...", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image= Image.open(uploaded_file)
    st.image(image, caption="Uploaded Fundus Image", use_container_width=True)
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing image through PyTorch model..."):
            try:
                buf= io.BytesIO()
                image.save(buf, format="PNG")
                byte_im=buf.getvalue()
                files={"file": (uploaded_file.name, byte_im, "image/png")}
                response= requests.post(API_URL, files=files, timeout=30)
                if response.ok:
                    data= response.json()
                    st.success("Analysis Complete")
                    col1, col2= st.columns(2)
                    with col1:
                        st.metric("Predicted Severity", f"Stage {data['predicted_class']}")
                    with col2:
                        st.metric("Model Confidence", f"{data['confidence'] *100:.1f}%")
                    st.markdown(f"### Diagnosis:\n**{data['diagnosis_label']}**")
                    st.progress(data['predicted_class']/4.0)
                else:
                    try:
                        detail = response.json().get("detail", "Unknown error")
                    except ValueError:
                        detail =response.text or "Unknown error"
                    st.error(f"API Error {response.status_code}: {detail}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach API at {API_URL}: {e}")