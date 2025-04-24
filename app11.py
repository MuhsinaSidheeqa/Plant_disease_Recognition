import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio
import pickle
from tensorflow.keras.models import load_model
import base64


model = load_model('plant-disease.h5')


class_indices = { 'Healthy': 0, 'Powdery': 1,'Rust': 2}
index_to_class = {v: k for k, v in class_indices.items()}

def add_background_image(image_file):
    with open(image_file, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()


    css_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-position: bottom center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css_style, unsafe_allow_html=True)



with open("history.pkl", "rb") as f:
    history = pickle.load(f)

# Set app page config
st.set_page_config(page_title="Plant Disease Recognition", layout="centered")
st.sidebar.title("📌Navigation")
page = st.sidebar.radio("Go to", ["🏠Home", "📤Upload Image", "📈Model Performance"])

if page == "🏠Home":
    add_background_image('green.jpg')
    st.title("Plant Disease Recognition")
    st.markdown("""
    Welcome to the **Plant Disease Recognition App!** 🌱
    
A simple and fast tool designed to help you maintain the health of your plants. Upload a leaf image, and the app will analyze and identify whether your plant is:
- Healthy 🌿
- Affected by Rust 🍂
- Struggling with Powdery Mildew 🥀
    """)


elif page == "📤Upload Image":
    st.title("📤Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = image.resize((225, 225))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button('Predict'):
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction)
            predicted_label = index_to_class[predicted_index]

            st.success(f"Predicted Class: {predicted_label} ({confidence * 100:.2f}% confidence)")


elif page == "📈Model Performance":
    st.title("Model Performance ")

    layout = go.Layout(
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#FFFFFF", size=14),
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        title=dict(text="Model Accuracy", x=0.5, font=dict(color="#FFFFFF", size=16)),
    )

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(y=[0.5985, 0.8766, 0.9337, 0.9401, 0.9276],
                             mode="lines", name="Accuracy", line=dict(color="#00FF00")))  # Bright green
    fig.add_trace(go.Scatter(y=[0.9333, 0.9500, 0.9667, 0.9667, 0.9333],
                             mode="lines", name="Validation Accuracy", line=dict(color="#00FFFF")))  # Cyan

    fig.show()

    layout = go.Layout(
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#FFFFFF", size=14),
        xaxis_title='Epoch',
        yaxis_title='Loss',
        title=dict(text="Model Loss ", x=0.5, font=dict(color="#FFFFFF", size=16)),
    )

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(y=[0.9103, 0.3423, 0.2030, 0.1727, 0.2017],
                             mode="lines", name="Loss", line=dict(color="#FF0000")))
    fig.add_trace(go.Scatter(y=[0.2149, 0.1431, 0.1212, 0.1608, 0.1285],
                             mode="lines", name="Validation Loss", line=dict(color="#FFFF00")))

    fig.show()