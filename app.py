import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas


st.title("MNIST Digit Recognizer with Streamlit")

model = tf.keras.models.load_model(r"cnn.h5", compile=False)

canvas_size = 280 
img = Image.new("L", (canvas_size, canvas_size), color=0)
draw = ImageDraw.Draw(img)

st.write("Draw a digit (0-9) below:")
canvas = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=canvas_size,
    height=canvas_size,
)

if canvas.image_data is not None:
    
    img_small = Image.fromarray(canvas.image_data[:, :, 0]).resize((28,28)).convert("L")
    img_array = np.array(img_small).reshape(1,28,28,1)/255.0

    pred = model.predict(img_array)
    st.write("Predicted probabilities:")
    for i, p in enumerate(pred[0]):
        st.write(f"{i}: {p:.4f}")


    st.write(f"Predicted digit: {np.argmax(pred)}")




