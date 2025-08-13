import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(r"App/cnn.h5")

def recognize_digit(data):
    if data is None:
        return "No Image"

    if isinstance(data, dict):
        data = list(data.values())[0]

    if isinstance(data, np.ndarray):
        data = Image.fromarray(data)

    if not isinstance(data, Image.Image):
        return "Invalid input"

    
    image = np.array(data.convert('L'))
    
    image = np.array(Image.fromarray(image).resize((28,28)))
   
    image = image.reshape((1,28,28,1)).astype('float32') / 255.0

    prediction = model.predict(image)
    return {str(i): float(prediction[0][i]) for i in range(10)}

iface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Sketchpad(scale=280, image_mode='L'), 
    outputs=gr.Label(),
    live=True
)

iface.launch(share=True, show_error=True,debug=True)

