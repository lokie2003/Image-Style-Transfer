import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import streamlit as st

# Load the style transfer model from TensorFlow Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image
def preprocess_image(image, target_image_size):
    image = image.convert('RGB')
    image = image.resize(target_image_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to convert TensorFlow tensor to PIL image
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# Streamlit app
def main():
    st.title('Style Transfer')
    st.write("Upload a content image and a style image to apply style transfer.")
    
    # Upload content image
    content_image = st.file_uploader("Select Content Image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])
    
    # Upload style image
    style_image = st.file_uploader("Select Style Image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])
    
    if content_image and style_image:
        if st.button("Apply Style Transfer"):
            content_image = Image.open(content_image)
            style_image = Image.open(style_image)
            
            target_image_size = (256, 256)
            content_image = preprocess_image(content_image, target_image_size)
            style_image = preprocess_image(style_image, target_image_size)

            content_image = tf.constant(content_image, dtype=tf.float32)
            style_image = tf.constant(style_image, dtype=tf.float32)

            stylized_image = hub_model(content_image, style_image)[0]

            # Display the stylized image
            st.subheader("Stylized Image")
            st.image(tensor_to_image(stylized_image), use_column_width=True)

if __name__ == '__main__':
    main()
