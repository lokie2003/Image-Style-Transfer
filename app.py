#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__, template_folder='templates')

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the style transfer model from TensorFlow Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image, target_image_size):
    image = image.convert('RGB')
    image = image.resize(target_image_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return redirect(request.url)

    content_file = request.files['content_image']
    style_file = request.files['style_image']

    if content_file.filename == '' or style_file.filename == '':
        return redirect(request.url)

    if content_file and allowed_file(content_file.filename) and style_file and allowed_file(style_file.filename):
        content_filename = secure_filename(content_file.filename)
        style_filename = secure_filename(style_file.filename)

        content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

        content_file.save(content_path)
        style_file.save(style_path)

        target_image_size = (256, 256)
        content_image = Image.open(content_path)
        style_image = Image.open(style_path)

        # Ensure both images are in RGB mode
        content_image = content_image.convert('RGB')
        style_image = style_image.convert('RGB')

        # Resize and preprocess the images
        content_image = preprocess_image(content_image, target_image_size)
        style_image = preprocess_image(style_image, target_image_size)

        # Convert images to TensorFlow tensors
        content_image = tf.constant(content_image, dtype=tf.float32)
        style_image = tf.constant(style_image, dtype=tf.float32)

        stylized_image = hub_model(content_image, style_image)[0]

        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
        tensor_to_image(stylized_image).save(output_image_path)

        return render_template('index.html', result=True, output_image=output_image_path)

    return redirect(request.url)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
