from flask import Flask, request, render_template, redirect
import os
import cv2
import numpy as np
import pickle
from keras.models import load_model
from keras.src.utils import img_to_array
from keras.src.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

# Init app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model components
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
model = pickle.load(open('model/best_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))
feature_extractor = load_model('model/feature_extractor.h5')

# Prediction function
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)

    features = feature_extractor.predict(np.expand_dims(img_array, axis=0), verbose=0)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    return CATEGORIES[prediction]

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']

        # Make sure the upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)

        label = predict_image(img_path)
        return render_template('index.html', label=label, img_path=img_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)