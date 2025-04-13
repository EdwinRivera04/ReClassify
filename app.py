from flask import Flask, request, render_template, redirect
import os
import cv2
import numpy as np
import pickle
from keras.models import load_model
from keras.src.utils import img_to_array
from keras.src.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import shutil

# Init app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SAMPLES_FOLDER = 'static/samples'
os.makedirs(SAMPLES_FOLDER, exist_ok=True)

with open('model/sample_test_data.pkl', 'rb') as f:
    sample_data = pickle.load(f)

X_test = sample_data['X_test']
y_test = sample_data['y_test']
img_test = sample_data['img_test']
y_pred = sample_data['y_pred']


# Load model components
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
model = pickle.load(open('model/best_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))
feature_extractor = load_model('model/feature_extractor.h5')

# Prediction function
from PIL import Image

def predict_image(image_path):
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"❌ Unable to open image: {e}")

    pil_img = pil_img.resize((224, 224))
    img_array = img_to_array(pil_img)
    img_array = preprocess_input(img_array)

    features = feature_extractor.predict(np.expand_dims(img_array, axis=0), verbose=0)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    return CATEGORIES[prediction]

# Sample Prediction Generator
def generate_sample_predictions():
    sample_paths = []

    for i in range(4):
        idx = np.random.randint(0, len(X_test))
        img = img_test[idx]
        true_label = CATEGORIES[y_test[idx]]
        pred_label = CATEGORIES[y_pred[idx]]

        # Convert to RGB and save
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = f"sample_{i}_true_{true_label}_pred_{pred_label}.jpg"
        filepath = os.path.join(SAMPLES_FOLDER, filename)
        plt.imsave(filepath, img_rgb)
        sample_paths.append({
            "path": filepath,
            "true": true_label,
            "pred": pred_label
        })

    return sample_paths

# Routes
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']

        if img.filename == '' or not allowed_file(img.filename):
            return render_template('index.html', label="Invalid file type. Upload a JPG or PNG.")

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)

        try:
            label = predict_image(img_path)
        except ValueError as e:
            return render_template('index.html', label=str(e))

        return render_template('index.html', label=label, img_path=img_path)

    return render_template('index.html')

@app.route('/samples')
def samples():
    # Clear previous sample images
    shutil.rmtree(SAMPLES_FOLDER)
    os.makedirs(SAMPLES_FOLDER, exist_ok=True)

    sample_images = generate_sample_predictions()
    return render_template('samples.html', sample_images=sample_images)

@app.route('/stats')
def stats():
    with open('model/eval_stats.txt', 'r') as f:
        content = f.read()
    return render_template('stats.html', stats_content=content)

if __name__ == '__main__':
    app.run(debug=True)