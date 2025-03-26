import os
import cv2
import numpy as np

# Set paths
DATASET_DIR = 'TrashNet'
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = 100  # Resize to 100x100

def load_data():
    data = []
    labels = []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATASET_DIR, category)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img.flatten())
                labels.append(label)
    return np.array(data), np.array(labels)