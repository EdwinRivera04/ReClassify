from keras.src.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.src.utils import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle

# Settings
DATASET_DIR = 'TrashNet'
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = 224  # Required input size for MobileNetV2

# Load pre-trained model (no top layers, for feature extraction)
feature_extractor = EfficientNetB0(weights="imagenet", include_top=False, pooling='avg', input_shape=(224, 224, 3))

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']  # You can try 'linear' or 'poly' later
}

# Load and extract features
def load_and_extract_features():
    data = []
    labels = []
    raw_images = []  # For visualization
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATASET_DIR, category)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                raw_images.append(img)  # Save raw image for later display
                img = img_to_array(img)
                img = preprocess_input(img)
                features = feature_extractor.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                data.append(features)
                labels.append(label)
    return np.array(data), np.array(labels), raw_images

# Get features and labels
X, y, raw_images = load_and_extract_features()

# Split data
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(X, y, raw_images, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
clf = grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, refit=True, verbose=2, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

# Use the best model found
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameters Found:", grid.best_params_)
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

stats = {
    "best_params": grid.best_params_,
    "report": classification_report(y_test, y_pred, target_names=CATEGORIES, output_dict=False)
}

# Save test data and predictions
with open('model/sample_test_data.pkl', 'wb') as f:
    pickle.dump({
        'X_test': X_test,
        'y_test': y_test,
        'img_test': img_test,
        'y_pred': y_pred
    }, f)

with open('model/eval_stats.txt', 'w') as f:
    f.write("Best Parameters Found:\n")
    f.write(str(stats["best_params"]) + "\n\n")
    f.write(stats["report"])

# Visualize some predictions
def show_sample_predictions():
    for i in range(5):
        idx = np.random.randint(0, len(X_test))
        img = img_test[idx]
        true_label = CATEGORIES[y_test[idx]]
        pred_label = CATEGORIES[y_pred[idx]]
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"True: {true_label} | Pred: {pred_label}")
        plt.axis('off')
        plt.show()

show_sample_predictions()


def predict_image(image_path, model, scaler, feature_extractor):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)

    # Extract features using EfficientNet
    features = feature_extractor.predict(np.expand_dims(img_array, axis=0), verbose=0)

    # Normalize the feature vector
    features_scaled = scaler.transform(features)

    # Predict with SVM
    prediction = model.predict(features_scaled)[0]
    predicted_label = CATEGORIES[prediction]

    # Show image with prediction
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()


# Run prediction on image.jpg
predict_image("image.jpg", best_model, scaler, feature_extractor)