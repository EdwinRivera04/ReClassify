# ReClassify: Trash Classification with EfficientNet + SVM

An automated trash classification system using **EfficientNetB0** and a fine-tuned **Support Vector Machine (SVM)**.  
ReClassify identifies recyclable materials across six categories to assist with smarter, cleaner waste sorting.

---

## Model Overview

- **Feature Extractor**: EfficientNetB0 (pretrained on ImageNet)
- **Classifier**: Support Vector Machine (RBF kernel)
- **Tuning**: GridSearchCV for `C`, `gamma`, `kernel`
- **Accuracy**: **91% on TrashNet dataset**

---

## Dataset

ReClassify uses the [TrashNet dataset](https://github.com/garythung/trashnet), containing:

- 2,500+ labeled images
- 6 classes:  
  `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- Images resized to **224×224** and preprocessed for EfficientNet

---

## How It Works

1. Load and preprocess images
2. Extract feature vectors with **EfficientNetB0**
3. Normalize features using `StandardScaler`
4. Train an SVM classifier with the best hyperparameters
5. Evaluate accuracy, F1-score, and generate predictions

---

## Results

| Class      | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| Cardboard  | 0.96      | 0.94   | 0.95     |
| Glass      | 0.91      | 0.92   | 0.91     |
| Metal      | 0.84      | 0.91   | 0.87     |
| Paper      | 0.90      | 0.97   | 0.93     |
| Plastic    | 0.94      | 0.85   | 0.89     |
| Trash      | 0.89      | 0.67   | 0.76     |

- **Overall Accuracy**: **91%**
- **Macro F1 Score**: **0.89**
- **Weighted F1 Score**: **0.90**

---

## Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

**Main packages:**
- `tensorflow` / `keras`
- `scikit-learn`
- `opencv-python`
- `matplotlib`
- `numpy`

---

**Run the classifier:**

```bash
python main.py
```

This script will:
- Load images from the `TrashNet/` dataset
- Extract features using **EfficientNetB0**
- Train and evaluate the **SVM** model
- Display classification report and prediction samples

---

## Project Structure

```
├── TrashNet/                # Dataset (6 folders by class)
├── main.py                  # Main training & evaluation script
├── preprocess.py            # Image loading/preprocessing helper
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Add image augmentation to boost low-sample classes like `trash`
- Try **ResNet** or **EfficientNetV2** for deeper feature extraction
- Deploy as a web app using **Streamlit** or **Flask**
- Implement **real-time webcam classification**

---

## Author

**Edwin Rivera**  
[GitHub](https://github.com/EdwinRivera04)  
[LinkedIn](https://www.linkedin.com/in/edwin-rivera04/)