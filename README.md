# Dual ML Application: Pima Diabetes Risk Predictor & OCR Engine

## Overview

This repository contains two machine learning pipelines:
- Pima Diabetes Risk Predictor: A classification model trained on the Pima Indians Diabetes dataset to estimate an individual’s risk of diabetes.
- CNN-based Alphanumeric OCR Engine: A convolutional neural network that recognizes handwritten or printed alphanumeric characters from images.

## Methodology

### Pima Diabetes Risk Predictor
- Dataset: Pima Indians Diabetes dataset (768 samples, 8 features).
- Data Preparation: Handled missing values, performed feature scaling, and split data into training and test sets.
- Modeling: Experimented with Logistic Regression, Random Forest, and Support Vector Machine.
- Evaluation: Compared models using accuracy, precision, recall, F1-score, and ROC-AUC.

### CNN-based Alphanumeric OCR Engine
- Dataset: Custom or MNIST-extended dataset for alphanumeric characters.
- Preprocessing: Image resizing, grayscale conversion, normalization.
- Architecture: Multi-layer CNN built with TensorFlow/Keras (Conv → ReLU → Pool → Dense → Softmax).
- Pipeline: Ingest image → preprocess → predict character → post-process output.


## Real-World Applications
- Healthcare: Early diabetes risk screening tool to assist clinicians and patients in proactive health management.
- Document Digitization: Automated transcription of forms, invoices, license plates, and archival documents.
- Accessibility: Assistive tools for visually impaired users by converting images of text into digital form.

## Technology Comparison

| Component  | Chosen Technology                          | Alternatives                        | Rationale                                                             |
|------------|--------------------------------------------|-------------------------------------|-----------------------------------------------------------------------|
| Predictor  | Logistic Regression, Random Forest, SVM<br/>(Scikit-learn) | K-Nearest Neighbors, Naive Bayes      | Balances interpretability, performance, and broad community support.  |
| OCR Engine | Convolutional Neural Network<br/>(TensorFlow / Keras)    | Tesseract OCR, SVM + HOG features    | End-to-end feature learning yields higher accuracy on complex imagery.|


## Repository Structure
```
Dual-ML-Application-Pima-Diabetes-Risk-Predictor-And-OCR-Engine/
├── diabetes_predictor/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluate.py
│   └── diabetes_model.pkl
├── ocr_engine/
│   ├── preprocess.py
│   ├── cnn_model.py
│   ├── predict.py
│   └── ocr_weights.h5
└── README.md
└── Requirements.txt
```

### What each file does

###### `diabetes_predictor/`

- ***`data_preprocessing.py`***
  Loads the Pima dataset, handles missing values, scales features, and splits into train/test sets.

- ***`model_training.py`***  
  Trains your chosen classifiers (Logistic Regression, Random Forest, SVM) and saves the best model.

- ***`evaluate.py`***  
  Loads the saved model, runs predictions on the test set, and computes metrics (accuracy, precision, recall, F1, ROC-AUC).

- ***`diabetes_model.pkl`***  
  Serialized (“pickled”) best-performing model for fast inference.

###### `ocr_engine/`

- ***`preprocess.py`***  
  Reads input images, resizes to the CNN’s expected dimensions, converts to grayscale, and normalizes pixel values.

- ***`cnn_model.py`***  
  Defines and trains your convolutional architecture; at completion, exports weights to `ocr_weights.h5`.

- ***`predict.py`***  
  Loads `ocr_weights.h5` and runs inference on a new image, outputting the recognized character.

- ***`ocr_weights.h5`***  
  The trained CNN’s weights—used by `predict.py` for character recognition.

###### Top-Level

- ***`README.md`***  
  Project overview, instructions, and documentation.

- ***`requirements.txt`***  
  Lists all Python dependencies (e.g., scikit-learn, TensorFlow) for easy setup.

## Installation & Usage
### Clone the Repository

```bash
git clone https://github.com/sruthi7sri/Dual-ML-Application-Pima-Diabetes-Risk-Predictor-And-OCR-Engine.git
cd Dual-ML-Application-Pima-Diabetes-Risk-Predictor-And-OCR-Engine
```
### Create & Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Diabetes Predictor:
```bash
python diabetes_predictor/model_training.py
python diabetes_predictor/evaluate.py 
```
### Run OCR Engine:
```bash
python ocr_engine/cnn_model.py
python ocr_engine/predict.py --image path/to/sample.png    
```
## Dependencies:
```bash
numpy>=1.19.5
pandas>=1.1.5
scikit-learn>=0.24.1
tensorflow>=2.4.0
matplotlib>=3.3.2
pillow>=8.0.1
opencv-python>=4.5.1
```
## Project Goals
- Accurate Risk Prediction: Develop and validate a classification model to estimate an individual’s risk of diabetes using the Pima Indians Diabetes dataset.
- Robust OCR Capability: Build a convolutional neural network pipeline that reliably recognizes handwritten or printed alphanumeric characters from varied image sources.
- End-to-End Workflow: Showcase complete ML workflows from data ingestion and preprocessing through model training, evaluation, and saving for inference.
- Deployment Readiness: Provide clear instructions and tooling (including a requirements file and GitHub deployment steps) for seamless integration into portfolios and applications.
- Scalability & Extensibility: Structure code and documentation to allow easy extension to other medical datasets or OCR character sets.

## License
© 2024 Sruthisri. All rights reserved.