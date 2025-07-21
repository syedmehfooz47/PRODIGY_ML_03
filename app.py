import os
import numpy as np
import sys
import webbrowser
import pickle
from threading import Timer
import warnings
from PIL import Image
from flask import Flask, render_template, request, jsonify

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.svm import SVC
import random

MODEL_PATH = 'svm_image_classifier.pkl'
FEATURE_EXTRACTOR_PATH = 'vgg16_feature_extractor.h5'

app = Flask(__name__)
global_model_data = {}


def load_images_from_directory(data_dir, sample_size):
    filepaths = []
    labels = []
    class_names = ['cats', 'dogs']

    for label, category in enumerate(class_names):
        category_dir = os.path.join(data_dir, category)
        if not os.path.isdir(category_dir):
            sys.exit(f"Error: Directory not found - {category_dir}")

        all_files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if
                     f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sample_files = random.sample(all_files, min(len(all_files), sample_size // 2))
        filepaths.extend(sample_files)
        labels.extend([label] * len(sample_files))

    if not filepaths:
        sys.exit("Error: No images found. Please check the data directory structure.")

    print(f"Sampled {len(filepaths)} images for training.")
    return filepaths, labels


def extract_features_from_paths(filepaths, model):
    features = []
    total_files = len(filepaths)
    for i, filepath in enumerate(filepaths):
        try:
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array_expanded = np.expand_dims(img_array, axis=0)
            preprocessed_img = preprocess_input(img_array_expanded)
            feature_vector = model.predict(preprocessed_img, verbose=0).flatten()
            features.append(feature_vector)
            sys.stdout.write(f"\rExtracting features: {i + 1}/{total_files}")
            sys.stdout.flush()
        except Exception:
            pass
    print("\nFeature extraction complete.")
    return np.array(features)


def initialize_and_train_model():
    TRAINING_SAMPLE_SIZE = 200
    DATA_DIRECTORY = 'data'

    print("--- Model Training Initialized ---")
    filepaths, labels = load_images_from_directory(DATA_DIRECTORY, TRAINING_SAMPLE_SIZE)

    feature_extractor = VGG16(weights='imagenet', include_top=False, pooling='avg')
    features = extract_features_from_paths(filepaths, feature_extractor)
    labels = np.array(labels)

    svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
    svm_classifier.fit(features, labels)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(svm_classifier, f)

    feature_extractor.save(FEATURE_EXTRACTOR_PATH)

    print(f"✅ Model trained and saved to '{MODEL_PATH}'")
    return feature_extractor, svm_classifier


def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_EXTRACTOR_PATH):
        print(f"Loading pre-trained model from '{MODEL_PATH}'...")
        with open(MODEL_PATH, 'rb') as f:
            svm_classifier = pickle.load(f)
        from tensorflow.keras.models import load_model
        feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
        print("✅ Model loaded successfully.")
    else:
        feature_extractor, svm_classifier = initialize_and_train_model()

    global_model_data['feature_extractor'] = feature_extractor
    global_model_data['svm_classifier'] = svm_classifier


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(img_array_expanded)

        feature_extractor = global_model_data['feature_extractor']
        svm_classifier = global_model_data['svm_classifier']

        image_features = feature_extractor.predict(preprocessed_img, verbose=0).flatten()
        image_features = np.array([image_features])

        prediction = svm_classifier.predict(image_features)
        prediction_proba = svm_classifier.predict_proba(image_features)

        label_map = {0: 'Cat', 1: 'Dog'}
        predicted_label = label_map[prediction[0]]
        confidence = np.max(prediction_proba)

        print(f"Prediction: {predicted_label} | Confidence: {confidence:.2%}")

        return jsonify({
            'prediction': predicted_label,
            'confidence': f"{confidence:.2%}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == '__main__':
    load_or_train_model()
    Timer(1, open_browser).start()
    app.run(port=5000, debug=False)
