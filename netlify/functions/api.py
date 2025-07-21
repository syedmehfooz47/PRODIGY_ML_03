import os
import numpy as np
import pickle
import warnings
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.svm import SVC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Correctly locate the model files within the Netlify environment
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'svm_image_classifier.pkl')
FEATURE_EXTRACTOR_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'vgg16_feature_extractor.h5')

app = Flask(__name__)
global_model_data = {}


def load_model_and_extractor():
    """Load the SVM model and VGG16 feature extractor."""
    try:
        with open(MODEL_PATH, 'rb') as model_file:
            svm_model = pickle.load(model_file)
        feature_extractor = VGG16(weights='imagenet', include_top=False)
        # The line below is not needed when loading a pre-trained model.
        # feature_extractor.load_weights(FEATURE_EXTRACTOR_PATH)
        return svm_model, feature_extractor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def predict_image_class(image_path, svm_model, feature_extractor):
    """Predict the class of an image."""
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = feature_extractor.predict(img_array)
        features = features.flatten().reshape(1, -1)
        prediction = svm_model.predict(features)
        return 'Dog' if prediction[0] == 1 else 'Cat'
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error"


# This is the handler that Netlify will use
def handler(event, context):
    """Netlify serverless function handler."""
    if 'svm_model' not in global_model_data:
        svm_model, feature_extractor = load_model_and_extractor()
        if svm_model is None:
            return {'statusCode': 500, 'body': 'Model could not be loaded.'}
        global_model_data['svm_model'] = svm_model
        global_model_data['feature_extractor'] = feature_extractor

    if event['httpMethod'] == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filepath = os.path.join('/tmp', file.filename)
            file.save(filepath)

            prediction = predict_image_class(
                filepath,
                global_model_data['svm_model'],
                global_model_data['feature_extractor']
            )
            os.remove(filepath)
            return jsonify({'prediction': prediction})

    return jsonify({'error': 'Invalid request method'}), 405


if __name__ == '__main__':
    app.run(debug=True)