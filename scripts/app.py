import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template, jsonify

from utils import load_config


############## Global Variables ##############

CONFIG = load_config()

##############################################

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = '../artifacts/trained_model.keras'
IMG_HEIGHT =  CONFIG['data']['img_height'] # Must match your training config
IMG_WIDTH = CONFIG['data']['img_width']

# --- Load Model (Global Scope) ---
# We load this once when the app starts, not on every request.
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
print("Model loaded successfully.")

def preprocess_image_from_stream(file_stream):
    """
    Reads bytes directly from the Flask file object, decodes,
    resizes, and normalizes.
    """
    # 1. Read bytes from the uploaded file stream
    raw_bytes = file_stream.read()
    
    # 2. Decode (Universal decoder)
    img = tf.io.decode_image(raw_bytes, channels=1, expand_animations=False)
    
    # 3. Explicitly set shape (Critical for graph execution safety)
    img.set_shape([None, None, 1])
    
    # 4. Resize
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    
    # 5. Normalize (0-255 -> 0.0-1.0)
    img = tf.cast(img, tf.float32) / 255.0
    
    # 6. Add Batch Dimension (H, W, C) -> (1, H, W, C)
    img = tf.expand_dims(img, axis=0)
    
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Check if images are present
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Please upload both image1 and image2'}), 400
        
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        # 2. Preprocess both images
        # We pass the file objects directly
        input_a = preprocess_image_from_stream(file1)
        input_b = preprocess_image_from_stream(file2)
        
        # 3. Predict
        # The model expects a list of inputs: [input_a, input_b]
        prediction = model.predict([input_a, input_b])
        
        # 4. Extract score (Convert numpy float to Python float for JSON)
        similarity_score = float(prediction[0][0])
        
        # Threshold logic (optional)
        verdict = "SAME" if similarity_score > 0.5 else "DIFFERENT"
        
        return jsonify({
            'similarity_score': similarity_score,
            'verdict': verdict
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run locally
    #print("Starting Flask server...")
    #app.run(debug=True, port=5000)

    print("Loading model...")
    model = tf.keras.models.load_model("../artifacts/trained_model.keras")
    print("Model loaded successfully.")