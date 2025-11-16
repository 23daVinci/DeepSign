from PIL import Image
import tensorflow as tf
import io
from preprocess import DataSerializer



import tensorflow as tf

# --- Use the SAME constants from training ---
IMAGE_HEIGHT = 105
IMAGE_WIDTH = 105

def preprocess_image_for_inference(image_path):
    """
    Loads an image file, decodes it, resizes, and normalizes it
    identically to the training pipeline.
    """
    
    # 1. Read the raw file bytes
    raw_bytes = tf.io.read_file(image_path)
    
    # 2. Decode as grayscale (1 channel)
    #    tf.io.decode_image is robust (handles PNG, JPEG, etc.)
    img = tf.io.decode_image(raw_bytes, channels=1)
    
    # 3. Resize to the model's expected input size
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    
    # 4. Normalize pixels from [0, 255] to [0.0, 1.0]
    img = tf.cast(img, tf.float32) / 255.0
    
    return img