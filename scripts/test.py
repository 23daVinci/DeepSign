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


def predict_similarity(self, image_path_1, image_path_2, model):
    """
    Takes two image paths, preprocesses them, and asks the model 
    for a similarity score.
    """
    # 1. Preprocess both images individually using your EXISTING function
    # Shape: (H, W, 1)
    img1 = self.preprocess_image_for_inference(image_path_1)
    img2 = self.preprocess_image_for_inference(image_path_2)

    # 2. Add the "Batch Dimension"
    # Models expect inputs as a batch. We create a batch of size 1.
    # Shape becomes: (1, H, W, 1)
    img1_batch = tf.expand_dims(img1, axis=0)
    img2_batch = tf.expand_dims(img2, axis=0)

    # 3. Pass inputs as a LIST to the model
    # The model expects [input_layer_1, input_layer_2]
    prediction = model.predict([img1_batch, img2_batch])

    # 4. Extract the score
    # The output is a numpy array like [[0.85]]. We want the float 0.85.
    similarity_score = prediction[0][0]
    
    return similarity_score