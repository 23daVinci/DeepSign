import os
import tensorflow as tf
from PIL import Image
import numpy as np
import tempfile

from preprocess import DataSerializer 
from utils import load_config


tf.get_logger().setLevel('ERROR')

############## Global Variables ##############

CONFIG = load_config()

##############################################


def test_load_and_encode_image():
    # Create a temporary grayscale image
    img_array = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='L')

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = f"{tmp.name}"
        img.save(temp_path)

    serializer = DataSerializer()
    encoded = serializer.load_and_encode_images(temp_path)

    # Check if returned object is a Tensor
    print(isinstance(encoded, tf.Tensor))

    # Check if dtype is string (byte string)
    print(encoded.dtype == tf.string)

    # Decode to check size and shape
    decoded_img = tf.image.decode_png(encoded, channels=1)
    print(decoded_img.shape == (CONFIG['data']['img_height'], CONFIG['data']['img_width'], 1))  # Height, Width, Channels
    #print(decoded_img.shape)

    # Cleanup
    os.remove(temp_path)


if __name__ == "__main__":
    test_load_and_encode_image()
