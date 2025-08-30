import tensorflow as tf
import io
import os
from PIL import Image
import pandas as pd
import logging
import matplotlib.pyplot as plt

from utils import load_config 


############## Global Variables ##############

CONFIG = load_config()
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='logs/preprocess.log',
                   filemode='w')

##############################################


class DataSerializer:
    """
    Class to serialize image pairs into TFRecord format for training a signature matching model.
    This class handles loading images, encoding them, and wrapping them into TensorFlow Example format.
    It also reads image pairs and their labels from a CSV file.

    Note: No preprocessing like resizing or normalization is done here; images are stored in their original form.

    Attributes:
        None: This class does not have any instance attributes.
    
    Methods:
        load_and_encode_images(img_path: str) -> tf.image.encode_png:
            Loads an image from its path, converts it to grayscale, and standardizes it.
        
        create_example(img1_bytes: tf.image.encode_png, img2_bytes: tf.image.encode_png, label) -> tf.train.Example:
            Wraps an image pair into tf.train.Example.

        serialize(set: str) -> None:
            Serializes image pairs into a TFRecord file.
    """
    def __init__(self):
        LOGGER.info("DataSerializer initialized.")


    def load_and_encode_images(self, img_path: str) -> tf.image.encode_png:
        """
        Loads and image from its path, converts it to grayscale, and standardises it.

        Args:
            img_path (str): Disk path of the image

        Returns:
            Byte string representation of the image
        """
        #image = Image.open(img_path).convert('L')  # convert to grayscale
        image = Image.open(img_path)
        #image = image.resize((CONFIG['data']['img_width'], CONFIG['data']['img_height']))       # standardize size
        # Convert to numpy and add channel dimension: [H, W] → [H, W, 1]
        image_array = tf.convert_to_tensor(image, dtype=tf.uint8)
        #image_array = tf.expand_dims(image_array, axis=-1)  # Now shape is (H, W, 1)
        return tf.image.encode_png(image_array)
    

    def create_example(self, img1_bytes: tf.image.encode_png, img2_bytes: tf.image.encode_png, label) -> tf.train.Example:
        """
        Wraps an image pair into tf.train.Example.

        Args:
            img1_bytes (tf.image.encode_png): Byte string representation of the first image
            img2_bytes (tf.image.encode_png): Byte string representation of the second image
            label (int): Pair label

        Returns:
            tf.train.Example of the image pair.
        """
        feature = {
            'image1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img1_bytes.numpy()])),
            'image2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img2_bytes.numpy()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    

    def serialize(self, set: str) -> None:
        """ 
        Serializes image pairs into a TFRecord file.
        
        Args:
            set (str): The dataset type, can be 'train', or 'test'.

        Returns:
            None: Writes the serialized data to a TFRecord file.
        """
        LOGGER.info(f"Serializing {set} dataset...")
        # Get image pairs
        image_pairs = self._get_image_pairs(set)

        with tf.io.TFRecordWriter(CONFIG['data']['train_serialized_path']) as writer:
            try:
                for img1_path, img2_path, label in image_pairs:
                    # Get byte string representations of images
                    img1_bytes = self.load_and_encode_images(img1_path)
                    img2_bytes = self.load_and_encode_images(img2_path)
                    # Create a serialized example of image pair and label for TFRecord
                    example = self.create_example(img1_bytes, img2_bytes, label)
                    writer.write(example.SerializeToString())
            except Exception as e:
                LOGGER.error(f"Error processing image pair {img1_path}, {img2_path}: {e}")
                raise e
            else:
                LOGGER.info(f"Serialized {len(image_pairs)} image pairs to TFRecord for {set} set.")


    def _get_image_pairs(self, set: str) -> list[tuple]:
        """
        Reads the CSV file containing image pairs and their labels, and returns a list of tuples.
        
        Args:
            set (str): The dataset type, can be 'train' or 'test'.

        Returns:
            list(tuple): A list of tuples where each tuple contains the paths of two images and their label.
        """
        if set == 'train':
            CSV_PATH = CONFIG['data']['train_pairs']
            IMG_DIR = CONFIG['data']['train_dir']
        else:
            CSV_PATH = CONFIG['data']['test_pairs']
            IMG_DIR = CONFIG['data']['test_dir']

        img_pairs_df = pd.read_csv(CSV_PATH, header=None, names=['img1', 'img2', 'label'])

        # Append full path to each image and convert to tuple list
        image_pair_list = [
                            (
                                os.path.normpath(os.path.join(IMG_DIR, row.img1)),
                                os.path.normpath(os.path.join(IMG_DIR, row.img2)),
                                row.label
                            )
                            for row in img_pairs_df.itertuples(index=False)
                          ]

        return image_pair_list
        




class DataParser:
    """ 
    Class to parse TFRecord files into TensorFlow datasets for training a signature matching model.
    
    Methods:
        test_batches(num_batches: int = 1):
            Displays a few batches of images from the dataset for visual inspection.
        
        parse_example(example_proto):
            Parses a single TFRecord example into image tensors and label.
            
        get_dataset(tfrecord_path: str, batch_size: int = 32) -> tf.data.Dataset:
            Creates a TensorFlow dataset from a TFRecord file.
    """

    def __init__(self) -> None:
        LOGGER.info("DataParser initialized.")
        self.train_dataset = None
        self.val_dataset = None

    
    def _count_records(self, dataset: tf.data.Dataset) -> int:
        """
        Counts the number of records in the dataset.

        Args:
            dataset (tf.data.Dataset): The TensorFlow dataset to count records in.

        Returns:
            int: The number of records in the dataset.        
        """
        try:
            records_count = dataset.reduce(0, lambda x, _: x + 1).numpy()
        except Exception as e:
            LOGGER.error(f"Error counting records in dataset: {e}")
            raise ValueError(f"Error counting records in dataset: {e}")
        
        return records_count

    
    def _get_train_and_val_sets(self, dataset: tf.data.Dataset) -> tuple:
        """
        Splits the dataset into training and validation sets based on the configured validation fraction.

        Args:
            dataset (tf.data.Dataset): The TensorFlow dataset to split.

        Returns:
            tuple: A tuple containing the training and validation datasets.        
        """
        val_fraction = CONFIG['training']['val_fraction']

        if val_fraction >= 1.0:
            raise ValueError("Validation fractions must be less than 1.0")
        
        total_records = self._count_records(dataset)
        val_size = int(total_records * val_fraction)

        val_dataset = dataset.take(val_size)
        train_dataset = dataset.skip(val_size)

        return (train_dataset, val_dataset)


    def test_batches(self, num_batches: int = 1):
        """        
        Displays a few batches of images from the dataset for visual inspection.
        
        Args:
            num_batches (int): Number of batches to display. Default is 1.

        Returns:
            None: Displays images using matplotlib.
        """
        for (img1_batch, img2_batch), labels in self.dataset.take(num_batches):
            batch_size = img1_batch.shape[0]
            for i in range(batch_size):
                plt.figure(figsize=(4,2))
                
                plt.subplot(1,2,1)
                plt.imshow(img1_batch[i, :, :, 0], cmap='gray')
                plt.title("Image 1")
                plt.axis('off')
                
                plt.subplot(1,2,2)
                plt.imshow(img2_batch[i, :, :, 0], cmap='gray')
                plt.title(f"Image 2\nLabel: {labels[i].numpy()}")
                plt.axis('off')
                
                plt.show()


    def parse_example(self, example_proto):
        """
        Parses a single TFRecord example into image tensors and label.
        Args:
            example_proto: A serialized TFRecord example.   
        Returns:
            A tuple containing two image tensors and a label tensor.
        """
        feature_description = {
            'image1': tf.io.FixedLenFeature([], tf.string),
            'image2': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        try:
            parsed = tf.io.parse_single_example(example_proto, feature_description)
        except tf.errors.InvalidArgumentError as e:
            LOGGER.error(f"Failed to parse example: {e}")
            raise ValueError(f"Failed to parse example: {e}")
        
        img1 = tf.image.decode_png(parsed['image1'], channels=1)
        img2 = tf.image.decode_png(parsed['image2'], channels=1)

        img1 = tf.image.convert_image_dtype(img1, tf.float32)
        img2 = tf.image.convert_image_dtype(img2, tf.float32)
        label = tf.cast(parsed['label'], tf.float32)

        # Converting the image to grayscale
        img1 = tf.image.rgb_to_grayscale(img1)
        img2 = tf.image.rgb_to_grayscale(img2)

        # Resize images to the target size
        img1 = tf.image.resize_with_pad(img1, CONFIG['data']['img_height'], CONFIG['data']['img_width'])
        img2 = tf.image.resize_with_pad(img2, CONFIG['data']['img_height'], CONFIG['data']['img_width'])

        # Rescale pixel values to [0, 1]
        img1 = img1 / 255.0 
        img2 = img2 / 255.0
        
        return (img1, img2), label


    def get_dataset(self, tfrecord_path: str, batch_size: int = 32) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Creates a TensorFlow dataset from a TFRecord file.
        
        Args:
            tfrecord_path (str): Path to the TFRecord file.
            batch_size (int): Size of the batches to be returned by the dataset.
        
        Returns:
            tuple: A tuple containing the training and validation datasets.
        """

        try:
            print(f"Loading TFRecord file from: {tfrecord_path}")
            dataset = tf.data.TFRecordDataset(tfrecord_path, buffer_size=CONFIG['data']['TFRecord_buffer_size'])
        except Exception as e:
            LOGGER.error(f"TFRecord file not found at {tfrecord_path}: {e}")
            raise FileNotFoundError(f"TFRecord file not found at {tfrecord_path}: {e}")
        
        # Apply parsing function to each example in the dataset
        dataset = dataset.map(self.parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(CONFIG['data']['shuffle_buffer_size']) 

        # Split into training and validation sets
        train_dataset, val_dataset = self._get_train_and_val_sets(dataset)

        # Creating batches
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        # Prefetching 
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        return (self.train_dataset, self.val_dataset)





if __name__ == "__main__":
    #serializer = DataSerializer()
    # Serialize the training dataset
    #serializer.serialize('train')

    data_parser = DataParser()
    # Parse the training TFRecord file and create a dataset
    train_ds, val_ds = data_parser.get_dataset(CONFIG['data']['train_serialized_path'], batch_size=CONFIG['data']['batch_size'])

    # Test the dataset by displaying a few batches of images
    #data_parser.test_batches(num_batches=1)