############################### Imports ##################################

import logging
import tensorflow as tf

from utils import load_config
from preprocess import DataParser

##########################################################################


########################### Global Variables #############################

CONFIG = load_config()
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='logs/preprocess.log',
                   filemode='w')

##########################################################################


class DeepSignTrainer:
    def __init__(self):
        logging.info("DeepSignTrainer initialized.")

        # Get the dataset
        try:
            self.dataset = self._get_dataset()
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise
        else:
            logging.info(f"Dataset loaded.")

        # Build the model

        

    def _get_dataset(self) -> tf.data.Dataset:
        """        
        Loads the training dataset from the specified TFRecord file.

        Returns:
            tf.data.Dataset: A TensorFlow dataset containing the training data.
        """
        parser = DataParser()
        return parser.get_dataset(tfrecord_path=CONFIG['data']['train_serialized_path'])


    def _get_model(self) -> tf.keras.Model:
        pass


    def train(self, model: tf.keras.Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset | None) -> None:
        pass



if __name__ == "__main__":
    trainer = DeepSignTrainer()
    # Assuming you have a model to train
    # model = trainer._get_model()
    # trainer.train(model, trainer.dataset, None)

