############################### Imports ##################################

import logging
import tensorflow as tf
from huggingface_hub import hf_hub_download

from utils import load_config
from preprocess import DataParser
from models import DeepSign

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

        # Get the compiled model
        self.model = self._get_model()

        
    def _get_dataset(self, source: str = 'local') -> tf.data.Dataset:
        """        
        Loads the training dataset from the specified TFRecord file.

        Returns:
            tf.data.Dataset: A TensorFlow dataset containing the training data.
        """
        if source == 'HF_hub':
            # Download data from Hugging Face Hub
            data_path = self._download_data_from_HF_hub()
            
        parser = DataParser()
        return parser.get_dataset(tfrecord_path=CONFIG['data']['train_serialized_path'])


    def _get_model(self) -> tf.keras.Model:
        img_height = CONFIG['data']['img_height']
        img_width = CONFIG['data']['img_width']
        embedding_dim = CONFIG['model']['embedding_dim']

        deep_sign = DeepSign()
        # Build and compile the model
        model = deep_sign.build_and_compile_model(
                                                    input_shape=(img_height, img_width, 1),
                                                    embedding_dim=embedding_dim
                                                )
        return model


    def _download_data_from_HF_hub(self):
        """
        Downloads the training data from Hugging Face Hub.

        Returns:
            downloaded_data: The downloaded data.
        """

        downloaded_data = None
        try:
            downloaded_daa = hf_hub_download(
                                                repo_id=CONFIG['data']['HuggingFace']['train_repo_id'], 
                                                filename=CONFIG['data']['HuggingFace']['train_filename'] 
                                            )
        except Exception as e:
            logging.error(f"Error downloading data from Hugging Face Hub: {e}")
            raise
        else:
            logging.info(f"Data downloaded successfully from Hugging Face Hub.")
            return downloaded_data


    def train(self) -> None:
        try:
            logging.info("Starting training...")
            self.model.fit(
                            self.dataset,
                            epochs=CONFIG['training']['epochs'],
                            callbacks=[
                                tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss',
                                    patience=5,
                                    restore_best_weights=True
                                )
                            ]
                          )
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise
        else:
            logging.info("Training completed successfully.")
        

    def get_trained_model(self) -> tf.keras.Model:
        """
        Returns the trained model.

        Returns:
            tf.keras.Model: The trained DeepSign model.
        """
        return self.model




if __name__ == "__main__":
    trainer = DeepSignTrainer()
    # Start training
    trainer.train()

