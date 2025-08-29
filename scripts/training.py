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
    """
    A class to handle the training of the DeepSign model.
    
    Methods:
        save_model: Saves the trained model to the specified path.
        train: Trains the DeepSign model using the loaded dataset.
    """
    
    def __init__(self):
        logging.info("DeepSignTrainer initialized.")
        print(tf.test.gpu_device_name()) 

        # Get the train and val dataset
        try:
            self.train_data, self.val_data = self._get_dataset()
        except Exception as e:
            logging.error(f"Error loading datasets: {e}")
            raise
        else:
            logging.info(f"Datasets loaded.")

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
            downloaded_data = hf_hub_download(
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
        """
        Trains the DeepSign model using the loaded dataset.

        Raises:
            Exception: If an error occurs during training.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Cannot proceed with training.")
        if self.train_data is None:
            raise ValueError("Training Dataset is not loaded. Cannot proceed with training.")
        if self.val_data is None:
            raise ValueError("Validation Dataset is not loaded. Cannot proceed with training.")
        
        try:
            logging.info("Starting training...")
            self.model.fit(
                            self.train_data,
                            validation_data=self.val_data,
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
            self._save_model()
            logging.info("Model saved successfully after training.")
        

    def _save_model(self) -> None:
        """
        Saves the trained model to the specified path.

        Raises:
            Exception: If an error occurs during saving.
        """
        try:
            self.model.export(CONFIG['model']['save_path'])
            logging.info(f"Model saved successfully at {CONFIG['model']['save_path']}.")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise




if __name__ == "__main__":
    trainer = DeepSignTrainer()
    # Start training
    trainer.train()
    # Save the trained model
    #trainer.save_model()

