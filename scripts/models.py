###################### Imports #######################

import logging
import tensorflow as tf
from typing import Tuple

from utils import load_config

######################################################


############## Global Variables ##############

CONFIG = load_config()
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='logs/preprocess.log',
                   filemode='w')

##############################################



class DeepSign:
    def __init__(self):
        LOGGER.info("DeepSign model initialized.")

    
    def get_model_summary(self) -> None:
        """
        Saves the model summary to a text file.
        """
        with open('../artifacts/model_summary.txt', 'w', encoding="utf-8") as f:
            model = self.build_encoder(
                input_shape=(CONFIG['data']['img_height'], CONFIG['data']['img_width'], 1),
                embedding_dim=CONFIG['model']['embedding_dim']
            )
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print("Model summary saved to 'artifacts/model_summary.txt'")


    def conv_block(self, x: tf.Tensor, filters: int, kernel_size: int = 3, strides: int = 1, pool: bool = False) -> tf.Tensor:
        """
        Creates a convolutional block with Conv2D, BatchNormalization, ReLU activation, and optionally MaxPooling.
        
        Args:
            x (tf.Tensor): Input tensor to the convolutional block.
            filters (int): Number of filters for the Conv2D layer.
            kernel_size (int): Size of the convolution kernel. Default is 3.
            strides (int): Stride size for the convolution. Default is 1.
            pool (bool): Whether to include a MaxPooling layer. Default is False.
            
        Returns:
            x (tf.Tensor): Output tensor after applying the convolutional block.
        """
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        if pool:
            x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        return x
    

    def _resize_with_pad_fn(self, image: tf.Tensor, target_height: int, target_width: int) -> tf.Tensor:
        """
        Wraps tf.image.resize_with_pad for use in a Lambda layer.
        This function receives the entire 4D batch tensor.
        """
        # 'image' is already the 4D batch (Batch, H, W, C).
        # tf.image.resize_with_pad is vectorized and accepts a 4D batch.
        
        resized = tf.image.resize_with_pad(
            image, target_height, target_width, method='bilinear'
        )
        
        # 'resized' is now the 4D batch with the target shape (Batch, Target_H, Target_W, C)
        return resized
    

    def build_encoder(self, input_shape: tuple[int, int, int], embedding_dim:int = 128, dropout: float = 0.3) -> tf.keras.Model:
        """
        Builds a lightweight encoder model for feature extraction from input images.
        
        Args:
            input_shape (Tuple[int, int, int]): Shape of the input images (height, width, channels).
            embedding_dim (int): Dimension of the output embeddings. Default is 128.
            
        Returns:
            model (tf.keras.Model): Compiled Keras model for the encoder.
        """
        logging.info("Building encoder model.")

        # Define the raw input layer
        input = tf.keras.layers.Input(shape=input_shape, name="input_layer")

        # Block 1
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        # Block 2
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        # Block 3
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        # Flatten and Dense (The "Embedding" Vector)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(embedding_dim, activation='relu')(x)

        # Optional: Dropout to prevent overfitting
        x = tf.keras.layers.Dropout(dropout)(x)

        return tf.keras.models.Model(inputs=input, outputs=x, name="embedding_network")

    
    def build_and_compile_model(self, input_shape:Tuple[int,int,int], embedding_dim:int=128, head_units:int=128) -> tf.keras.Model:
        """
        Builds a Siamese network model with integrated preprocessing.
        
        Args:
            input_shape (Tuple[int, int, int]): Shape of the *preprocessed* images 
                                                (target_height, target_width, channels).
            embedding_dim (int): Dimension of the output embeddings. Default is 128.
            head_units (int): Number of units in the dense layers of the head. Default is 128.
            dropout (float): Dropout rate for regularization. Default is 0.3.
            
        Returns:
            model (tf.keras.Model): Compiled Keras model for the Siamese network.
        """
        logging.info("Building DeepSign model with preprocessing...")
        
        # Get the embedding network
        embedding_net = self.build_encoder(input_shape, embedding_dim)

        # Define the Two Inputs
        input_a = tf.keras.layers.Input(shape=input_shape, name="input_a")
        input_b = tf.keras.layers.Input(shape=input_shape, name="input_b")

        # Pass inputs through the embedding network
        feat_vec_a = embedding_net(input_a)
        feat_vec_b = embedding_net(input_b)

        # Calculate the Distance (L1 Distance)
        # |v1 - v2|
        distance = tf.keras.layers.Lambda(
                                            lambda tensors: tf.abs(tensors[0] - tensors[1]), 
                                            name="L1_distance"
                                        )([feat_vec_a, feat_vec_b])

        # Prediction Layer
        # Dense(1) with Sigmoid outputs a probability (0.0 to 1.0)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(distance)

        model = tf.keras.models.Model(inputs=[input_a, input_b], outputs=outputs, name="DeepSign_Network")
        
        logging.info("Model built successfully with integrated preprocessing.")

        # Compile the model
        model = self.compile(model)
        self.model = model
        return model
    

    def compile(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Compiles the DeepSign model with specified optimizer, loss function, and metrics.

        Args:
            model (tf.keras.Model): The Keras model to compile.
        """
        logging.info("Compiling the DeepSign model...")

        try:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['training']['learning_rate']),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                        tf.keras.metrics.AUC(name="auc")]
            )
        except Exception as e:
            logging.error("Error compiling the model: %s", e)
            raise
        else:
            logging.info("Model compiled successfully and is ready for training.")
            return model



if __name__ == "__main__":
    # Example usage
    img_height = CONFIG['data']['img_height']
    img_width = CONFIG['data']['img_width']
    embedding_dim = CONFIG['model']['embedding_dim']

    deep_sign = DeepSign()
    
    # Get model summary
    #deep_sign.get_model_summary()

    # Build and compile the model
    model = deep_sign.build_and_compile_model(input_shape=(img_height, img_width, 1), embedding_dim=embedding_dim)
