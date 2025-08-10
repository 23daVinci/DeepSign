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
        pass

    
    def get_model_summary(self) -> None:
        with open('../artifacts/model_summary.txt', 'w', encoding="utf-8") as f:
            model = self.build_model(
                input_shape=(CONFIG['data']['img_height'], CONFIG['data']['img_width'], 1),
                embedding_dim=CONFIG['model']['embedding_dim']
            )
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print("Model summary saved to 'artifacs/model_summary.txt'")

    
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
    

    def build_encoder(self, input_shape:Tuple[int,int,int], embedding_dim:int = 128) -> tf.keras.Model:
        """
        Builds a lightweight encoder model for feature extraction from input images.
        
        Args:
            input_shape (Tuple[int, int, int]): Shape of the input images (height, width, channels).
            embedding_dim (int): Dimension of the output embeddings. Default is 128.
            
        Returns:
            model (tf.keras.Model): Compiled Keras model for the encoder.
        """
        logging.info("Building encoder model with input shape: %s and embedding dimension: %d", input_shape, embedding_dim)

        inp = tf.keras.layers.Input(shape=input_shape, name="encoder_input")
        x = self.conv_block(inp, 32, kernel_size=3, pool=True)   # /2
        x = self.conv_block(x, 32, kernel_size=3, pool=False)
        x = self.conv_block(x, 64, kernel_size=3, pool=True)      # /4
        x = self.conv_block(x, 64, kernel_size=3, pool=False)
        x = self.conv_block(x, 128, kernel_size=3, pool=True)     # /8
        x = self.conv_block(x, 128, kernel_size=3, pool=False)
        # TODO: Optional extra block if input larger or more capacity needed
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(embedding_dim, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # L2-normalize embeddings (optional; helpful for distance metrics)
        x = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="embed_l2norm")(x)

        model = tf.keras.Model(inputs=inp, outputs=x, name="lightweight_encoder")
        return model

    
    def build_model(self, input_shape:Tuple[int,int,int], embedding_dim:int=128, head_units:int=128, dropout:float=0.3) -> tf.keras.Model:
        """
        Builds a Siamese network model for comparing pairs of images.
        
        Args:
            input_shape (Tuple[int, int, int]): Shape of the input images (height, width, channels).
            embedding_dim (int): Dimension of the output embeddings. Default is 128.
            head_units (int): Number of units in the dense layers of the head. Default is 128.
            dropout (float): Dropout rate for regularization. Default is 0.3.
            
        Returns:
            model (tf.keras.Model): Compiled Keras model for the Siamese network.
        """
        encoder = self.build_encoder(input_shape, embedding_dim=embedding_dim)

        input_a = tf.keras.layers.Input(shape=input_shape, name="img_a")
        input_b = tf.keras.layers.Input(shape=input_shape, name="img_b")

        emb_a = encoder(input_a)  # shape: (batch, embedding_dim)
        emb_b = encoder(input_b)

        # similarity features: absolute difference (L1). Optionally add elementwise multiply.
        diff = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([emb_a, emb_b])
        # Optionally augment features:
        # mult = tf.keras.layers.Multiply()([emb_a, emb_b])
        # features = tf.keras.layers.Concatenate()([diff, mult])

        x = tf.keras.layers.Dense(head_units, activation="relu")(diff)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(head_units//2, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid", name="is_same")(x)

        model = tf.keras.Model(inputs=[input_a, input_b], outputs=out, name="DeepSign")
        return model




if __name__ == "__main__":
    # Example usage
    img_height = CONFIG['data']['img_height']
    img_width = CONFIG['data']['img_width']
    embedding_dim = CONFIG['model']['embedding_dim']

    deep_sign = DeepSign()
    # Get model summary
    deep_sign.get_model_summary()
    
    # Initialize DataParser
    #data_parser = DataParser()
    # Example of how to use test_batches (assuming dataset is set)
    # data_parser.test_batches(num_batches=2)