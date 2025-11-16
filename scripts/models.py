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
    

    def build_encoder(self, input_shape: tuple[int, int, int], embedding_dim:int = 128) -> tf.keras.Model:
        """
        Builds a lightweight encoder model for feature extraction from input images.
        
        Args:
            input_shape (Tuple[int, int, int]): Shape of the input images (height, width, channels).
            embedding_dim (int): Dimension of the output embeddings. Default is 128.
            
        Returns:
            model (tf.keras.Model): Compiled Keras model for the encoder.
        """
        logging.info("Building encoder model.")

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

    
    def build_and_compile_model(self, input_shape:Tuple[int,int,int], embedding_dim:int=128, head_units:int=128, dropout:float=0.3) -> tf.keras.Model:
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

        # NEW: Define the shape and type of the *raw* inputs
        RAW_INPUT_SHAPE = (None, None, 1)  # (H, W, 1) - accepts any size
        RAW_INPUT_DTYPE = tf.uint8
        
        # NEW: Get target dimensions from the 'input_shape' argument
        TARGET_HEIGHT = input_shape[0]
        TARGET_WIDTH = input_shape[1]

        # This is what the Lambda layer will output (channels-last)
        FINAL_ITEM_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, 1)

        # --- NEW: Build the shared "base_model" that includes preprocessing ---
        
        # 1. Define the raw input layer
        raw_input = tf.keras.layers.Input(shape=RAW_INPUT_SHAPE, dtype=RAW_INPUT_DTYPE, name="raw_input")
        
        # 2. Add Resizing layer (using the helper function)
        resized = tf.keras.layers.Lambda(
            lambda x: self._resize_with_pad_fn(x, TARGET_HEIGHT, TARGET_WIDTH),
            name="resize_with_pad",
            output_shape=FINAL_ITEM_SHAPE
        )(raw_input)
        
        # 3. Add Normalization layer (scales uint8 [0, 255] -> float32 [0, 1])
        rescaled = tf.keras.layers.Rescaling(scale=1./255.0, name="rescaling")(resized)
        
        # 4. Get your *existing* encoder
        #    It expects the preprocessed shape, which 'rescaled' now has.
        encoder = self.build_encoder(input_shape, embedding_dim=embedding_dim)

        # 5. Connect the preprocessed tensor to the encoder
        embedding = encoder(rescaled)
        
        # 6. Create the final 'base_model'
        base_model = tf.keras.Model(inputs=raw_input, outputs=embedding, name="preprocessing_plus_encoder")
        
        # --- END of new base_model ---

        
        # --- Build the Siamese model using the *new* base_model ---
        
        # NEW: The inputs now accept raw images
        input_a = tf.keras.layers.Input(shape=RAW_INPUT_SHAPE, dtype=RAW_INPUT_DTYPE, name="img_a")
        input_b = tf.keras.layers.Input(shape=RAW_INPUT_SHAPE, dtype=RAW_INPUT_DTYPE, name="img_b")

        # NEW: Use the 'base_model' (with preprocessing) for both inputs
        emb_a = base_model(input_a)
        emb_b = base_model(input_b)

        # The rest of your model logic is UNCHANGED
        diff = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([emb_a, emb_b])

        x = tf.keras.layers.Dense(head_units, activation="relu")(diff)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(head_units//2, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid", name="is_same")(x)

        model = tf.keras.Model(inputs=[input_a, input_b], outputs=out, name="DeepSign")
        
        logging.info("Model built successfully with integrated preprocessing.")
        logging.info("Model inputs: [(None, None, 1, uint8), (None, None, 1, uint8)]")
        logging.info("Model output: (None, 1, float32)")

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
