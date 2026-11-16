import keras
from keras import layers, regularizers

from src import config


def build_cnn_medium(input_shape=(64, 64, 3), n_classes=29, l2_reg=1e-4, rescale=False):
    """CNN Medium from the lab. rescale=True normalizes uint8 input on the device."""
    inputs = keras.Input(shape=input_shape, name="input")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs) if rescale else inputs

    # Bloque 1
    x = layers.Conv2D(32,  (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloque 2
    x = layers.Conv2D(64,  (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloque 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloque 4 + GlobalAvgPool
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Clasificador con L2
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.5)(x)
    # float32 output keeps softmax stable under mixed precision; no-op otherwise.
    outputs = layers.Dense(n_classes, activation="softmax", dtype="float32", name="output")(x)

    return keras.Model(inputs, outputs, name="CNN_Medium")


def compile_model(model, jit_compile=False):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=jit_compile,
    )
    return model
