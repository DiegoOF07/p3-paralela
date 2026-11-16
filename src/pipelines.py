import pickle
from pathlib import PureWindowsPath

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src import config


def load_splits():
    """Loads the lab partition, rebuilding its Windows paths for the local dataset."""
    with open(config.SPLITS_PKL, "rb") as f:
        splits = pickle.load(f)

    out = {}
    for name in ("train", "val", "test"):
        df = splits[name].copy()
        df["path"] = [
            str(config.TRAIN_DIR / cls / PureWindowsPath(p).name)
            for p, cls in zip(df["path"], df["clase"])
        ]
        out[name] = df
    return out


def make_generator(df, batch_size=config.BATCH_SIZE, shuffle=True):
    """Original lab pipeline: decodes and resizes every JPEG on each epoch."""
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col="path",
        y_col="clase",
        target_size=config.TARGET_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=config.SEED,
        interpolation="bicubic",
    )


def load_arrays(split):
    """Loads prepared uint8 images (N, 64, 64, 3) and integer labels (N,)."""
    X = np.load(config.NPY_DIR / f"X_{split}.npy")
    y = np.load(config.NPY_DIR / f"y_{split}.npy")
    return X, y


def synthetic_arrays(n, seed=config.SEED):
    """Random arrays with the real shapes, to test stages before the .npy files exist."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 256, size=(n, *config.TARGET_SIZE, 3), dtype=np.uint8)
    y = rng.integers(0, config.N_CLASSES, size=n, dtype=np.int32)
    return X, y


def make_array_dataset(X, y, batch_size=config.BATCH_SIZE, shuffle=True):
    """tf.data pipeline over in-memory arrays; batches stay uint8 until the model."""
    # Pin to CPU so the CPU-to-GPU copy happens per batch, as in a real input pipeline.
    with tf.device("/CPU:0"):
        X = tf.constant(X)
        y = tf.one_hot(y, config.N_CLASSES)

    # Shuffling indices avoids a second full copy of the images in a shuffle buffer.
    idx = tf.data.Dataset.range(len(X))
    if shuffle:
        idx = idx.shuffle(len(X), seed=config.SEED, reshuffle_each_iteration=True)
    return (
        idx.batch(batch_size)
        .map(lambda i: (tf.gather(X, i), tf.gather(y, i)), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
