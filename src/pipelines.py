import pickle
from pathlib import PureWindowsPath

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
