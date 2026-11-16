import csv
import time

import keras

from src import config

COLUMNS = [
    "fase", "etapa", "maquina", "dispositivo", "lote", "repeticion",
    "epoca", "segundos", "imagenes_por_s", "val_acc",
]


class EpochTimer(keras.callbacks.Callback):
    """Times the training part of each epoch, excluding validation."""

    def __init__(self):
        super().__init__()
        self.epochs = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()
        self._train_end = None

    def on_test_begin(self, logs=None):
        # Fired by fit() right before validation starts.
        if self._train_end is None:
            self._train_end = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        end = time.perf_counter()
        train_end = self._train_end or end
        logs = logs or {}
        self.epochs.append({
            "epoca": epoch + 1,
            "segundos": train_end - self._start,
            "val_segundos": end - train_end,
            "val_acc": logs.get("val_accuracy"),
        })


def append_rows(rows, path=config.RUNS_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLUMNS})
