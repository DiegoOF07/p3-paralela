"""One timed training run of a single stage.

    python -m src.train --stage E3 --rep 1
"""
import argparse
import os

from src import config

# Each stage inherits the previous one and changes a single thing.
# XLA is disabled explicitly until E6: Keras 3 enables it by default on GPU.
STAGES = {
    "E0": {"device": "cpu", "data": "generator", "batch": config.BATCH_SIZE},
    "E1": {"device": "gpu", "data": "generator", "batch": config.BATCH_SIZE},
    "E2": {"device": "cpu", "data": "arrays", "batch": config.BATCH_SIZE},
    "E3": {"device": "gpu", "data": "arrays", "batch": config.BATCH_SIZE},
    "E4": {"device": "gpu", "data": "arrays", "batch": config.LARGE_BATCHES[0]},
    "E5": {"device": "gpu", "data": "arrays", "batch": config.TUNED_BATCH, "mixed": True},
    "E6": {"device": "gpu", "data": "arrays", "batch": config.TUNED_BATCH, "mixed": True, "xla": True},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--rep", type=int, default=1)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--synthetic", action="store_true", help="random arrays instead of the .npy files")
    p.add_argument("--no-save", action="store_true", help="do not write to runs.csv")
    p.add_argument("--no-monitor", action="store_true")
    return p.parse_args()


def setup_device(device):
    # Must run before TensorFlow is imported.
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu":
        if not gpus:
            raise SystemExit("GPU stage requested but TensorFlow sees no GPU.")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    elif gpus:
        raise SystemExit("CPU stage requested but a GPU is still visible.")
    print(f"TensorFlow {tf.__version__} | GPUs: {gpus or 'none'}")


def load_data(stage, batch, synthetic):
    from src import pipelines

    if stage["data"] == "generator":
        splits = pipelines.load_splits()
        train = pipelines.make_generator(splits["train"], batch, shuffle=True)
        val = pipelines.make_generator(splits["val"], batch, shuffle=False)
        return train, val, len(splits["train"])

    if synthetic:
        X_tr, y_tr = pipelines.synthetic_arrays(60900)
        X_va, y_va = pipelines.synthetic_arrays(13050, seed=config.SEED + 1)
    else:
        X_tr, y_tr = pipelines.load_arrays("train")
        X_va, y_va = pipelines.load_arrays("val")
    train = pipelines.make_array_dataset(X_tr, y_tr, batch, shuffle=True)
    val = pipelines.make_array_dataset(X_va, y_va, batch, shuffle=False)
    return train, val, len(X_tr)


def measure_transfer(data, n=50):
    """Mean seconds to copy one input batch from host memory to the GPU."""
    import time

    import tensorflow as tf

    x = next(iter(data))[0]
    with tf.device("/CPU:0"):
        x = tf.identity(tf.convert_to_tensor(x))

    times = []
    for i in range(n + 5):
        start = time.perf_counter()
        with tf.device("/GPU:0"):
            g = tf.identity(x)
        g[0, 0, 0, 0].numpy()  # waits for the async copy to finish
        if i >= 5:  # first copies include allocator warm-up
            times.append(time.perf_counter() - start)
    seconds = sum(times) / len(times)
    mb = x.numpy().nbytes / 1e6
    print(f"transfer: {mb:.1f} MB per batch ({x.dtype.name}), {seconds * 1e3:.2f} ms, "
          f"{mb / seconds / 1e3:.2f} GB/s")
    return seconds


def main():
    args = parse_args()
    stage = STAGES[args.stage]
    setup_device(stage["device"])

    import contextlib

    import keras

    from src import model
    from src.monitor import ResourceMonitor
    from src.timing import EpochTimer, append_rows

    batch = args.batch or stage["batch"]
    keras.utils.set_random_seed(config.SEED)
    if stage.get("mixed"):
        keras.mixed_precision.set_global_policy("mixed_float16")

    train_data, val_data, n_train = load_data(stage, batch, args.synthetic)

    net = model.build_cnn_medium(rescale=stage["data"] == "arrays")
    model.compile_model(net, jit_compile=stage.get("xla", False))
    print(f"stage {args.stage}: batch {batch}, policy {keras.mixed_precision.global_policy().name}, "
          f"jit_compile {net.jit_compile}, optimizer {type(net.optimizer).__name__}")

    common = {"etapa": args.stage, "maquina": config.MACHINE, "dispositivo": stage["device"],
              "lote": batch, "repeticion": args.rep}
    rows = []
    if stage["device"] == "gpu":
        seconds = measure_transfer(train_data)
        rows.append({**common, "fase": "transfer", "segundos": round(seconds, 6),
                     "imagenes_por_s": round(batch / seconds, 1)})

    timer = EpochTimer()
    tag = f"{args.stage}_b{batch}_r{args.rep}"
    if args.no_save:
        tag = f"test_{tag}"
    monitor = contextlib.nullcontext() if args.no_monitor else ResourceMonitor(tag)
    with monitor:
        # Validate only after the last epoch: accuracy is recorded, timing is unaffected.
        net.fit(train_data, epochs=args.epochs, validation_data=val_data,
                validation_freq=args.epochs, shuffle=False,  # pipelines shuffle themselves
                callbacks=[timer], verbose=2)

    for e in timer.epochs:
        val_acc = e["val_acc"]
        print(f"epoch {e['epoca']}: train {e['segundos']:.1f}s, val {e['val_segundos']:.1f}s"
              + (f", val_acc {val_acc:.4f}" if val_acc is not None else ""))
        rows.append({
            **common, "fase": "train", "epoca": e["epoca"],
            "segundos": round(e["segundos"], 3),
            "imagenes_por_s": round(n_train / e["segundos"], 1),
            "val_acc": "" if val_acc is None else round(val_acc, 4),
        })
    if not args.no_save:
        append_rows(rows)
        print(f"{len(rows)} rows written to {config.RUNS_CSV}")


if __name__ == "__main__":
    main()
