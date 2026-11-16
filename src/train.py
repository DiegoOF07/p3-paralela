"""One timed training run of a single stage.

    python -m src.train --stage E0 --rep 1
"""
import argparse
import os

STAGES = {
    "E0": {"device": "cpu", "data": "generator"},
    "E1": {"device": "gpu", "data": "generator"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--rep", type=int, default=1)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
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


def main():
    args = parse_args()
    stage = STAGES[args.stage]
    setup_device(stage["device"])

    import contextlib

    import keras

    from src import config, model, pipelines
    from src.monitor import ResourceMonitor
    from src.timing import EpochTimer, append_rows

    epochs = args.epochs or config.EPOCHS
    batch = args.batch or stage.get("batch", config.BATCH_SIZE)
    keras.utils.set_random_seed(config.SEED)

    splits = pipelines.load_splits()
    train_data = pipelines.make_generator(splits["train"], batch, shuffle=True)
    val_data = pipelines.make_generator(splits["val"], batch, shuffle=False)
    n_train = len(splits["train"])

    net = model.compile_model(model.build_cnn_medium())
    timer = EpochTimer()

    tag = f"{args.stage}_b{batch}_r{args.rep}"
    if args.no_save:
        tag = f"test_{tag}"
    monitor = contextlib.nullcontext() if args.no_monitor else ResourceMonitor(tag)
    with monitor:
        net.fit(train_data, epochs=epochs, validation_data=val_data,
                callbacks=[timer], verbose=2)

    rows = []
    for e in timer.epochs:
        print(f"epoch {e['epoca']}: train {e['segundos']:.1f}s, "
              f"val {e['val_segundos']:.1f}s, val_acc {e['val_acc']:.4f}")
        rows.append({
            "fase": "train", "etapa": args.stage, "maquina": config.MACHINE,
            "dispositivo": stage["device"], "lote": batch, "repeticion": args.rep,
            "epoca": e["epoca"], "segundos": round(e["segundos"], 3),
            "imagenes_por_s": round(n_train / e["segundos"], 1),
            "val_acc": round(e["val_acc"], 4),
        })
    if not args.no_save:
        append_rows(rows)
        print(f"{len(rows)} rows written to {config.RUNS_CSV}")


if __name__ == "__main__":
    main()
