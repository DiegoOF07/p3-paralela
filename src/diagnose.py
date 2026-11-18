"""Bottleneck diagnosis: is an epoch limited by the input pipeline or by the GPU math?

    python -m src.diagnose                 # both tests, GPU required
    python -m src.diagnose --only datos
    python -m src.diagnose --synthetic     # random arrays instead of the .npy files

"solo datos": iterates one epoch of each input pipeline and copies every batch to the
GPU, with no model. "solo matematica": trains the model on a batch that already lives
in GPU memory, so no decoding or host-to-device copy is involved.
If datos is much slower than matematica, the pipeline is the bottleneck.

Rows go to results/diagnose.csv (same columns as runs.csv, fase="diag"),
so they do not mix with the stage runs.
"""
import argparse
import time

from src import config

DIAG_CSV = config.RESULTS_DIR / "diagnose.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["datos", "matematica"], default=None)
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--synthetic", action="store_true", help="random arrays instead of the .npy files")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def time_pipeline(batches, n_batches):
    """Seconds to pull n_batches and copy each one to the GPU."""
    import tensorflow as tf

    it = iter(batches)
    next(it)  # first batch includes pipeline start-up
    start = time.perf_counter()
    n_images = 0
    for _ in range(n_batches - 1):
        x, _ = next(it)
        with tf.device("/GPU:0"):
            g = tf.identity(x)
        g[0, 0, 0, 0].numpy()  # waits for the copy
        n_images += x.shape[0]
    return time.perf_counter() - start, n_images


def solo_datos(batch, synthetic):
    from src import pipelines

    out = {}
    splits = pipelines.load_splits()
    gen = pipelines.make_generator(splits["train"], batch, shuffle=True)
    out["generator"] = time_pipeline(gen, len(gen))

    if synthetic:
        X, y = pipelines.synthetic_arrays(60900)
    else:
        X, y = pipelines.load_arrays("train")
    ds = pipelines.make_array_dataset(X, y, batch, shuffle=True)
    out["arrays"] = time_pipeline(ds, -(-len(X) // batch))
    return out


def solo_matematica(batch, n_train=60900, warmup=10):
    import numpy as np
    import tensorflow as tf

    from src import model

    net = model.build_cnn_medium(rescale=True)
    # Keras 3 turns XLA on by default with a GPU; keep it off to match E1-E5.
    model.compile_model(net, jit_compile=False)

    rng = np.random.default_rng(config.SEED)
    with tf.device("/GPU:0"):
        x = tf.constant(rng.integers(0, 256, size=(batch, *config.TARGET_SIZE, 3), dtype=np.uint8))
        y = tf.one_hot(tf.constant(rng.integers(0, config.N_CLASSES, size=batch)), config.N_CLASSES)

    steps = n_train // batch
    for _ in range(warmup):
        net.train_on_batch(x, y)  # graph tracing and cuDNN autotuning
    start = time.perf_counter()
    for _ in range(steps):
        net.train_on_batch(x, y)  # returns Python floats, so each step is synchronized
    return time.perf_counter() - start, steps * batch


def main():
    args = parse_args()
    from src.train import setup_device

    setup_device("gpu")

    import keras

    from src.timing import append_rows

    keras.utils.set_random_seed(config.SEED)
    results = {}
    if args.only in (None, "datos"):
        for name, r in solo_datos(args.batch, args.synthetic).items():
            results[f"datos_{name}"] = r
    if args.only in (None, "matematica"):
        results["matematica"] = solo_matematica(args.batch)

    rows = []
    print(f"\n{'test':<18}{'s/epoch':>10}{'img/s':>12}")
    for name, (seconds, n) in results.items():
        # Normalize to a full 60900-image epoch for a direct comparison.
        per_epoch = seconds / n * 60900
        print(f"{name:<18}{per_epoch:>10.1f}{n / seconds:>12.1f}")
        rows.append({"fase": "diag", "etapa": name, "maquina": config.MACHINE, "dispositivo": "gpu",
                     "lote": args.batch, "repeticion": 1, "segundos": round(per_epoch, 3),
                     "imagenes_por_s": round(n / seconds, 1)})

    if not args.no_save and not args.synthetic:
        append_rows(rows, path=DIAG_CSV)
        print(f"\n{len(rows)} rows appended to {DIAG_CSV}")


if __name__ == "__main__":
    main()
