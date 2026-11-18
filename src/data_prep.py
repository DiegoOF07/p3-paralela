"""Phase 1: decodes every JPEG once and saves the uint8 arrays used from E2 onward.

    python -m src.data_prep                       # sequential + parallel, saves .npy and timings
    python -m src.data_prep --mode paralelo       # only the parallel version
    python -m src.data_prep --limit 2000 --no-save  # quick test

Output in $ASL_DATA_DIR/npy/ (default ~/data/asl/npy/):
    X_train.npy (60900, 64, 64, 3) uint8   y_train.npy (60900,) int32
    X_val.npy   (13050, 64, 64, 3) uint8   y_val.npy   (13050,) int32
"""
import argparse
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from src import config

SPLITS = ("train", "val")


def preprocess_image(path, size=config.TARGET_SIZE):
    """Same as the lab (RGB, bicubic resize) but kept as uint8: the model rescales."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def class_order(splits):
    """Alphabetical class order, checked against label_map.pkl when it exists."""
    classes = sorted(splits["train"]["clase"].unique())
    label_map_pkl = config.DATA_DIR / "label_map.pkl"
    if label_map_pkl.exists():
        with open(label_map_pkl, "rb") as f:
            label_map = pickle.load(f)
        # label_map may be {class: idx} or {idx: class}.
        by_class = label_map if isinstance(next(iter(label_map)), str) else {v: k for k, v in label_map.items()}
        expected = sorted(by_class, key=by_class.get)
        if expected != classes:
            raise SystemExit(f"Class order differs from label_map.pkl: {expected} vs {classes}")
    if len(classes) != config.N_CLASSES:
        raise SystemExit(f"Expected {config.N_CLASSES} classes, found {len(classes)}")
    return {cls: i for i, cls in enumerate(classes)}


def load_sequential(paths):
    out = np.empty((len(paths), *config.TARGET_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        out[i] = preprocess_image(p)
    return out


def load_parallel(paths, workers):
    out = np.empty((len(paths), *config.TARGET_SIZE, 3), dtype=np.uint8)
    chunksize = max(1, len(paths) // (workers * 16))
    with Pool(workers) as pool:
        for i, arr in enumerate(pool.imap(preprocess_image, paths, chunksize=chunksize)):
            out[i] = arr
    return out


def run(mode, data, workers):
    """Decodes all splits with one mode; returns arrays and elapsed seconds."""
    start = time.perf_counter()
    arrays = {}
    for split, (paths, _) in data.items():
        arrays[split] = load_sequential(paths) if mode == "secuencial" else load_parallel(paths, workers)
    return arrays, time.perf_counter() - start


def warm_cache(data):
    """Reads every file once so the timed runs do not depend on a cold disk cache."""
    start = time.perf_counter()
    for paths, _ in data.values():
        for p in paths:
            with open(p, "rb") as f:
                f.read()
    print(f"warm-up (discarded): {time.perf_counter() - start:.1f} s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["secuencial", "paralelo", "ambos"], default="ambos")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--reps", type=int, default=1, help="timed repetitions per mode")
    p.add_argument("--limit", type=int, default=None, help="use only the first N images per split")
    p.add_argument("--no-save", action="store_true", help="write neither .npy files nor runs.csv rows")
    return p.parse_args()


def main():
    args = parse_args()
    from src.pipelines import load_splits
    from src.timing import append_rows

    splits = load_splits()
    label_of = class_order(splits)
    data = {}
    for split in SPLITS:
        df = splits[split] if args.limit is None else splits[split].head(args.limit)
        paths = list(df["path"])
        labels = np.array([label_of[c] for c in df["clase"]], dtype=np.int32)
        data[split] = (paths, labels)
    n_images = sum(len(p) for p, _ in data.values())
    print(f"{n_images} images ({', '.join(f'{s}: {len(data[s][0])}' for s in SPLITS)}), "
          f"{args.workers} workers")

    warm_cache(data)

    modes = ["secuencial", "paralelo"] if args.mode == "ambos" else [args.mode]
    results, rows = {}, []
    for mode in modes:
        for rep in range(1, args.reps + 1):
            arrays, seconds = run(mode, data, args.workers)
            results[mode] = arrays
            print(f"{mode} r{rep}: {seconds:.1f} s, {n_images / seconds:.1f} img/s")
            rows.append({
                "fase": "prep", "etapa": mode, "maquina": config.MACHINE, "dispositivo": "cpu",
                "lote": "", "repeticion": rep, "epoca": "",
                "segundos": round(seconds, 3), "imagenes_por_s": round(n_images / seconds, 1),
            })

    if len(results) == 2:
        same = all(np.array_equal(results["secuencial"][s], results["paralelo"][s]) for s in SPLITS)
        print(f"sequential and parallel outputs identical: {same}")
        if not same:
            raise SystemExit("Outputs differ; not saving.")

    if args.no_save or args.limit is not None:
        print("--no-save or --limit: nothing written")
        return

    arrays = results[modes[-1]]
    config.NPY_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        np.save(config.NPY_DIR / f"X_{split}.npy", arrays[split])
        np.save(config.NPY_DIR / f"y_{split}.npy", data[split][1])
        print(f"saved X_{split}.npy {arrays[split].shape} {arrays[split].dtype}, "
              f"y_{split}.npy {data[split][1].shape} {data[split][1].dtype}")
    append_rows(rows)
    print(f"{len(rows)} rows appended to {config.RUNS_CSV}")


if __name__ == "__main__":
    main()
