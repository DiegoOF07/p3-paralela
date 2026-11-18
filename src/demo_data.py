"""Invented results with the real file formats, to test plots.py before the official runs.

    python -m src.plots --demo          # writes everything to results/demo/ (git-ignored)

Numbers are made up (loosely shaped after the first E0/E1 runs and the synthetic
E3-E6 timings); they must never reach the report. results/runs.csv is never touched.
"""
import csv
from pathlib import Path

import numpy as np

from src import config
from src.analysis import COLUMNS

MACHINE = "demo"
N_TRAIN, N_PREP = 60900, 73950
N_CORES = 12

# (etapa, lote, device, epoch seconds, GPU % while training, VRAM MB, busy CPU %)
STAGES = [
    ("E0", 64, "cpu", 396.0, 2, 760, 85),
    ("E1", 64, "gpu", 55.6, 49, 1550, 14),
    ("E2", 64, "cpu", 300.0, 2, 760, 88),
    ("E3", 64, "gpu", 41.0, 70, 1900, 10),
    ("E4", 256, "gpu", 28.0, 85, 3200, 8),
    ("E4", 512, "gpu", 27.0, 88, 5100, 8),
    ("E5", 256, "gpu", 29.0, 80, 2600, 8),
    ("E6", 256, "gpu", 8.0, 93, 2400, 9),
]
IDLE_GPU = 30  # the Windows desktop alone shows ~30% in nvidia-smi


def _write(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _monitor(path, rng, seconds, gpu, vram, cpu, idle=False):
    header = ["t", "gpu_util", "vram_mb", "cpu_total"] + [f"cpu{i}" for i in range(N_CORES)]
    rows = []
    for i in range(int(seconds)):
        t = round(1.0 + i, 1)
        warm = (not idle) and t < 5
        g = IDLE_GPU if (idle or warm) else gpu
        cores = np.clip(rng.normal(3 if idle else cpu, 6, N_CORES), 0, 100).round(1)
        if not idle and cpu < 30:  # GPU stages: one feeder core does most of the host work
            cores[rng.integers(N_CORES)] = 95.0
        row = {"t": t, "gpu_util": int(np.clip(rng.normal(g, 5), 0, 100)),
               "vram_mb": int(760 if (idle or warm) else vram + rng.normal(0, 15)),
               "cpu_total": round(float(cores.mean()), 1)}
        row.update({f"cpu{j}": c for j, c in enumerate(cores)})
        rows.append(row)
    _write(path, rows, header)


def write_demo(out_dir):
    out_dir = Path(out_dir)
    evidence = out_dir / "evidence"
    rng = np.random.default_rng(config.SEED)
    runs = []

    for rep in (1, 2, 3):
        for etapa, secs in (("secuencial", 210.0), ("paralelo", 30.0)):
            s = secs * rng.normal(1, 0.02)
            runs.append({"fase": "prep", "etapa": etapa, "maquina": MACHINE, "dispositivo": "cpu",
                         "repeticion": rep, "segundos": round(s, 3), "imagenes_por_s": round(N_PREP / s, 1)})

    for etapa, lote, dev, secs, gpu, vram, cpu in STAGES:
        reps = config.REPS_CPU if dev == "cpu" else config.REPS_GPU
        _monitor(evidence / f"idle_{etapa}_b{lote}.csv", rng, config.IDLE_SECONDS, 0, 0, 0, idle=True)
        for rep in range(1, reps + 1):
            common = {"etapa": etapa, "maquina": MACHINE, "dispositivo": dev, "lote": lote, "repeticion": rep}
            if dev == "gpu":
                bytes_per = 4 if etapa == "E1" else 1
                t = lote * 64 * 64 * 3 * bytes_per / 4.0e9 * rng.normal(1, 0.08)
                runs.append({**common, "fase": "transfer", "segundos": round(t, 6),
                             "imagenes_por_s": round(lote / t, 1)})
            for epoca in range(1, config.EPOCHS + 1):
                s = secs * (1.12 if epoca == 1 else 1.0) * rng.normal(1, 0.015)
                acc = round(float(np.clip(rng.normal(0.97, 0.01), 0, 1)), 4) if epoca == config.EPOCHS else ""
                runs.append({**common, "fase": "train", "epoca": epoca, "segundos": round(s, 3),
                             "imagenes_por_s": round(N_TRAIN / s, 1), "val_acc": acc})
            # Shorter logs than a real run: enough samples for the figures.
            _monitor(evidence / f"{etapa}_b{lote}_r{rep}.csv", rng, min(secs * 3, 120), gpu, vram, cpu)

    diag = [{"fase": "diag", "etapa": e, "maquina": MACHINE, "dispositivo": "gpu", "lote": 64,
             "repeticion": 1, "segundos": s, "imagenes_por_s": round(N_TRAIN / s, 1)}
            for e, s in (("datos_generator", 52.0), ("datos_arrays", 1.5), ("matematica", 38.0))]

    kernel = []
    for rep in range(1, 6):
        for etapa, dev, ms in (("cpu_loop", "cpu", 95000.0), ("cpu_numpy", "cpu", 420.0),
                               ("h2d", "gpu", 0.30), ("gpu_naive", "gpu", 2.6),
                               ("gpu_shared", "gpu", 1.4), ("d2h", "gpu", 6.2)):
            kernel.append({"fase": "kernel", "etapa": etapa, "maquina": MACHINE, "dispositivo": dev,
                           "lote": 64, "repeticion": rep,
                           "segundos": round(ms / 1e3 * rng.normal(1, 0.03), 7)})

    paths = {"dir": out_dir, "evidence": evidence, "runs": out_dir / "runs.csv",
             "diag": out_dir / "diagnose.csv", "kernel": out_dir / "numba_conv.csv"}
    _write(paths["runs"], runs, COLUMNS)
    _write(paths["diag"], diag, COLUMNS)
    _write(paths["kernel"], kernel, COLUMNS)
    return paths


if __name__ == "__main__":
    print(write_demo(config.RESULTS_DIR / "demo"))
