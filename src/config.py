import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Keep the dataset on the Linux filesystem; reading it from /mnt/c is very slow.
DATA_DIR = Path(os.environ.get("ASL_DATA_DIR", Path.home() / "data" / "asl"))
TRAIN_DIR = DATA_DIR / "asl_alphabet_train" / "asl_alphabet_train"
SPLITS_PKL = DATA_DIR / "splits.pkl"
NPY_DIR = DATA_DIR / "npy"

RESULTS_DIR = ROOT / "results"
EVIDENCE_DIR = ROOT / "evidence"
RUNS_CSV = RESULTS_DIR / "runs.csv"

MACHINE = os.environ.get("MACHINE_NAME", "wsl2-rtx3050")

SEED = 42
TARGET_SIZE = (64, 64)
N_CLASSES = 29
BATCH_SIZE = 64
LEARNING_RATE = 5e-4

EPOCHS = 3
TIMED_EPOCHS = (2, 3)  # epoch 1 includes graph tracing and warm-up
REPS_CPU = 3
REPS_GPU = 5
