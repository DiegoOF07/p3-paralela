# p3-paralela

Training a CNN on the ASL Alphabet dataset (87,000 images, 29 classes): a comparison between the original sequential pipeline and an optimized pipeline on the GPU (CUDA).

## Structure

```text
original/   original lab notebooks (sequential baseline)
src/        project code
results/    run times (runs.csv) and plots
evidence/   GPU, VRAM and per-core CPU usage logs
report/     final report
```

## Setup (WSL2 + NVIDIA GPU)

TensorFlow dropped GPU support on native Windows after 2.10, so this has to run in WSL2 or Linux.

```bash
uv venv -p python3.12 ~/venvs/p3-cuda
VIRTUAL_ENV=~/venvs/p3-cuda uv pip install -r requirements.txt
```

The TensorFlow 2.21 wheel does not find `libcusolver.so.11` on its own and silently ignores the GPU. Add this line to the end of `~/venvs/p3-cuda/bin/activate`:

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cusolver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Check:

```bash
source ~/venvs/p3-cuda/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Data

Copy the dataset to the Linux filesystem (reading from `/mnt/c` is very slow):

```text
~/data/asl/
  asl_alphabet_train/asl_alphabet_train/<class>/*.jpg
  splits.pkl
```

Use `ASL_DATA_DIR` for a different location.

## Running

```bash
python -m src.train --stage E0 --rep 1   # CPU, original pipeline
python -m src.train --stage E1 --rep 1   # GPU, original pipeline
```

Each run appends one row per epoch to `results/runs.csv` and logs resource usage to `evidence/<stage>_b<batch>_r<rep>.csv`. `--epochs 1 --no-save` is for quick tests.

Columns in `runs.csv`:

```text
fase, etapa, maquina, dispositivo, lote, repeticion, epoca, segundos, imagenes_por_s, val_acc
```

`segundos` only counts the training part of the epoch, without validation.
