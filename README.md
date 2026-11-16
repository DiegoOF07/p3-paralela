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

## Stages

Each stage inherits the previous one and changes a single thing. Every run trains 3 epochs; the reported time is the mean of epochs 2 and 3.

| Stage | Device | Change |
|---|---|---|
| E0 | CPU | Original lab code (`ImageDataGenerator`) |
| E1 | GPU | Same code with the GPU visible |
| E2 | CPU | Prepared `.npy` arrays with `tf.data` |
| E3 | GPU | E2 on the GPU |
| E4 | GPU | Batch 256 and 512 |
| E5 | GPU | Mixed precision (`mixed_float16`) |
| E6 | GPU | XLA compilation |

Keras 3 turns XLA on by default when a GPU is present, so E0–E5 disable it explicitly and only E6 enables it.

E2 onward reads `X_train.npy`, `y_train.npy`, `X_val.npy` and `y_val.npy` from `~/data/asl/npy/` (uint8 images of 64×64×3, integer labels following the alphabetical order of the classes).

## Running

```bash
python -m src.train --stage E3 --rep 1           # a single run
python -m src.bench --stages E0 E2               # CPU stages (slow, best overnight)
python -m src.bench --stages E1 E3 E4 E5 E6      # GPU stages
python -m src.bench --dry-run                    # print the run plan
```

- Each training run appends one row per epoch to `results/runs.csv` and logs resource usage to `evidence/<stage>_b<batch>_r<rep>.csv`.
- GPU runs also add a `transfer` row: the mean time to copy one batch from host memory to the GPU.
- Before each stage, `bench.py` records an idle baseline in `evidence/idle_<stage>_b<batch>.csv`.
- `bench.py` skips runs already in `runs.csv`, so an interrupted night can be resumed by running the same command again.
- `--synthetic` uses random arrays instead of the `.npy` files and saves nothing; `--epochs 1 --no-save` is for quick tests.
- Validation only runs after the last epoch, so it doesn't affect the timing.

For GPU measurements, close browsers and apps that use hardware acceleration; the Windows desktop alone already shows up in `nvidia-smi`.

Columns in `runs.csv`:

```text
fase, etapa, maquina, dispositivo, lote, repeticion, epoca, segundos, imagenes_por_s, val_acc
```

`segundos` only counts the training part of the epoch, without validation.
