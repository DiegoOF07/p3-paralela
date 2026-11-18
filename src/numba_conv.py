"""Hand-written CUDA kernel for the first convolution of CNN Medium (step 5, CUDA Python).

    python -m src.numba_conv                       # GPU required; appends to results/numba_conv.csv
    python -m src.numba_conv --synthetic           # random images instead of X_train.npy
    NUMBA_ENABLE_CUDASIM=1 python -m src.numba_conv --small --no-save   # correctness only, no GPU

The layer is conv1 of build_cnn_medium: 3x3 kernel, padding "same", 3 -> 32 channels, ReLU,
NHWC layout, on a batch of 64x64 images. TensorFlow runs this layer with cuDNN; here the
same math is written by hand to show how the work is split into threads, blocks and grids.

Versions that are timed (same input, same weights, results checked against each other):
  cpu_loop    the sequential algorithm: nested loops compiled with @njit, one CPU core
  cpu_numpy   vectorized NumPy (9 shifted tensordot products)
  gpu_naive   one thread per output pixel and filter; grid (W/16, H/16, N*K)
  gpu_shared  one thread per output pixel computing all K filters; the 18x18 input tile
              (16x16 + 1-pixel halo) and the weights are staged in shared memory; grid (W/16, H/16, N)
  h2d, d2h    host-to-device copy of the batch and device-to-host copy of the output
"""
import argparse
import csv
import math
import time

import numpy as np
from numba import config as numba_config
from numba import cuda, njit

from src import config

KERNEL_CSV = config.RESULTS_DIR / "numba_conv.csv"
COLUMNS = [
    "fase", "etapa", "maquina", "dispositivo", "lote", "repeticion",
    "epoca", "segundos", "imagenes_por_s", "val_acc",
]

KSIZE = 3
PAD = KSIZE // 2
TILE = 16                 # threads per block side: 16 x 16 = 256 threads
SH = TILE + 2 * PAD       # input tile with its halo
C_IN = 3                  # conv1 input channels (RGB)
K_OUT = 32                # conv1 filters
SIMULATOR = bool(numba_config.ENABLE_CUDASIM)


# --------------------------------------------------------------------------- CPU references

@njit
def conv_cpu_loop(x, w, b, out):
    """Sequential reference: the same arithmetic the kernels do, one output at a time."""
    n_img, h, wd, c_in = x.shape
    k_out = w.shape[3]
    for n in range(n_img):
        for i in range(h):
            for j in range(wd):
                for k in range(k_out):
                    acc = b[k]
                    for dy in range(KSIZE):
                        yy = i + dy - PAD
                        if yy < 0 or yy >= h:
                            continue
                        for dx in range(KSIZE):
                            xx = j + dx - PAD
                            if xx < 0 or xx >= wd:
                                continue
                            for c in range(c_in):
                                acc += x[n, yy, xx, c] * w[dy, dx, c, k]
                    out[n, i, j, k] = acc if acc > 0 else 0.0


def conv_numpy(x, w, b):
    """Vectorized reference: sum over the 9 kernel offsets of a channel contraction."""
    n_img, h, wd, _ = x.shape
    xp = np.pad(x, ((0, 0), (PAD, PAD), (PAD, PAD), (0, 0)))
    out = np.broadcast_to(b, (n_img, h, wd, w.shape[3])).astype(np.float32)
    for dy in range(KSIZE):
        for dx in range(KSIZE):
            out = out + np.tensordot(xp[:, dy:dy + h, dx:dx + wd, :], w[dy, dx], axes=([3], [0]))
    return np.maximum(out, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- CUDA kernels

@cuda.jit
def conv_naive(x, w, b, out):
    """Thread (tx, ty) of block (bx, by, bz) computes out[n, row, col, k], with bz = n * K + k."""
    col, row = cuda.grid(2)
    n_img, h, wd, c_in = x.shape
    k_out = w.shape[3]
    nk = cuda.blockIdx.z
    n = nk // k_out
    k = nk % k_out
    if row >= h or col >= wd or n >= n_img:
        return
    acc = b[k]
    for dy in range(KSIZE):
        yy = row + dy - PAD
        if 0 <= yy < h:
            for dx in range(KSIZE):
                xx = col + dx - PAD
                if 0 <= xx < wd:
                    for c in range(c_in):
                        acc += x[n, yy, xx, c] * w[dy, dx, c, k]
    out[n, row, col, k] = acc if acc > 0 else 0.0


@cuda.jit
def conv_shared(x, w, b, out):
    """Block (bx, by, n) covers a 16x16 output tile of image n for all K filters.

    The block first copies its 18x18xC input tile (with zero padding at the borders)
    and every weight into shared memory; each input value is then read 9*K times from
    on-chip memory instead of global memory.
    """
    tile = cuda.shared.array((SH, SH, C_IN), dtype=np.float32)
    ws = cuda.shared.array((KSIZE, KSIZE, C_IN, K_OUT), dtype=np.float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    n = cuda.blockIdx.z
    n_img, h, wd, _ = x.shape
    row0 = cuda.blockIdx.y * TILE - PAD
    col0 = cuda.blockIdx.x * TILE - PAD

    # Cooperative load of the tile + halo: 256 threads cover 18*18 positions in strides.
    for i in range(ty, SH, TILE):
        for j in range(tx, SH, TILE):
            yy = row0 + i
            xx = col0 + j
            inside = 0 <= yy < h and 0 <= xx < wd and n < n_img
            for c in range(C_IN):
                tile[i, j, c] = x[n, yy, xx, c] if inside else 0.0

    # Cooperative load of the weights (3*3*3*32 = 864 floats).
    tid = ty * TILE + tx
    n_w = KSIZE * KSIZE * C_IN * K_OUT
    for t in range(tid, n_w, TILE * TILE):
        k = t % K_OUT
        c = (t // K_OUT) % C_IN
        dx = (t // (K_OUT * C_IN)) % KSIZE
        dy = t // (K_OUT * C_IN * KSIZE)
        ws[dy, dx, c, k] = w[dy, dx, c, k]
    cuda.syncthreads()

    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx
    if row >= h or col >= wd or n >= n_img:
        return
    for k in range(K_OUT):
        acc = b[k]
        for dy in range(KSIZE):
            for dx in range(KSIZE):
                for c in range(C_IN):
                    acc += tile[ty + dy, tx + dx, c] * ws[dy, dx, c, k]
        out[n, row, col, k] = acc if acc > 0 else 0.0


def launch_config(n_img, h, wd, k_out):
    block = (TILE, TILE)
    gx, gy = math.ceil(wd / TILE), math.ceil(h / TILE)
    return {
        "gpu_naive": ((gx, gy, n_img * k_out), block),
        "gpu_shared": ((gx, gy, n_img), block),
    }


# --------------------------------------------------------------------------- timing

def _timed_gpu(fn):
    """Milliseconds of GPU work: CUDA events on a real GPU, wall clock in the simulator."""
    if SIMULATOR:
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1e3
    start, end = cuda.event(), cuda.event()
    start.record()
    fn()
    end.record()
    end.synchronize()
    return cuda.event_elapsed_time(start, end)


def _timed_host(fn):
    t0 = time.perf_counter()
    result = fn()
    if not SIMULATOR:
        cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, result


def load_batch(batch, small, synthetic):
    rng = np.random.default_rng(config.SEED)
    if small:
        images = rng.integers(0, 256, size=(batch, 20, 20, C_IN), dtype=np.uint8)
    elif not synthetic and (config.NPY_DIR / "X_train.npy").exists():
        images = np.load(config.NPY_DIR / "X_train.npy", mmap_mode="r")[:batch]
        print(f"input: first {batch} images of {config.NPY_DIR / 'X_train.npy'}")
    else:
        images = rng.integers(0, 256, size=(batch, *config.TARGET_SIZE, C_IN), dtype=np.uint8)
        print("input: random uint8 images (synthetic)")
    # Same normalization as the Rescaling(1/255) layer of the model.
    x = np.ascontiguousarray(images, dtype=np.float32) / 255.0
    w = rng.normal(0, math.sqrt(2 / (KSIZE * KSIZE * C_IN)), size=(KSIZE, KSIZE, C_IN, K_OUT)).astype(np.float32)
    b = rng.normal(0, 0.01, size=K_OUT).astype(np.float32)
    return x.astype(np.float32), w, b


def append_rows(rows, path=KERNEL_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLUMNS})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--reps", type=int, default=config.REPS_GPU)
    p.add_argument("--small", action="store_true", help="tiny input (20x20) for the simulator")
    p.add_argument("--synthetic", action="store_true", help="random images instead of X_train.npy")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    batch = 2 if args.small else args.batch
    if SIMULATOR:
        print("NUMBA_ENABLE_CUDASIM=1: kernels run in the CUDA simulator (correctness only, "
              "times are meaningless and nothing is saved)")
    elif not cuda.is_available():
        raise SystemExit("No CUDA GPU visible to Numba. Use NUMBA_ENABLE_CUDASIM=1 --small to test.")
    else:
        dev = cuda.get_current_device()
        print(f"GPU: {dev.name.decode() if isinstance(dev.name, bytes) else dev.name}")

    x, w, b = load_batch(batch, args.small, args.synthetic)
    n_img, h, wd, _ = x.shape
    out_shape = (n_img, h, wd, K_OUT)
    configs = launch_config(n_img, h, wd, K_OUT)

    print(f"conv1: input {x.shape} float32 ({x.nbytes / 1e6:.2f} MB), weights {w.shape}, "
          f"output {out_shape} ({np.prod(out_shape) * 4 / 1e6:.2f} MB)")
    macs = np.prod(out_shape) * KSIZE * KSIZE * C_IN
    print(f"work: {np.prod(out_shape):,} outputs x {KSIZE * KSIZE * C_IN} multiply-adds = {macs / 1e6:.1f} M MACs")
    for name, (grid, block) in configs.items():
        threads = np.prod(grid) * np.prod(block)
        print(f"{name}: grid {grid} x block {block} = {np.prod(grid):,} blocks, {threads:,} threads")

    # Reference results (also compiles the @njit loop before timing).
    ref = conv_numpy(x, w, b)
    out_loop = np.empty(out_shape, np.float32)
    conv_cpu_loop(x, w, b, out_loop)
    assert np.allclose(out_loop, ref, atol=1e-4), "cpu_loop does not match NumPy"

    d_x, d_w, d_b = cuda.to_device(x), cuda.to_device(w), cuda.to_device(b)
    d_out = cuda.device_array(out_shape, np.float32)
    kernels = {"gpu_naive": conv_naive, "gpu_shared": conv_shared}
    for name, kern in kernels.items():  # warm-up: JIT compilation happens on the first launch
        grid, block = configs[name]
        kern[grid, block](d_x, d_w, d_b, d_out)
        got = d_out.copy_to_host()
        ok = np.allclose(got, ref, atol=1e-4)
        print(f"{name}: matches NumPy reference = {ok} (max abs diff {np.abs(got - ref).max():.2e})")
        if not ok:
            raise SystemExit(f"{name} gives a different result")

    times = {k: [] for k in ("cpu_loop", "cpu_numpy", "h2d", "gpu_naive", "gpu_shared", "d2h")}
    for _ in range(args.reps):
        t0 = time.perf_counter()
        conv_cpu_loop(x, w, b, out_loop)
        times["cpu_loop"].append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        conv_numpy(x, w, b)
        times["cpu_numpy"].append((time.perf_counter() - t0) * 1e3)

        ms, d_x = _timed_host(lambda: cuda.to_device(x))
        times["h2d"].append(ms)
        for name, kern in kernels.items():
            grid, block = configs[name]
            times[name].append(_timed_gpu(lambda: kern[grid, block](d_x, d_w, d_b, d_out)))
        ms, _ = _timed_host(lambda: d_out.copy_to_host())
        times["d2h"].append(ms)

    print(f"\n{'version':<12}{'ms (mean)':>12}{'std':>10}{'img/s':>14}")
    for name, values in times.items():
        v = np.array(values)
        print(f"{name:<12}{v.mean():>12.3f}{v.std(ddof=1) if len(v) > 1 else 0:>10.3f}{n_img / v.mean() * 1e3:>14,.0f}")
    copies = np.mean(times["h2d"]) + np.mean(times["d2h"])
    for name in kernels:
        k = np.mean(times[name])
        print(f"speedup {name}: vs cpu_loop {np.mean(times['cpu_loop']) / k:,.1f}x kernel only, "
              f"{np.mean(times['cpu_loop']) / (k + copies):,.1f}x with copies; "
              f"vs cpu_numpy {np.mean(times['cpu_numpy']) / k:,.1f}x / {np.mean(times['cpu_numpy']) / (k + copies):,.1f}x")

    if args.no_save or SIMULATOR or args.small:
        return
    rows = []
    for name, values in times.items():
        dev = "cpu" if name.startswith("cpu") else "gpu"
        for rep, ms in enumerate(values, start=1):
            rows.append({"fase": "kernel", "etapa": name, "maquina": config.MACHINE, "dispositivo": dev,
                         "lote": n_img, "repeticion": rep, "segundos": round(ms / 1e3, 7),
                         "imagenes_por_s": round(n_img / (ms / 1e3), 1)})
    append_rows(rows)
    print(f"\n{len(rows)} rows appended to {KERNEL_CSV}")


if __name__ == "__main__":
    main()
