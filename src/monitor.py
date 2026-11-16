import csv
import shutil
import subprocess
import threading
import time

import psutil

from src import config

NVIDIA_QUERY = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used",
    "--format=csv,noheader,nounits",
]


def _gpu_sample():
    try:
        out = subprocess.run(NVIDIA_QUERY, capture_output=True, text=True, timeout=5).stdout
        util, mem = out.strip().splitlines()[0].split(",")
        return int(util), int(mem)
    except (subprocess.SubprocessError, ValueError, IndexError):
        return "", ""


class ResourceMonitor:
    """Samples GPU usage, VRAM and per-core CPU usage once per interval to a CSV."""

    def __init__(self, tag, interval=1.0):
        self.path = config.EVIDENCE_DIR / f"{tag}.csv"
        self.interval = interval
        self.has_gpu = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        n_cores = psutil.cpu_count()
        header = ["t", "gpu_util", "vram_mb", "cpu_total"] + [f"cpu{i}" for i in range(n_cores)]
        psutil.cpu_percent(percpu=True)  # first call only primes the counters
        t0 = time.perf_counter()
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            while not self._stop.wait(self.interval):
                cores = psutil.cpu_percent(percpu=True)
                util, mem = _gpu_sample() if self.has_gpu else ("", "")
                total = round(sum(cores) / len(cores), 1)
                writer.writerow([round(time.perf_counter() - t0, 1), util, mem, total] + cores)
                f.flush()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
