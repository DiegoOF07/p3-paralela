"""Summaries of results/runs.csv, results/diagnose.csv, results/numba_conv.csv and evidence/.

Pure pandas: no TensorFlow import, so it runs on any laptop.
Grouping rules (README and team guide):
  - always filter by fase and maquina before grouping;
  - group by (etapa, lote), because E4 appears with two batch sizes;
  - per repetition, mean of the timed epochs (2 and 3); then mean and std across repetitions;
  - val_acc only exists on the last epoch.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

COLUMNS = [
    "fase", "etapa", "maquina", "dispositivo", "lote", "repeticion",
    "epoca", "segundos", "imagenes_por_s", "val_acc",
]

STAGE_ORDER = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
# Input dtype that travels to the device per stage: the original generator yields
# float32 already divided by 255; the .npy pipeline sends uint8 and rescales on the device.
STAGE_INPUT_BYTES = {"E0": 4, "E1": 4, "E2": 1, "E3": 1, "E4": 1, "E5": 1, "E6": 1}
STAGE_CHANGE = {
    "E0": "Código original del lab (CPU)",
    "E1": "Mismo código con GPU visible",
    "E2": ".npy + tf.data (CPU)",
    "E3": ".npy + tf.data (GPU)",
    "E4": "Lote grande",
    "E5": "Precisión mixta",
    "E6": "XLA (jit_compile)",
}
IMAGE_BYTES = config.TARGET_SIZE[0] * config.TARGET_SIZE[1] * 3
WARMUP_SECONDS = 5  # monitor samples dropped at the start of each run (process start-up)
BUSY_CORE_PCT = 50


# --------------------------------------------------------------------------- loading

def read_csvs(paths):
    """Concatenates the CSVs that exist; missing files count as empty."""
    frames = [pd.read_csv(p) for p in map(Path, paths) if p.exists() and p.stat().st_size > 0]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    for col in ("lote", "repeticion", "epoca", "segundos", "imagenes_por_s", "val_acc"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("fase", "etapa", "maquina", "dispositivo"):
        df[col] = df[col].astype(str)
    return df


def machines(df):
    return sorted(df["maquina"].dropna().unique()) if len(df) else []


def _rows(df, fase, machine):
    return df[(df["fase"] == fase) & (df["maquina"] == machine)].copy()


def stage_label(etapa, lote):
    return f"{etapa}-{int(lote)}" if etapa == "E4" else str(etapa)


def _stage_sort_key(etapa, lote):
    i = STAGE_ORDER.index(etapa) if etapa in STAGE_ORDER else len(STAGE_ORDER)
    return (i, lote)


# --------------------------------------------------------------------------- training

def train_summary(df, machine):
    """One row per (etapa, lote): epoch time mean/std across repetitions and speedups."""
    t = _rows(df, "train", machine)
    if t.empty:
        return pd.DataFrame()

    timed = t[t["epoca"].isin(config.TIMED_EPOCHS)]
    per_rep = (timed.groupby(["etapa", "lote", "repeticion"], as_index=False)
               .agg(dispositivo=("dispositivo", "first"),
                    segundos=("segundos", "mean"),
                    imagenes_por_s=("imagenes_por_s", "mean")))
    acc = (t[t["epoca"] == config.EPOCHS].groupby(["etapa", "lote"])["val_acc"]
           .agg(val_acc="mean", val_acc_std="std"))

    s = (per_rep.groupby(["etapa", "lote"])
         .agg(dispositivo=("dispositivo", "first"),
              reps=("repeticion", "nunique"),
              seg_media=("segundos", "mean"),
              seg_std=("segundos", "std"),
              img_s=("imagenes_por_s", "mean"))
         .join(acc).reset_index())
    s["seg_std"] = s["seg_std"].fillna(0.0)

    s = s.iloc[sorted(range(len(s)), key=lambda i: _stage_sort_key(s.at[i, "etapa"], s.at[i, "lote"]))]
    s = s.reset_index(drop=True)
    s.insert(0, "etapa_lote", [stage_label(e, b) for e, b in zip(s["etapa"], s["lote"])])
    s.insert(3, "cambio", s["etapa"].map(STAGE_CHANGE).fillna(""))

    base = s.loc[s["etapa"] == "E0", "seg_media"]
    s["speedup_vs_E0"] = base.iloc[0] / s["seg_media"] if len(base) else np.nan
    s["speedup_vs_anterior"] = s["seg_media"].shift(1) / s["seg_media"]
    return s


def best_stage(summary):
    """(label, seconds) of the fastest stage; None if empty."""
    if summary.empty:
        return None
    row = summary.loc[summary["seg_media"].idxmin()]
    return row["etapa_lote"], row["seg_media"]


def _stage_time(summary, etapa, lote=None):
    sel = summary[summary["etapa"] == etapa]
    if lote is not None:
        sel = sel[sel["lote"] == lote]
    return sel["seg_media"].iloc[0] if len(sel) else None


# --------------------------------------------------------------------------- transfer

def transfer_summary(df, machine):
    """Host-to-device copy of one batch, per (etapa, lote)."""
    t = _rows(df, "transfer", machine)
    if t.empty:
        return pd.DataFrame()
    s = (t.groupby(["etapa", "lote"])
         .agg(reps=("repeticion", "nunique"),
              ms_media=("segundos", lambda x: x.mean() * 1e3),
              ms_std=("segundos", lambda x: x.std() * 1e3),
              img_s=("imagenes_por_s", "mean"))
         .reset_index())
    s["ms_std"] = s["ms_std"].fillna(0.0)
    s["bytes_por_imagen"] = s["etapa"].map(STAGE_INPUT_BYTES).fillna(4) * IMAGE_BYTES
    s["mb_por_lote"] = s["lote"] * s["bytes_por_imagen"] / 1e6
    s["gb_s"] = s["mb_por_lote"] / s["ms_media"]  # MB/ms == GB/s
    s = s.iloc[sorted(range(len(s)), key=lambda i: _stage_sort_key(s.at[i, "etapa"], s.at[i, "lote"]))]
    s = s.reset_index(drop=True)
    s.insert(0, "etapa_lote", [stage_label(e, b) for e, b in zip(s["etapa"], s["lote"])])
    return s


# --------------------------------------------------------------------------- phase 1

def prep_summary(df, machine):
    """Phase 1 (data preparation): sequential vs parallel, speedup = seq / par."""
    t = _rows(df, "prep", machine)
    if t.empty:
        return pd.DataFrame()
    s = (t.groupby("etapa")
         .agg(reps=("repeticion", "nunique"),
              seg_media=("segundos", "mean"),
              seg_std=("segundos", "std"),
              img_s=("imagenes_por_s", "mean"))
         .reset_index())
    s["seg_std"] = s["seg_std"].fillna(0.0)
    order = {"secuencial": 0, "paralelo": 1}
    s = s.sort_values("etapa", key=lambda c: c.map(order).fillna(9)).reset_index(drop=True)
    seq = s.loc[s["etapa"] == "secuencial", "seg_media"]
    s["speedup"] = seq.iloc[0] / s["seg_media"] if len(seq) else np.nan
    return s


# --------------------------------------------------------------------------- diagnosis

DIAG_ORDER = ["datos_generator", "datos_arrays", "matematica"]
DIAG_LABEL = {
    "datos_generator": "Solo datos\n(ImageDataGenerator)",
    "datos_arrays": "Solo datos\n(.npy + tf.data)",
    "matematica": "Solo matemática\n(lote en GPU)",
}


def diag_summary(df, machine):
    t = _rows(df, "diag", machine)
    if t.empty:
        return pd.DataFrame()
    s = (t.groupby("etapa")
         .agg(lote=("lote", "first"), seg_epoca=("segundos", "mean"), img_s=("imagenes_por_s", "mean"))
         .reset_index())
    s = s.sort_values("etapa", key=lambda c: c.map({k: i for i, k in enumerate(DIAG_ORDER)}).fillna(9))
    return s.reset_index(drop=True)


# --------------------------------------------------------------------------- numba kernel

KERNEL_ORDER = ["cpu_loop", "cpu_numpy", "h2d", "gpu_naive", "gpu_shared", "d2h"]


def kernel_summary(df, machine):
    """Numba convolution benchmark (results/numba_conv.csv, fase="kernel")."""
    t = _rows(df, "kernel", machine)
    if t.empty:
        return pd.DataFrame()
    s = (t.groupby("etapa")
         .agg(dispositivo=("dispositivo", "first"), lote=("lote", "first"),
              reps=("repeticion", "nunique"),
              ms_media=("segundos", lambda x: x.mean() * 1e3),
              ms_std=("segundos", lambda x: x.std() * 1e3))
         .reset_index())
    s["ms_std"] = s["ms_std"].fillna(0.0)
    s = s.sort_values("etapa", key=lambda c: c.map({k: i for i, k in enumerate(KERNEL_ORDER)}).fillna(9))
    return s.reset_index(drop=True)


def kernel_speedups(ks):
    """Speedup of each GPU kernel against the sequential loop and NumPy, with and without copies."""
    if ks.empty:
        return pd.DataFrame()
    ms = dict(zip(ks["etapa"], ks["ms_media"]))
    copies = ms.get("h2d", 0.0) + ms.get("d2h", 0.0)
    rows = []
    for k in ("gpu_naive", "gpu_shared"):
        if k not in ms:
            continue
        for ref in ("cpu_loop", "cpu_numpy"):
            if ref not in ms:
                continue
            rows.append({"kernel": k, "referencia": ref,
                         "solo_kernel": ms[ref] / ms[k],
                         "con_transferencias": ms[ref] / (ms[k] + copies)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- evidence

RUN_FILE = re.compile(r"^(E\d)_b(\d+)_r(\d+)\.csv$")
IDLE_FILE = re.compile(r"^idle_(E\d)_b(\d+)\.csv$")


def _read_monitor(path):
    m = pd.read_csv(path)
    for col in m.columns:
        m[col] = pd.to_numeric(m[col], errors="coerce")
    return m


def _core_columns(m):
    return [c for c in m.columns if re.fullmatch(r"cpu\d+", c)]


def evidence_files(evidence_dir):
    """{(etapa, lote): {"runs": {rep: path}, "idle": path or None}}"""
    out = {}
    evidence_dir = Path(evidence_dir)
    if not evidence_dir.exists():
        return out
    for p in sorted(evidence_dir.glob("*.csv")):
        if m := RUN_FILE.match(p.name):
            key = (m[1], int(m[2]))
            out.setdefault(key, {"runs": {}, "idle": None})["runs"][int(m[3])] = p
        elif m := IDLE_FILE.match(p.name):
            out.setdefault((m[1], int(m[2])), {"runs": {}, "idle": None})["idle"] = p
    return out


def _monitor_stats(m, skip_warmup):
    if skip_warmup:
        m = m[m["t"] > WARMUP_SECONDS]
    cores = _core_columns(m)
    core_means = m[cores].mean() if cores else pd.Series(dtype=float)
    return {
        "gpu_util": m["gpu_util"].mean(),
        "gpu_util_p95": m["gpu_util"].quantile(0.95) if m["gpu_util"].notna().any() else np.nan,
        "vram_mb": m["vram_mb"].mean(),
        "vram_max_mb": m["vram_mb"].max(),
        "cpu_total": m["cpu_total"].mean(),
        "nucleos_ocupados": int((core_means > BUSY_CORE_PCT).sum()),
        "nucleos": len(cores),
    }


def gpu_usage_summary(evidence_dir):
    """Mean usage during training (averaged over repetitions) next to the idle baseline."""
    rows = []
    for (etapa, lote), files in evidence_files(evidence_dir).items():
        if not files["runs"]:
            continue
        stats = pd.DataFrame([_monitor_stats(_read_monitor(p), True) for p in files["runs"].values()])
        row = {"etapa": etapa, "lote": lote, "reps": len(files["runs"])}
        row.update({c: stats[c].mean() for c in ("gpu_util", "gpu_util_p95", "vram_mb", "cpu_total")})
        row["vram_max_mb"] = stats["vram_max_mb"].max()
        row["nucleos_ocupados"] = stats["nucleos_ocupados"].mean()
        row["nucleos"] = stats["nucleos"].max()
        if files["idle"] is not None:
            idle = _monitor_stats(_read_monitor(files["idle"]), False)
            row.update({"idle_gpu_util": idle["gpu_util"], "idle_vram_mb": idle["vram_mb"],
                        "idle_cpu_total": idle["cpu_total"]})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    s = pd.DataFrame(rows)
    s = s.iloc[sorted(range(len(s)), key=lambda i: _stage_sort_key(s.at[i, "etapa"], s.at[i, "lote"]))]
    s = s.reset_index(drop=True)
    s.insert(0, "etapa_lote", [stage_label(e, b) for e, b in zip(s["etapa"], s["lote"])])
    return s


def core_usage(evidence_dir, etapa, lote):
    """Mean usage per CPU core for one stage, averaged over its repetitions (after warm-up)."""
    files = evidence_files(evidence_dir).get((etapa, lote))
    if not files or not files["runs"]:
        return None
    means = []
    for p in files["runs"].values():
        m = _read_monitor(p)
        m = m[m["t"] > WARMUP_SECONDS]
        means.append(m[_core_columns(m)].mean())
    return pd.concat(means, axis=1).mean(axis=1)


def run_timeline(evidence_dir, etapa, lote, rep=1):
    files = evidence_files(evidence_dir).get((etapa, lote))
    if not files or rep not in files["runs"]:
        return None
    return _read_monitor(files["runs"][rep])


# --------------------------------------------------------------------------- headline numbers

def headline_speedups(train, prep):
    """The three speedups the plan reports; missing data reads as pending."""
    out = []

    if not prep.empty and {"secuencial", "paralelo"} <= set(prep["etapa"]):
        seq = prep.loc[prep["etapa"] == "secuencial", "seg_media"].iloc[0]
        par = prep.loc[prep["etapa"] == "paralelo", "seg_media"].iloc[0]
        out.append(("Fase 1: preparación secuencial / paralela", seq, par, seq / par))
    else:
        out.append(("Fase 1: preparación secuencial / paralela", None, None, None))

    e2, e3 = (_stage_time(train, "E2"), _stage_time(train, "E3")) if not train.empty else (None, None)
    out.append(("Fase 2, efecto del GPU: E2 / E3", e2, e3, e2 / e3 if e2 and e3 else None))

    e0 = _stage_time(train, "E0") if not train.empty else None
    best = best_stage(train)
    best_label, best_s = best if best else (None, None)
    label = f"Total: E0 / mejor etapa ({best_label})" if best_label else "Total: E0 / mejor etapa"
    total = e0 / best_s if e0 and best_s and best_label != "E0" else None
    out.append((label, e0, best_s, total))

    e1 = _stage_time(train, "E1") if not train.empty else None
    out.append(("Mismo código en GPU: E0 / E1", e0, e1, e0 / e1 if e0 and e1 else None))
    return pd.DataFrame(out, columns=["speedup", "t_antes_s", "t_despues_s", "valor"])


# --------------------------------------------------------------------------- markdown

def to_markdown(df, floatfmt=2):
    """Small Markdown table writer (avoids the optional tabulate dependency)."""
    if df.empty:
        return "_sin datos_\n"

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:,.{floatfmt}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def write_tables(out_dir, machine, train, transfer, prep, diag, usage, kernel, speedups):
    """Writes every non-empty summary as CSV plus a single Markdown file for the report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "etapas": train, "transfer": transfer, "prep": prep, "diagnostico": diag,
        "uso_recursos": usage, "numba": kernel, "speedups": speedups,
    }
    written = []
    for name, df in tables.items():
        if df is None or df.empty:
            continue
        path = out_dir / f"tabla_{name}_{machine}.csv"
        df.round(4).to_csv(path, index=False)
        written.append(path)

    md = [f"# Resultados — {machine}\n"]
    if not train.empty:
        cols = ["etapa_lote", "dispositivo", "cambio", "reps", "seg_media", "seg_std",
                "img_s", "val_acc", "speedup_vs_E0", "speedup_vs_anterior"]
        md += ["## Tiempo por época (promedio de las épocas 2 y 3)\n", to_markdown(train[cols])]
    if not speedups.empty:
        sp = speedups.copy()
        sp["valor"] = sp["valor"].map(lambda v: f"{v:.2f}×" if pd.notna(v) else "pendiente")
        md += ["## Speedups reportados\n", to_markdown(sp)]
    if not transfer.empty:
        cols = ["etapa_lote", "reps", "mb_por_lote", "ms_media", "ms_std", "gb_s"]
        md += ["## Transferencia CPU → GPU (un lote)\n", to_markdown(transfer[cols], 3)]
    if not prep.empty:
        md += ["## Fase 1: preparación de datos\n", to_markdown(prep)]
    if not diag.empty:
        md += ["## Diagnóstico (segundos por época equivalente)\n", to_markdown(diag)]
    if not usage.empty:
        cols = [c for c in ["etapa_lote", "reps", "idle_gpu_util", "gpu_util", "gpu_util_p95",
                            "idle_vram_mb", "vram_max_mb", "cpu_total", "nucleos_ocupados"] if c in usage]
        md += ["## Uso de recursos (media por segundo, sin los primeros 5 s)\n", to_markdown(usage[cols], 1)]
    if not kernel.empty:
        md += ["## Kernel de convolución en Numba (ms por llamada)\n", to_markdown(kernel, 3)]
        ksp = kernel_speedups(kernel)
        if not ksp.empty:
            md += ["### Speedup del kernel\n", to_markdown(ksp)]
    path = out_dir / f"resumen_{machine}.md"
    path.write_text("\n".join(md), encoding="utf-8")
    written.append(path)
    return written
