"""Tables and figures for the report, built from results/ and evidence/.

    python -m src.plots                               # official machine (MACHINE_NAME)
    python -m src.plots --machine colab-t4 --runs colab_runs.csv
    python -m src.plots --all-machines                # one set of outputs per machine
    python -m src.plots --demo                        # invented data, outputs in results/demo/

Every figure is skipped with a message when its data does not exist yet, so the
script can run at any point of the project.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import analysis as an  # noqa: E402
from src import config  # noqa: E402

DIAG_CSV = config.RESULTS_DIR / "diagnose.csv"
KERNEL_CSV = config.RESULTS_DIR / "numba_conv.csv"

# Colors carry one job each: device identity (CPU / GPU) and idle reference.
CPU = "#eb6834"
GPU = "#2a78d6"
IDLE = "#b5b3ad"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e2dc"
DEVICE_COLOR = {"cpu": CPU, "gpu": GPU}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 10,
    "legend.frameon": False,
})


def _save(fig, out_dir, name, machine):
    path = Path(out_dir) / f"{name}_{machine}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _skip(name, why):
    print(f"  skip {name}: {why}")


def _device_legend(ax, devices):
    from matplotlib.patches import Patch
    handles = [Patch(color=DEVICE_COLOR[d], label=d.upper()) for d in ("cpu", "gpu") if d in devices]
    ax.legend(handles=handles, loc="upper right")


def _bar_labels(ax, bars, texts, pad=3, err=None):
    err = [0.0] * len(bars) if err is None else list(err)
    for b, t, e in zip(bars, texts, err):
        ax.annotate(t, (b.get_x() + b.get_width() / 2, b.get_height() + e), xytext=(0, pad),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)


# --------------------------------------------------------------------------- figures

def fig_epoch_time(train, out_dir, machine):
    if train.empty:
        return _skip("fig_tiempo_epoca", "no train rows")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(train))
    colors = [DEVICE_COLOR.get(d, IDLE) for d in train["dispositivo"]]
    bars = ax.bar(x, train["seg_media"], 0.62, yerr=train["seg_std"], color=colors,
                  error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    _bar_labels(ax, bars, [f"{v:.1f} s" for v in train["seg_media"]], err=train["seg_std"])
    ax.set_xticks(x, train["etapa_lote"])
    ax.set_ylabel("Segundos por época (media de épocas 2 y 3)")
    ax.set_title("Tiempo por época de entrenamiento por etapa")
    ax.set_ylim(0, train["seg_media"].max() * 1.15)
    _device_legend(ax, set(train["dispositivo"]))
    ax.text(0, -0.16, f"Barras de error: desviación estándar entre repeticiones · {machine}",
            transform=ax.transAxes, fontsize=8, color=INK_2)
    return _save(fig, out_dir, "fig_tiempo_epoca", machine)


def fig_speedup(train, out_dir, machine):
    if train.empty or "E0" not in set(train["etapa"]) or len(train) < 2:
        return _skip("fig_speedup", "needs E0 and at least one other stage")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(train))
    colors = [DEVICE_COLOR.get(d, IDLE) for d in train["dispositivo"]]
    bars = ax.bar(x, train["speedup_vs_E0"], 0.62, color=colors)
    _bar_labels(ax, bars, [f"{v:.1f}×" for v in train["speedup_vs_E0"]])
    ax.axhline(1, color=INK_2, linewidth=1, linestyle="--")
    ax.set_xticks(x, train["etapa_lote"])
    ax.set_ylabel("Speedup frente a E0 (×)")
    ax.set_title("Aceleración acumulada frente al código original (E0)")
    ax.set_ylim(0, train["speedup_vs_E0"].max() * 1.15)
    _device_legend(ax, set(train["dispositivo"]))
    return _save(fig, out_dir, "fig_speedup", machine)


def fig_gpu_usage(usage, out_dir, machine):
    if usage.empty or usage["gpu_util"].isna().all():
        return _skip("fig_gpu_uso", "no GPU samples in evidence/")
    width = max(10, 1.25 * len(usage) + 2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(width, 4.2))
    x = np.arange(len(usage))
    w = 0.38
    has_idle = "idle_gpu_util" in usage
    shift = w / 2 if has_idle else 0

    if has_idle:
        a1.bar(x - shift, usage["idle_gpu_util"], w, color=IDLE, label="Reposo (15 s antes de la etapa)")
    b = a1.bar(x + shift, usage["gpu_util"], w, color=GPU, label="Entrenando")
    _bar_labels(a1, b, [f"{v:.0f}%" for v in usage["gpu_util"]])
    a1.set_xticks(x, usage["etapa_lote"], fontsize=9)
    a1.set_ylim(0, 110)
    a1.set_ylabel("Utilización del GPU (%)")
    a1.set_title("Uso del GPU")

    if has_idle:
        a2.bar(x - shift, usage["idle_vram_mb"], w, color=IDLE)
    b = a2.bar(x + shift, usage["vram_max_mb"], w, color=GPU)
    _bar_labels(a2, b, [f"{v:,.0f}" for v in usage["vram_max_mb"]])
    a2.axhline(6144, color=INK_2, linestyle="--", linewidth=1)
    a2.text(-0.4, 6144, "Límite 6 GB (RTX 3050)", ha="left", va="bottom", fontsize=8, color=INK_2)
    a2.set_xticks(x, usage["etapa_lote"], fontsize=9)
    a2.set_ylim(0, max(6144, usage["vram_max_mb"].max()) * 1.12)
    a2.set_ylabel("VRAM (MB; entrenando = máximo)")
    a2.set_title("Memoria del GPU")

    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, -0.07, "Media por segundo de nvidia-smi, sin los primeros 5 s de arranque; "
             "promedio entre repeticiones.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out_dir, "fig_gpu_uso", machine)


def fig_cpu_cores(evidence_dir, train, out_dir, machine):
    """Heatmap stage x core: where the host work runs (all cores in CPU stages, few in GPU stages)."""
    stages = [(e, int(b)) for e, b in zip(train["etapa"], train["lote"])] if not train.empty else []
    if not stages:
        stages = list(an.evidence_files(evidence_dir))
    labels, rows = [], []
    for etapa, lote in stages:
        cores = an.core_usage(evidence_dir, etapa, lote)
        if cores is not None:
            dev = "CPU" if etapa in ("E0", "E2") else "GPU"
            labels.append(f"{an.stage_label(etapa, lote)} ({dev})")
            rows.append(cores.values)
    if not rows:
        return _skip("fig_cpu_nucleos", "no per-core samples in evidence/")

    data = np.vstack(rows)
    n_stages, n_cores = data.shape
    fig, ax = plt.subplots(figsize=(1.6 + 0.55 * n_cores, 1.2 + 0.45 * n_stages))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(n_stages):
        for j in range(n_cores):
            v = data[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 60 else INK)
    ax.set_xticks(range(n_cores), [str(j) for j in range(n_cores)])
    ax.set_yticks(range(n_stages), labels)
    ax.set_xlabel("Núcleo lógico del CPU")
    ax.set_title("Uso medio por núcleo del CPU durante el entrenamiento (%)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("% de uso")
    return _save(fig, out_dir, "fig_cpu_nucleos", machine)


def fig_timelines(evidence_dir, train, out_dir, machine):
    """GPU utilization and VRAM over time for repetition 1 of every stage."""
    keys = [(e, int(b)) for e, b in zip(train["etapa"], train["lote"])] if not train.empty else []
    runs = [(k, an.run_timeline(evidence_dir, *k)) for k in keys]
    runs = [(k, m) for k, m in runs if m is not None and m["gpu_util"].notna().any()]
    if not runs:
        return _skip("fig_timeline", "no GPU timelines in evidence/")

    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 1.2 + 1.5 * len(runs)), squeeze=False)
    for ax, ((etapa, lote), m) in zip(axes[:, 0], runs):
        ax.plot(m["t"], m["gpu_util"], color=GPU, linewidth=1.4, label="GPU %")
        ax.plot(m["t"], m["cpu_total"], color=CPU, linewidth=1.4, label="CPU total %")
        ax.set_ylim(0, 105)
        ax.set_ylabel(an.stage_label(etapa, lote), rotation=0, ha="right", va="center", fontweight="bold")
        ax.set_title(f"VRAM máx {m['vram_mb'].max():,.0f} MB", loc="right", fontsize=8,
                     fontweight="normal", color=INK_2, pad=2)
    axes[-1, 0].set_xlabel("Segundos desde el inicio de la corrida")
    fig.suptitle("Uso de GPU y CPU por segundo (repetición 1)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save(fig, out_dir, "fig_timeline", machine)


def fig_transfer(transfer, out_dir, machine):
    if transfer.empty:
        return _skip("fig_transfer", "no transfer rows")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(transfer))
    b = a1.bar(x, transfer["ms_media"], 0.6, yerr=transfer["ms_std"], color=GPU,
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    _bar_labels(a1, b, [f"{v:.2f} ms" for v in transfer["ms_media"]], err=transfer["ms_std"])
    a1.set_xticks(x, transfer["etapa_lote"])
    a1.set_ylabel("ms por lote")
    a1.set_title("Copia de un lote CPU → GPU")
    a1.set_ylim(0, (transfer["ms_media"] + transfer["ms_std"]).max() * 1.25)

    b = a2.bar(x, transfer["mb_por_lote"], 0.6, color=GPU)
    _bar_labels(a2, b, [f"{v:.1f} MB" for v in transfer["mb_por_lote"]])
    a2.set_xticks(x, transfer["etapa_lote"])
    a2.set_ylabel("MB por lote")
    a2.set_title("Tamaño del lote transferido")
    a2.set_ylim(0, transfer["mb_por_lote"].max() * 1.2)
    fig.text(0.01, -0.03, "E0–E1 envían float32 (4 B por canal); desde E2 viajan uint8 (1 B) y "
             "la normalización ocurre en el GPU.", fontsize=8, color=INK_2)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_transfer", machine)


def fig_prep(prep, out_dir, machine):
    if prep.empty:
        return _skip("fig_prep", "no prep rows (run src.data_prep)")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(len(prep))
    b = ax.bar(x, prep["seg_media"], 0.55, yerr=prep["seg_std"], color=CPU,
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    _bar_labels(ax, b, [f"{s:.1f} s · {sp:.1f}×" for s, sp in zip(prep["seg_media"], prep["speedup"])],
                err=prep["seg_std"])
    ax.set_xticks(x, prep["etapa"].str.capitalize())
    ax.set_ylabel("Segundos (73,950 imágenes)")
    ax.set_title("Fase 1: preparación de imágenes en CPU")
    ax.set_ylim(0, prep["seg_media"].max() * 1.2)
    return _save(fig, out_dir, "fig_prep", machine)


def fig_diag(diag, out_dir, machine):
    if diag.empty:
        return _skip("fig_diagnostico", "no diagnose.csv rows (run src.diagnose)")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(diag))
    colors = [GPU if e == "matematica" else CPU for e in diag["etapa"]]
    b = ax.bar(x, diag["seg_epoca"], 0.55, color=colors)
    _bar_labels(ax, b, [f"{v:.1f} s" for v in diag["seg_epoca"]])
    ax.set_xticks(x, [an.DIAG_LABEL.get(e, e) for e in diag["etapa"]])
    ax.set_ylabel("Segundos por época equivalente")
    ax.set_title("Diagnóstico: ¿limita la carga de datos o la matemática?")
    ax.set_ylim(0, diag["seg_epoca"].max() * 1.18)
    return _save(fig, out_dir, "fig_diagnostico", machine)


def fig_kernel(kernel, out_dir, machine):
    if kernel.empty:
        return _skip("fig_numba", "no numba_conv.csv rows (run src.numba_conv)")
    ms = dict(zip(kernel["etapa"], kernel["ms_media"]))
    copies = ms.get("h2d", 0.0) + ms.get("d2h", 0.0)
    bars = []  # (label, ms, color)
    for k, name in (("cpu_loop", "CPU: bucle secuencial"), ("cpu_numpy", "CPU: NumPy vectorizado")):
        if k in ms:
            bars.append((name, ms[k], CPU))
    for k, name in (("gpu_naive", "GPU: kernel simple"), ("gpu_shared", "GPU: kernel mem. compartida")):
        if k in ms:
            bars.append((f"{name}\n(solo kernel)", ms[k], GPU))
            if copies:
                bars.append((f"{name}\n(+ copias H2D y D2H)", ms[k] + copies, IDLE))
    bars = bars[::-1]  # first item on top
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.55 * len(bars)))
    y = np.arange(len(bars))
    vals = [b[1] for b in bars]
    ax.barh(y, vals, 0.6, color=[b[2] for b in bars])
    ax.set_xscale("log")
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:,.2f} ms", (v, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [b[0] for b in bars], fontsize=9)
    ax.set_xlim(min(vals) / 2, max(vals) * 8)
    ax.set_xlabel("ms por llamada (escala logarítmica)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    lote = int(kernel["lote"].dropna().iloc[0]) if kernel["lote"].notna().any() else "?"
    ax.set_title(f"Convolución 3×3 de conv1 (lote {lote}) con CUDA Python (Numba)")
    return _save(fig, out_dir, "fig_numba", machine)


# --------------------------------------------------------------------------- driver

def build(machine, runs, diag_df, kernel_df, evidence_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"== {machine} -> {out_dir}")

    train = an.train_summary(runs, machine)
    transfer = an.transfer_summary(runs, machine)
    prep = an.prep_summary(runs, machine)
    diag = an.diag_summary(diag_df, machine)
    kernel = an.kernel_summary(kernel_df, machine)
    # evidence/ files carry no machine column: they belong to the machine that wrote them.
    usage = an.gpu_usage_summary(evidence_dir) if evidence_dir else an.gpu_usage_summary("")
    speedups = an.headline_speedups(train, prep)

    written = an.write_tables(out_dir, machine, train, transfer, prep, diag, usage, kernel, speedups)
    for fn in (lambda: fig_epoch_time(train, out_dir, machine),
               lambda: fig_speedup(train, out_dir, machine),
               lambda: fig_gpu_usage(usage, out_dir, machine),
               lambda: fig_cpu_cores(evidence_dir, train, out_dir, machine) if evidence_dir else None,
               lambda: fig_timelines(evidence_dir, train, out_dir, machine) if evidence_dir else None,
               lambda: fig_transfer(transfer, out_dir, machine),
               lambda: fig_prep(prep, out_dir, machine),
               lambda: fig_diag(diag, out_dir, machine),
               lambda: fig_kernel(kernel, out_dir, machine)):
        path = fn()
        if path:
            written.append(path)

    for p in written:
        print(f"  wrote {p.relative_to(config.ROOT) if p.is_relative_to(config.ROOT) else p}")
    print()
    print(speedups.assign(valor=speedups["valor"].map(lambda v: f"{v:.2f}x" if v == v and v is not None else "pendiente"))
          .to_string(index=False))
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--machine", default=config.MACHINE, help="maquina value to report")
    p.add_argument("--all-machines", action="store_true", help="one set of outputs per machine found")
    p.add_argument("--runs", nargs="*", default=[], help="extra runs.csv files (e.g. from Colab)")
    p.add_argument("--diag", nargs="*", default=[], help="extra diagnose.csv files")
    p.add_argument("--kernel", nargs="*", default=[], help="extra numba_conv.csv files")
    p.add_argument("--evidence", default=None,
                   help="monitor logs of --machine (default: evidence/ for MACHINE_NAME, none for others)")
    p.add_argument("--out", default=str(config.RESULTS_DIR))
    p.add_argument("--demo", action="store_true", help="invented data in results/demo/ to test the plots")
    return p.parse_args()


def main():
    args = parse_args()
    if args.demo:
        from src import demo_data
        demo = demo_data.write_demo(config.RESULTS_DIR / "demo")
        runs = an.read_csvs([demo["runs"]])
        diag = an.read_csvs([demo["diag"]])
        kernel = an.read_csvs([demo["kernel"]])
        build(demo_data.MACHINE, runs, diag, kernel, demo["evidence"], demo["dir"])
        return

    runs = an.read_csvs([config.RUNS_CSV, *args.runs])
    diag = an.read_csvs([DIAG_CSV, *args.diag])
    kernel = an.read_csvs([KERNEL_CSV, *args.kernel])
    found = sorted(set(an.machines(runs)) | set(an.machines(diag)) | set(an.machines(kernel)))
    targets = found if args.all_machines else [args.machine]
    if not found:
        print("No results yet: run src.bench first, or try --demo.")
        return
    for machine in targets:
        if machine not in found:
            print(f"No rows for machine {machine!r}; found: {', '.join(found)}")
            continue
        # evidence/ holds the logs of the machine that ran the stages here (MACHINE_NAME).
        evidence = args.evidence if args.evidence is not None else (
            config.EVIDENCE_DIR if machine == config.MACHINE else "")
        build(machine, runs, diag, kernel, evidence, args.out)


if __name__ == "__main__":
    main()
