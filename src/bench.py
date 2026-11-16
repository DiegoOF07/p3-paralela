"""Runs every stage with its repetitions, one fresh process per run.

    python -m src.bench --stages E0 E2      # CPU stages, overnight
    python -m src.bench --stages E1 E3 E4 E5 E6
"""
import argparse
import csv
import subprocess
import sys
import time

from src import config
from src.monitor import ResourceMonitor
from src.train import STAGES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", default=sorted(STAGES), choices=sorted(STAGES))
    p.add_argument("--reps", type=int, default=None, help="override repetitions for every stage")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--synthetic", action="store_true", help="random arrays; nothing is saved")
    p.add_argument("--force", action="store_true", help="rerun runs already in runs.csv")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def plan_runs(stages, reps_override):
    runs = []
    for name in stages:
        stage = STAGES[name]
        batches = config.LARGE_BATCHES if name == "E4" else (stage["batch"],)
        reps = reps_override or (config.REPS_CPU if stage["device"] == "cpu" else config.REPS_GPU)
        for batch in batches:
            runs.append((name, batch, list(range(1, reps + 1))))
    return runs


def done_runs():
    """(stage, batch, rep) already recorded for this machine, so a crashed night can resume."""
    if not config.RUNS_CSV.exists():
        return set()
    with open(config.RUNS_CSV) as f:
        return {
            (r["etapa"], int(r["lote"]), int(r["repeticion"]))
            for r in csv.DictReader(f)
            if r["fase"] == "train" and r["maquina"] == config.MACHINE
        }


def idle_baseline(tag):
    print(f"  idle baseline ({config.IDLE_SECONDS}s) -> evidence/idle_{tag}.csv", flush=True)
    with ResourceMonitor(f"idle_{tag}"):
        time.sleep(config.IDLE_SECONDS)


def main():
    args = parse_args()
    done = set() if (args.force or args.synthetic) else done_runs()
    failures = []
    start = time.perf_counter()

    for name, batch, reps in plan_runs(args.stages, args.reps):
        pending = [r for r in reps if (name, batch, r) not in done]
        print(f"== {name} batch {batch}: reps {pending or 'all done'}", flush=True)
        if not pending:
            continue
        if not args.dry_run and not args.synthetic:
            idle_baseline(f"{name}_b{batch}")

        for rep in pending:
            cmd = [sys.executable, "-m", "src.train", "--stage", name, "--batch", str(batch),
                   "--rep", str(rep), "--epochs", str(args.epochs)]
            if args.synthetic:
                cmd += ["--synthetic", "--no-save"]
            print("  $ " + " ".join(cmd[1:]), flush=True)
            if args.dry_run:
                continue
            t0 = time.perf_counter()
            result = subprocess.run(cmd, cwd=config.ROOT)
            minutes = (time.perf_counter() - t0) / 60
            status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            print(f"  {name} b{batch} r{rep}: {status} in {minutes:.1f} min", flush=True)
            if result.returncode != 0:
                failures.append(f"{name} b{batch} r{rep}")

    print(f"\nbench finished in {(time.perf_counter() - start) / 60:.1f} min")
    if failures:
        print("failed runs: " + ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
