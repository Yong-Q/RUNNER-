import argparse
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from ase.io import read
from .common import read_json, fresh_dir, dump, select_frames, write_frames


def main():
    p = argparse.ArgumentParser(
        description="RuNNer / MACE gas active-learning workflow (Slurm)"
    )
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("doctor")
    x.add_argument("--config")
    for name in ("sample", "benchmark"):
        x = sub.add_parser(name)
        x.add_argument("--config", required=True)
        x.add_argument("--input", required=True)
        x.add_argument("--index", type=int, default=0)
        x.add_argument("--out", required=True)
        if name == "sample":
            x.add_argument("--steps", type=int)
        else:
            x.add_argument("--warmup", type=int, default=2)
            x.add_argument("--repeats", type=int, default=5)
    x = sub.add_parser("select")
    x.add_argument("--input", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--budget", type=int, default=10)
    x.add_argument("--minimum", type=float, default=0.0)
    x.add_argument("--maximum", type=float, default=1.0)
    x.add_argument("--min-step-gap", type=int, default=10)
    x = sub.add_parser("prepare-vasp")
    x.add_argument("--input", required=True)
    x.add_argument("--reference", required=True)
    x.add_argument("--out", required=True)
    x = sub.add_parser("run-vasp")
    x.add_argument("task")
    x = sub.add_parser("run-job")
    x.add_argument("task")
    x.add_argument("argv", nargs=argparse.REMAINDER)
    x = sub.add_parser("collect")
    x.add_argument("--tasks", required=True)
    x.add_argument("--out", required=True)
    x = sub.add_parser("prepare-train")
    x.add_argument("--input", required=True)
    x.add_argument("--config", required=True)
    x.add_argument("--out", required=True)
    x = sub.add_parser("assemble-runner")
    x.add_argument("--training", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--library", required=True)
    x = sub.add_parser("status")
    x.add_argument("path")
    a = p.parse_args()
    if a.command == "doctor":
        versions = {}
        for name in (
            "ase",
            "numpy",
            "mace-torch",
            "graph-longrange",
            "runnerase",
            "runnerase-prediction",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        report = {
            "versions": versions,
            "sbatch": shutil.which("sbatch"),
            "vasp_std_current_shell": shutil.which("vasp_std"),
            "VASP_PP_PATH": os.getenv("VASP_PP_PATH"),
            "compute_job": os.getenv("SLURM_JOB_ID"),
        }
        if a.config:
            cfg = read_json(a.config)
            paths = cfg.get("models", []) + [
                cfg[k] for k in ("potential", "library") if k in cfg
            ]
            report["paths"] = {str(x): Path(x).exists() for x in paths}
            report["backend"] = cfg["backend"]
        print(json.dumps(report, indent=2))
        return
    if a.command in ("sample", "benchmark"):
        from .sampling import sample, benchmark

        cfg = read_json(a.config)
        if a.command == "sample":
            result = sample(cfg, a.input, a.index, a.out, a.steps)
        else:
            result = benchmark(cfg, a.input, a.index, a.out, a.warmup, a.repeats)
    elif a.command == "select":
        result = fresh_dir(a.out)
        frames = select_frames(
            read(a.input, ":"), a.minimum, a.maximum, a.budget, a.min_step_gap
        )
        if frames:
            write_frames(result / "candidates.extxyz", frames)
        dump(
            result / "selection.json",
            {
                "count": len(frames),
                "calibrated": False,
                "minimum": a.minimum,
                "maximum": a.maximum,
            },
        )
    elif a.command == "prepare-vasp":
        from .labeling import prepare_vasp

        result = prepare_vasp(a.input, read_json(a.reference), a.out)
    elif a.command == "run-vasp":
        from .labeling import run_vasp

        run_vasp(a.task)
        result = a.task
    elif a.command == "run-job":
        from .jobs import run_claimed
        from .common import require_compute

        require_compute()
        argv = a.argv[1:] if a.argv and a.argv[0] == "--" else a.argv
        if not argv:
            raise ValueError("Missing job command")
        run_claimed(a.task, argv)
        result = a.task
    elif a.command == "collect":
        from .labeling import collect

        result = collect(a.tasks, a.out)
    elif a.command == "prepare-train":
        from .training import prepare_training

        result = prepare_training(a.input, read_json(a.config), a.out)
    elif a.command == "assemble-runner":
        from .training import assemble_runner

        result = assemble_runner(a.training, a.out, a.library)
    else:
        root = Path(a.path)
        summaries = []
        for filename in (
            "summary.json",
            "manifest.json",
            "collection.json",
            "label.json",
            "failure.json",
            "benchmark.json",
            "execution_claim.json",
            "exit_status.json",
        ):
            path = root / filename
            if path.exists():
                summaries.append({"file": filename, "data": read_json(path)})
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return
    print(result)
