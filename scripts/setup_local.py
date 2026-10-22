#!/usr/bin/env python3
"""Create local site configuration without replacing existing files."""

import argparse, json, shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--runner-build", default="build/runner_v4")
a = p.parse_args()
local = root / "local"
local.mkdir(exist_ok=True)
cfg = json.loads((root / "configs/polar.example.json").read_text())
cfg["models"] = [str(Path(a.model_dir).resolve() / Path(x).name) for x in cfg["models"]]
with (local / "polar.json").open("x") as f:
    json.dump(cfg, f, indent=2)
runner = json.loads((root / "configs/runner.example.json").read_text())
runner["library"] = str((root / a.runner_build / "libRuNNer.so").resolve())
runner["potential"] = str((local / "runner_ag_committee").resolve())
with (local / "runner_ag.json").open("x") as f:
    json.dump(runner, f, indent=2)
source = root / "external/runner2/tests/regression/committee_prediction"
if source.exists():
    target = local / "runner_ag_committee"
    target.mkdir()
    for name in (
        "input.nn",
        "input.data",
        "Ag3.data",
        "scaling.data",
        "qeq_scaling.data",
    ):
        shutil.copy2(source / name, target / name)
    for i in (1, 2):
        shutil.copytree(source / f"{i}_template", target / str(i))
print(local)
