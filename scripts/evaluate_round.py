#!/usr/bin/env python3
"""Evaluate a committee against one explicitly identified labeled dataset."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rnal.backends import load_backend
from rnal.common import read_json, fresh_dir, dump

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
config = read_json(a.config)
frames = read(a.data, ":")
calc = load_backend(config)
out = fresh_dir(a.out)
rows = []
for atoms in frames:
    calc.calculate(atoms)
    forces = np.asarray(calc.results["forces"])
    reference = np.asarray(atoms.arrays["REF_forces"])
    rows.append(
        {
            "case": atoms.info.get("case"),
            "group": atoms.info["group"],
            "atoms": len(atoms),
            "reference_id": atoms.info["reference_id"],
            "predicted_energy_eV": float(calc.results["energy"]),
            "reference_energy_eV": float(atoms.info["REF_energy"]),
            "energy_error_eV_per_atom": float(
                (calc.results["energy"] - atoms.info["REF_energy"]) / len(atoms)
            ),
            "force_component_RMSE_eVA": float(
                np.sqrt(np.mean((forces - reference) ** 2))
            ),
            "maximum_absolute_force_component_error_eVA": float(
                np.abs(forces - reference).max()
            ),
            "max_force_disagreement_eVA": float(calc.results["force_sigma"].max()),
        }
    )
dump(
    out / "evaluation.json",
    {
        "config": config,
        "frames": rows,
        "interpretation": "Tiny workflow test; membership in train/valid must be read from the training manifest. No calibrated uncertainty or production accuracy claim.",
    },
)
print(json.dumps(rows, indent=2))
