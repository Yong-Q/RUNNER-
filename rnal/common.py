import hashlib
import json
import os
from pathlib import Path
import numpy as np
from ase.io import write


def fresh_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def dump(path, data):
    path = Path(path)
    with path.open("x") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def geometry_id(atoms):
    return fingerprint(
        {
            "Z": atoms.numbers.tolist(),
            "R": atoms.positions.round(7).tolist(),
            "cell": atoms.cell.array.round(7).tolist(),
            "pbc": atoms.pbc.tolist(),
        }
    )


def force_disagreement(forces):
    f = np.asarray(forces, float)
    if f.ndim != 3 or f.shape[0] < 2 or f.shape[-1] != 3 or not np.isfinite(f).all():
        raise ValueError("Need finite forces [at least 2 members, atoms, 3].")
    return np.sqrt(np.mean(np.sum((f - f.mean(axis=0)) ** 2, axis=-1), axis=0))


def select_frames(
    frames, minimum=0.0, maximum=float("inf"), budget=10, min_step_gap=10
):
    if not 0 <= minimum < maximum or budget < 1 or min_step_gap < 0:
        raise ValueError("Invalid selection settings")
    ranked = sorted(
        frames, key=lambda a: float(a.info["force_sigma_max"]), reverse=True
    )
    chosen = []
    seen = set()
    steps = {}
    for a in ranked:
        score = float(a.info["force_sigma_max"])
        if not np.isfinite(score) or not minimum <= score <= maximum:
            continue
        key = geometry_id(a)
        group = str(a.info.get("group", "ungrouped"))
        step = a.info.get("step")
        if key in seen:
            continue
        if step is not None and any(
            abs(int(step) - s) < min_step_gap for s in steps.get(group, [])
        ):
            continue
        b = a.copy()
        b.calc = None
        b.info["selection"] = "uncalibrated_force_disagreement"
        chosen.append(b)
        seen.add(key)
        if step is not None:
            steps.setdefault(group, []).append(int(step))
        if len(chosen) >= budget:
            break
    return chosen


def split_groups(frames, fraction=0.2, seed=42):
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1")
    if any("group" not in a.info for a in frames):
        raise ValueError("Every frame needs a source group")
    seen = {}
    for a in frames:
        key = geometry_id(a)
        group = str(a.info["group"])
        if key in seen and seen[key] != group:
            raise ValueError(
                "Identical geometry appears in different groups; deduplicate or merge source groups before splitting"
            )
        seen[key] = group
    groups = sorted({str(a.info["group"]) for a in frames})
    if len(groups) < 2:
        raise ValueError("Need at least two independent groups for a split")
    np.random.default_rng(seed).shuffle(groups)
    n = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    valid = set(groups[:n])
    return (
        [a for a in frames if str(a.info["group"]) not in valid],
        [a for a in frames if str(a.info["group"]) in valid],
    )


def write_frames(path, frames):
    if not frames:
        raise ValueError("No frames to write")
    if Path(path).exists():
        raise FileExistsError(path)
    write(path, frames, format="extxyz", write_results=False)


def require_compute():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Run model inference/training/DFT through sbatch on a compute node."
        )


def total_charge(atoms):
    values = [
        float(atoms.info[k]) for k in ("charge", "total_charge") if k in atoms.info
    ]
    if any(not np.isfinite(q) for q in values) or (
        len(values) == 2 and abs(values[0] - values[1]) > 1e-10
    ):
        raise ValueError("Conflicting or non-finite total charge metadata")
    return values[0] if values else 0.0


def structure_id(atoms):
    return fingerprint(
        {
            "geometry": geometry_id(atoms),
            "charge": total_charge(atoms),
            "spin": float(atoms.info.get("spin", 1)),
            "external_field": np.asarray(
                atoms.info.get("external_field", [0, 0, 0])
            ).tolist(),
        }
    )
