import hashlib
import os
import shlex
import sys
from pathlib import Path
import numpy as np
from ase.io import read
from .common import (
    fresh_dir,
    dump,
    read_json,
    fingerprint,
    write_frames,
    require_compute,
    total_charge,
    structure_id,
)


def validate_reference(reference):
    p = reference.get("vasp_parameters", {})
    if (
        not reference.get("method_id")
        or not p.get("xc")
        or float(p.get("encut", 0)) <= 0
    ):
        raise ValueError(
            "Reference needs method_id, xc and positive encut; choose a DFT reference first"
        )
    if p.get("nsw", 0) != 0 or p.get("ibrion", -1) != -1:
        raise ValueError("Labeling must use static single points: NSW=0, IBRION=-1")
    if reference.get("energy_convention", "free_energy") != "free_energy":
        raise ValueError("Use force-consistent VASP free_energy labels")
    if (
        "nelect" in p
        or p.get("ispin", 1) != 1
        or p.get("nupdown", 0) != 0
        or p.get("lsorbit", False)
        or p.get("lnoncollinear", False)
    ):
        raise ValueError("Reference contradicts neutral singlet scope")
    for key in ("efield", "efield_pead", "magmom"):
        if np.any(np.asarray(p.get(key, 0), dtype=float) != 0):
            raise ValueError(
                "Reference introduces spin or field beyond supported scope"
            )
    if not np.isfinite(float(p["encut"])):
        raise ValueError("Invalid ENCUT")
    return dict(p, nsw=0, ibrion=-1)


def restore_label(original, order, payload, method):
    if payload.get("converged") is not True:
        raise ValueError("Unconverged VASP output rejected")
    order = np.asarray(order, int)
    if sorted(order.tolist()) != list(range(len(original))):
        raise ValueError("Invalid atom order mapping")
    if payload["numbers"] != original.numbers[order].tolist():
        raise ValueError("VASP species/order mismatch")
    f = np.asarray(payload["forces"], float)
    energy = float(payload["energy"])
    if (
        f.shape != (len(original), 3)
        or not np.isfinite(f).all()
        or not np.isfinite(energy)
    ):
        raise ValueError("Invalid reference E/F")
    a = original.copy()
    a.calc = None
    a.arrays["REF_forces"] = np.empty_like(f)
    a.arrays["REF_forces"][order] = f
    a.info.update(
        REF_energy=energy,
        reference_id=method,
        label_source="vasp",
        energy_convention="free_energy",
    )
    return a


def slurm_script(command, workdir, jobname="label", cores=4, module=None):
    # User-controlled executable/config paths are shell-quoted; job names are fixed.
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={jobname}",
        "#SBATCH --partition=debug",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={int(cores)}",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --time=01:00:00",
        "#SBATCH --mem=8G",
        f"#SBATCH --output={workdir}/slurm-%j.out",
        f"#SBATCH --error={workdir}/slurm-%j.err",
        "set -euo pipefail",
        f"cd {shlex.quote(str(workdir))}",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "export PYTHONDONTWRITEBYTECODE=1",
        'export MPLCONFIGDIR="$PWD/cache/matplotlib" XDG_CACHE_HOME="$PWD/cache"',
    ]
    if module:
        lines.extend(
            [
                "if ! type module >/dev/null 2>&1; then",
                "  for init in /opt/modules/init/bash /usr/share/Modules/init/bash /etc/profile.d/modules.sh; do",
                '    if [ -f "$init" ]; then source "$init"; break; fi',
                "  done",
                "fi",
                f"module load {shlex.quote(module)}",
            ]
        )
    lines.append(" ".join(shlex.quote(str(x)) for x in command))
    return "\n".join(lines) + "\n"


def prepare_vasp(input_path, reference, out, python=None):
    from ase.calculators.vasp import Vasp

    params = validate_reference(reference)
    frames = read(input_path, ":")
    if not frames:
        raise ValueError("Empty candidate file")
    root = fresh_dir(out)
    method = fingerprint(reference)
    tasks = []
    dump(root / "reference.json", reference)
    package = Path(__file__).resolve().parents[1]
    for i, a in enumerate(frames):
        if (
            total_charge(a) != 0
            or a.info.get("spin", 1) != 1
            or np.any(a.info.get("external_field", [0, 0, 0]))
        ):
            raise ValueError(
                "Default VASP labeling supports neutral singlets without external field only"
            )
        if np.any(a.pbc) and not np.all(a.pbc):
            raise ValueError("Mixed PBC unsupported")
        if np.any(a.get_initial_magnetic_moments()):
            raise ValueError("Initial magnetic moments outside singlet scope")
        task = fresh_dir(root / f"frame_{i:05d}")
        original = a.copy()
        original.calc = None
        original.info.setdefault("group", f"input_{Path(input_path).resolve()}")
        write_frames(task / "original.extxyz", [original])
        order = np.argsort(a.get_chemical_symbols(), kind="stable")
        cell = a[order]
        cell.calc = None
        if not np.any(a.pbc):
            cell.center(vacuum=float(reference.get("vacuum_A", 10.0)))
        if abs(np.linalg.det(cell.cell.array)) < 1e-6:
            raise ValueError("Cell has zero volume")
        cell.pbc = True
        write_frames(task / "vasp_atoms.extxyz", [cell])
        calc = Vasp(directory=str(task), **params)
        # Generates real inputs using the locally licensed pseudopotentials.
        calc.write_input(cell)
        meta = {
            "original_hash": structure_id(original),
            "input_hashes": {
                name: hashlib.sha256((task / name).read_bytes()).hexdigest()
                for name in ("INCAR", "POSCAR", "KPOINTS", "POTCAR")
            },
            "order": order.tolist(),
            "reference_hash": method,
            "reference": reference,
            "prepared_positions": cell.positions.tolist(),
            "prepared_cell": cell.cell.array.tolist(),
        }
        dump(task / "task.json", meta)
        command = [
            python or sys.executable,
            str(package / "scripts/run_cli.py"),
            "run-vasp",
            str(task.resolve()),
        ]
        (task / "job.slurm").write_text(
            slurm_script(
                command,
                task.resolve(),
                cores=reference.get("cores", 4),
                module=reference.get("module"),
            )
        )
        tasks.append(str(task.relative_to(root)))
    dump(
        root / "manifest.json",
        {
            "reference_hash": method,
            "tasks": tasks,
            "source": str(Path(input_path).resolve()),
            "submitted": False,
        },
    )
    return root


def run_vasp(task):
    from ase.calculators.vasp import Vasp

    require_compute()
    task = Path(task).resolve()
    meta = read_json(task / "task.json")
    if (task / "label.json").exists() or (task / "OUTCAR").exists():
        raise FileExistsError(
            "VASP task already has results; preserve it and prepare a new attempt"
        )
    from .jobs import claim_task

    claim_task(task)
    if meta["reference_hash"] != fingerprint(meta["reference"]):
        raise ValueError("Reference settings were modified")
    for name, digest in meta["input_hashes"].items():
        if hashlib.sha256((task / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Prepared VASP inputs were modified")
    atoms = read(task / "vasp_atoms.extxyz")
    params = validate_reference(meta["reference"])
    if not np.allclose(
        atoms.positions, meta["prepared_positions"], atol=1e-8, rtol=0
    ) or not np.allclose(atoms.cell.array, meta["prepared_cell"], atol=1e-8, rtol=0):
        raise ValueError("Prepared atoms changed")
    command = meta["reference"].get("command", "mpiexec vasp_std")
    calc = Vasp(directory=str(task), command=command, **params)
    atoms.calc = calc
    try:
        energy = atoms.get_potential_energy(force_consistent=True)
        forces = atoms.get_forces(apply_constraint=False)
        payload = {
            "energy": float(energy),
            "forces": forces.tolist(),
            "numbers": atoms.numbers.tolist(),
            "converged": bool(calc.converged),
            "reference_hash": meta["reference_hash"],
            "positions": atoms.positions.tolist(),
            "cell": atoms.cell.array.tolist(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        }
        if not payload["converged"]:
            raise ValueError("VASP SCF did not converge")
        dump(task / "label.json", payload)
    except Exception as exc:
        dump(task / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def collect(tasks_path, out):
    root = Path(tasks_path)
    manifest = read_json(root / "manifest.json")
    frames = []
    rejected = []
    for name in manifest["tasks"]:
        task = root / name
        try:
            meta = read_json(task / "task.json")
            label = read_json(task / "label.json")
            if (
                label.get("reference_hash") != manifest["reference_hash"]
                or meta["reference_hash"] != manifest["reference_hash"]
            ):
                raise ValueError("Reference method mismatch")
            if not np.allclose(
                label["positions"], meta["prepared_positions"], atol=1e-6, rtol=0
            ) or not np.allclose(
                label["cell"], meta["prepared_cell"], atol=1e-6, rtol=0
            ):
                raise ValueError("Labeled geometry changed")
            original = read(task / "original.extxyz")
            if structure_id(original) != meta["original_hash"]:
                raise ValueError(
                    "Original structure was modified after task preparation"
                )
            frames.append(
                restore_label(
                    original, meta["order"], label, manifest["reference_hash"]
                )
            )
        except (OSError, KeyError, ValueError) as exc:
            rejected.append({"task": name, "reason": str(exc)})
    directory = fresh_dir(out)
    if frames:
        write_frames(directory / "labeled.extxyz", frames)
    dump(
        directory / "collection.json",
        {
            "accepted": len(frames),
            "rejected": rejected,
            "reference_hash": manifest["reference_hash"],
        },
    )
    return directory
