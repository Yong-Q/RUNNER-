import time
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from .backends import load_backend
from .common import fresh_dir, dump, select_frames, write_frames, total_charge


def sample(config, input_path, index, out, steps=None):
    cfg = config.get("sampling", {})
    nsteps = cfg.get("steps", 100) if steps is None else steps
    if nsteps < 0:
        raise ValueError("steps must be nonnegative")
    directory = fresh_dir(out)
    dump(directory / "config.json", config)
    atoms = read(input_path, index=index)
    atoms.info.setdefault("charge", total_charge(atoms))
    atoms.info.setdefault("spin", 1)
    atoms.info.setdefault("external_field", np.zeros(3))
    calc = load_backend(config)
    atoms.calc = calc
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    temperature = float(cfg.get("temperature_K", 300))
    dt = float(cfg.get("timestep_fs", 0.25))
    interval = int(cfg.get("save_interval", 5))
    if dt <= 0 or interval < 1 or temperature <= 0:
        raise ValueError("Invalid MD settings")
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
    if len(atoms) > 1:
        Stationary(atoms)
    dyn = Langevin(
        atoms,
        timestep=dt * units.fs,
        temperature_K=temperature,
        friction=float(cfg.get("friction_per_fs", 0.01)) / units.fs,
        rng=rng,
        fixcm=False,
    )
    saved = []
    records = []
    stopped = None
    t0 = time.perf_counter()
    for step in range(nsteps + 1):
        if step:
            dyn.run(1)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        if "force_sigma" not in calc.results:
            raise ValueError("Sampling requires at least two committee members")
        sigma = float(calc.results["force_sigma"].max())
        if not np.isfinite(forces).all() or not np.isfinite(energy):
            raise ValueError("Non-finite MD state")
        stop = sigma > float(cfg.get("stop_sigma_eVA", 1.0))
        if step % interval == 0 or stop or step == nsteps:
            a = atoms.copy()
            a.calc = None
            a.info.update(
                group=directory.resolve().name,
                step=step,
                force_sigma_max=sigma,
                ml_energy=energy,
                uncertainty_calibrated=False,
                backend=config["backend"],
            )
            a.arrays["ml_forces"] = forces.copy()
            a.arrays["force_sigma"] = calc.results["force_sigma"].copy()
            saved.append(a)
            write(directory / "trajectory.extxyz", a, append=True, write_results=False)
            records.append(
                {
                    "step": step,
                    "energy_eV": energy,
                    "temperature_K": atoms.get_temperature(),
                    "force_sigma_max_eVA": sigma,
                }
            )
        if stop:
            stopped = "force_disagreement_above_stop"
            break
    selection = config.get("selection", {})
    chosen = select_frames(saved, **selection)
    if chosen:
        write_frames(directory / "candidates.extxyz", chosen)
    dump(
        directory / "summary.json",
        {
            "status": "stopped" if stopped else "completed",
            "reason": stopped,
            "steps_completed": dyn.nsteps,
            "frames": len(saved),
            "candidates": len(chosen),
            "elapsed_s": time.perf_counter() - t0,
            "calibrated_uncertainty": False,
            "records": records,
        },
    )
    return directory


def benchmark(config, input_path, index, out, warmup=2, repeats=5):
    if warmup < 0 or repeats < 1:
        raise ValueError("Invalid repeat counts")
    directory = fresh_dir(out)
    atoms = read(input_path, index=index)
    calc = load_backend(config)
    times = []
    for i in range(warmup + repeats):
        # Explicit calculate bypasses ASE unchanged-geometry caching.
        start = time.perf_counter()
        calc.calculate(atoms)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
    dump(
        directory / "benchmark.json",
        {
            "backend": config["backend"],
            "config": config,
            "atoms": len(atoms),
            "members": len(calc.results["energy_members"]),
            "warmup": warmup,
            "repeats": repeats,
            "seconds_per_call": times,
            "median_us_per_atom_per_step": float(np.median(times)) * 1e6 / len(atoms),
            "scope": "full committee energy/force call; backend may also compute stress; excludes MD integration and I/O",
            "same_geometry_repeated": True,
            "units": "us/atom/step",
        },
    )
    return directory
