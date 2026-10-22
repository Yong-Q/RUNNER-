import re
import sys
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from .common import fresh_dir, dump, split_groups, write_frames
from .labeling import slurm_script


def prepare_training(data, config, out):
    frames = read(data, ":")
    if not frames or any(
        "REF_energy" not in a.info or "REF_forces" not in a.arrays for a in frames
    ):
        raise ValueError("Training requires labeled REF_energy and REF_forces")
    for a in frames:
        f = np.asarray(a.arrays["REF_forces"])
        if (
            not np.isfinite(float(a.info["REF_energy"]))
            or f.shape != (len(a), 3)
            or not np.isfinite(f).all()
        ):
            raise ValueError("Invalid or non-finite training labels")
        if a.info.get("energy_convention") != "free_energy":
            raise ValueError("Expected force-consistent free_energy labels")
    methods = {a.info.get("reference_id") for a in frames}
    if None in methods or len(methods) != 1:
        raise ValueError("One consistent reference_id is required")
    if any(a.info.get("label_source") != "vasp" for a in frames):
        raise ValueError("Production training accepts VASP labels only")
    train, valid = split_groups(
        frames, config.get("valid_fraction", 0.2), config.get("split_seed", 42)
    )
    seeds = config.get("seeds", [17, 29, 43])
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Need at least 2 distinct member seeds")
    backend = config["backend"]
    if backend == "runner":
        template = Path(config["input_nn_template"]).read_text()
        if re.search(
            r"^\s*feature_map_default\s*(?:#.*)?$", template, re.M
        ) and re.search(r"^\s*symfunction_\w+\s", template, re.M):
            raise ValueError(
                "feature_map_default requires symfunction without a dedicated suffix"
            )
        generation = re.search(r"^\s*nnp_generation\s+(\d+)", template, re.M)
        if not generation:
            raise ValueError("RuNNer template must declare nnp_generation")
        if int(generation[1]) >= 3:
            raise ValueError(
                "This initial training adapter supports 2G only; 3G/4G require a separately validated charge-label and staged training workflow"
            )
    directory = fresh_dir(out).resolve()
    write_frames(directory / "train.extxyz", train)
    write_frames(directory / "valid.extxyz", valid)
    models = []
    for seed in seeds:
        member = fresh_dir(directory / f"seed_{seed}")
        if backend in ("mace", "polar"):
            name = f"committee_{seed}"
            modeldir = member / "models"
            args = {
                "name": name,
                "model": "PolarMACE" if backend == "polar" else "MACE",
                "train_file": str(directory / "train.extxyz"),
                "valid_file": str(directory / "valid.extxyz"),
                "energy_key": "REF_energy",
                "forces_key": "REF_forces",
                "total_charge_key": "charge",
                "total_spin_key": "spin",
                "seed": seed,
                "device": config.get("device", "cpu"),
                "default_dtype": "float64",
                "batch_size": config.get("batch_size", 2),
                "valid_batch_size": 2,
                "max_num_epochs": config.get("epochs", 100),
                "lr": float(
                    config.get("lr", 1e-4 if config.get("foundation_model") else 1e-3)
                ),
                "plot": bool(config.get("plot", False)),
                "E0s": config.get("E0s", "average"),
                "loss": "weighted",
                "energy_weight": 1.0,
                "forces_weight": 100.0,
                "model_dir": str(modeldir),
                "checkpoints_dir": str(member / "checkpoints"),
                "log_dir": str(member / "logs"),
                "results_dir": str(member / "results"),
            }
            if config.get("foundation_model"):
                args["foundation_model"] = str(
                    Path(config["foundation_model"]).resolve()
                )
            elif backend == "polar":
                raise ValueError("Polar fine-tuning requires foundation_model")
            command = [
                config.get("python", sys.executable),
                "-m",
                "mace.cli.run_train",
            ] + [f"--{k}={v}" for k, v in args.items()]
            dump(member / "training_args.json", args)
            models.append(str(modeldir / f"{name}.model"))
        elif backend == "runner":
            labeled = []
            for a in train:
                b = a.copy()
                b.calc = SinglePointCalculator(
                    b, energy=a.info["REF_energy"], forces=a.arrays["REF_forces"]
                )
                labeled.append(b)
            write(member / "input.data", labeled, format="runnerdata")
            text = template
            for key, value in [
                ("runner_mode", "train"),
                ("random_seed", str(seed)),
                ("test_fraction", "0.0"),
            ]:
                text = (
                    re.sub(
                        r"^\s*" + key + r"\s+.*$", key + " " + value, text, flags=re.M
                    )
                    if re.search(r"^\s*" + key + r"\s+", text, re.M)
                    else text + "\n" + key + " " + value + "\n"
                )
            (member / "input.nn").write_text(text)
            command = [str(Path(config["executable"]).resolve()), str(member)]
        else:
            raise ValueError("Unknown training backend")
        wrapper = [
            config.get("python", sys.executable),
            str(Path(__file__).resolve().parents[1] / "scripts/run_cli.py"),
            "run-job",
            str(member),
            "--",
        ] + command
        script = slurm_script(wrapper, member, jobname="train_committee", cores=1)
        cpus = int(config.get("threads", 4))
        script = script.replace(
            "#SBATCH --cpus-per-task=1", f"#SBATCH --cpus-per-task={cpus}"
        ).replace(
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
            f"export OMP_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS={cpus}",
        )
        (member / "job.slurm").write_text(script)
    dump(
        directory / "manifest.json",
        {
            "backend": backend,
            "seeds": seeds,
            "train": len(train),
            "valid": len(valid),
            "reference_id": next(iter(methods)),
            "status": "prepared_not_trained",
            "train_groups": sorted({str(a.info["group"]) for a in train}),
            "valid_groups": sorted({str(a.info["group"]) for a in valid}),
            "runner_validation": "held-out valid.extxyz is excluded from native training; evaluate separately",
        },
    )
    if models:
        dump(
            directory / "next_committee.json",
            {
                "backend": backend,
                "models": models,
                "device": config.get("device", "cpu"),
                "dtype": "float64",
                "threads": config.get("threads", 4),
                "uncertainty_calibrated": False,
            },
        )
    return directory


def assemble_runner(training_path, out, library):
    """Assemble completed 2G fits only when they share the same descriptor scaling."""
    import hashlib
    import shutil
    from .common import read_json

    training_path = Path(training_path).resolve()
    manifest = read_json(training_path / "manifest.json")
    if manifest["backend"] != "runner" or len(manifest["seeds"]) < 2:
        raise ValueError("Need a trained RuNNer committee")
    members = [training_path / f"seed_{s}" for s in manifest["seeds"]]
    reference_scaling = (members[0] / "scaling.data").read_bytes()
    reference_input = (members[0] / "input.nn").read_text()
    normalized = lambda s: re.sub(
        r"^\s*random_seed\s+.*$", "random_seed <member>", s, flags=re.M
    )
    if not re.search(r"^\s*nnp_generation\s+2\s*$", reference_input, re.M):
        raise ValueError("Assembly currently supports 2G short-range fits only")
    weights = []
    for member in members:
        if read_json(member / "exit_status.json")["returncode"] != 0:
            raise ValueError("A training member failed")
        if (member / "scaling.data").read_bytes() != reference_scaling:
            raise ValueError("Member descriptor scaling differs")
        if normalized((member / "input.nn").read_text()) != normalized(reference_input):
            raise ValueError("Member architectures differ")
        paths = sorted(member.glob("opt.weights_short.*.out"))
        if not paths:
            raise ValueError("No fitted short-range weights found")
        weights.append(paths)
    if any([p.name for p in w] != [p.name for p in weights[0]] for w in weights):
        raise ValueError("Member species differ")
    directory = fresh_dir(out).resolve()
    text = re.sub(
        r"^\s*dont_use_old_(?:scaling|weights)\b[^\n]*\n?",
        "",
        reference_input,
        flags=re.M,
    )
    for key, value in [
        ("runner_mode", "predict"),
        ("num_committee_members", str(len(members))),
        ("initialization_method", "read"),
    ]:
        text = (
            re.sub(r"^\s*" + key + r"\s+.*$", key + " " + value, text, flags=re.M)
            if re.search(r"^\s*" + key + r"\s+", text, re.M)
            else text + "\n" + key + " " + value + "\n"
        )
    if not re.search(r"^\s*calculate_forces\b", text, re.M):
        text += "\ncalculate_forces\n"
    (directory / "input.nn").write_text(text)
    (directory / "scaling.data").write_bytes(reference_scaling)
    hashes = {}
    for i, paths in enumerate(weights, 1):
        member = fresh_dir(directory / str(i))
        for path in paths:
            target = member / path.name.replace("opt.", "", 1).replace(".out", ".data")
            shutil.copy2(path, target)
            hashes[str(target.relative_to(directory))] = hashlib.sha256(
                target.read_bytes()
            ).hexdigest()
    dump(
        directory / "committee.json",
        {
            "backend": "runner",
            "potential": str(directory),
            "library": str(Path(library).resolve()),
            "uncertainty_calibrated": False,
        },
    )
    dump(
        directory / "assembly.json",
        {
            "training_path": str(training_path),
            "seeds": manifest["seeds"],
            "weight_sha256": hashes,
            "scaling_sha256": hashlib.sha256(reference_scaling).hexdigest(),
            "not_validated_for_production": True,
        },
    )
    return directory
