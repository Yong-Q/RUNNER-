import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read
from rnal.common import dump, write_frames, structure_id
from rnal.labeling import collect
from rnal.training import prepare_training


class IntegrityTests(unittest.TestCase):
    def frames(self):
        result = []
        for i in range(6):
            a = Atoms(
                "CO2",
                positions=[[0, 0, 0], [0, 0, 1.15 + i * 0.01], [0, 0, -1.15]],
                cell=[12] * 3,
            )
            a.info.update(
                group=f"test_only_{i // 2}",
                REF_energy=-1.0 - i * 0.01,
                reference_id="unit_test_fake_method",
                label_source="vasp",
                energy_convention="free_energy",
            )
            a.arrays["REF_forces"] = np.full((3, 3), i * 0.001)
            result.append(a)
        return result

    def test_training_rejects_unlabeled_or_mixed_reference(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            a = self.frames()
            a[0].info["reference_id"] = "different"
            write_frames(p / "data.xyz", a)
            with self.assertRaises(ValueError):
                prepare_training(p / "data.xyz", {"backend": "mace"}, p / "out")
            self.assertFalse((p / "out").exists())

    def test_training_has_independent_seeds_and_heldout_groups(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            write_frames(p / "data.xyz", self.frames())
            out = prepare_training(
                p / "data.xyz", {"backend": "mace", "seeds": [7, 19]}, p / "training"
            )
            m = json.loads((out / "manifest.json").read_text())
            self.assertFalse(set(m["train_groups"]) & set(m["valid_groups"]))
            for seed in [7, 19]:
                args = json.loads((out / f"seed_{seed}/training_args.json").read_text())
                self.assertEqual(args["seed"], seed)
                self.assertEqual(args["valid_file"], str(out / "valid.extxyz"))
                script = (out / f"seed_{seed}/job.slurm").read_text()
                self.assertIn("#SBATCH", script)
                self.assertNotIn("#PBS", script)

    def test_runner_export_preserves_energy_force_units(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            frames = self.frames()
            write_frames(p / "data.xyz", frames)
            (p / "input.nn").write_text(
                "nnp_generation 2\nrunner_mode train\nelements C O\ntest_fraction 0.1\n"
            )
            cfg = {
                "backend": "runner",
                "input_nn_template": str(p / "input.nn"),
                "executable": "/usr/bin/true",
                "seeds": [7, 19],
            }
            out = prepare_training(p / "data.xyz", cfg, p / "training")
            expected = read(out / "train.extxyz", ":")
            actual = read(out / "seed_7/input.data", ":", format="runnerdata")
            for a, b in zip(expected, actual):
                self.assertAlmostEqual(
                    a.info["REF_energy"], b.get_potential_energy(), places=6
                )
                np.testing.assert_allclose(
                    a.arrays["REF_forces"], b.get_forces(), atol=1e-7
                )
            self.assertIn("test_fraction 0.0", (out / "seed_7/input.nn").read_text())

    def test_4g_training_requires_explicit_staged_workflow(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            write_frames(p / "data.xyz", self.frames())
            (p / "input.nn").write_text("nnp_generation 4\n")
            with self.assertRaisesRegex(ValueError, "charge-label"):
                prepare_training(
                    p / "data.xyz",
                    {"backend": "runner", "input_nn_template": str(p / "input.nn")},
                    p / "out",
                )
            self.assertFalse((p / "out").exists())

    def test_collect_rejects_changed_geometry_and_nonconvergence(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            root = p / "tasks"
            root.mkdir()
            a = Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.1]], cell=[10] * 3)
            a.info["group"] = "unit_test"
            for i in range(3):
                d = root / str(i)
                d.mkdir()
                write_frames(d / "original.extxyz", [a])
                dump(
                    d / "task.json",
                    {
                        "original_hash": structure_id(a),
                        "order": [0, 1],
                        "reference_hash": "x",
                        "prepared_positions": a.positions.tolist(),
                        "prepared_cell": a.cell.array.tolist(),
                    },
                )
                positions = a.positions.copy()
                if i == 1:
                    positions[1, 2] += 0.1
                dump(
                    d / "label.json",
                    {
                        "converged": i != 2,
                        "reference_hash": "x",
                        "energy": -2.0,
                        "forces": np.zeros((2, 3)).tolist(),
                        "numbers": [7, 7],
                        "positions": positions.tolist(),
                        "cell": a.cell.array.tolist(),
                    },
                )
            dump(
                root / "manifest.json",
                {"tasks": ["0", "1", "2"], "reference_hash": "x"},
            )
            out = collect(root, p / "out")
            result = json.loads((out / "collection.json").read_text())
            self.assertEqual(result["accepted"], 1)
            self.assertEqual(len(result["rejected"]), 2)


if __name__ == "__main__":
    unittest.main()


class ReviewRegressionTests(unittest.TestCase):
    frames = IntegrityTests.frames

    def test_nonfinite_training_labels_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            frames = self.frames()
            frames[0].info["REF_energy"] = float("nan")
            write_frames(p / "data.xyz", frames)
            with self.assertRaises(ValueError):
                prepare_training(p / "data.xyz", {"backend": "mace"}, p / "out")
            self.assertFalse((p / "out").exists())

    def test_duplicate_geometry_across_groups_rejected(self):
        from rnal.common import split_groups

        a = self.frames()[0]
        b = a.copy()
        b.info["group"] = "different"
        with self.assertRaises(ValueError):
            split_groups([a, b])

    def test_reference_cannot_override_charge_spin_or_field(self):
        from rnal.labeling import validate_reference

        for override in (
            {"nelect": 10},
            {"ispin": 2},
            {"efield": 0.1},
            {"efield_pead": [0, 0, 1]},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    validate_reference(
                        {
                            "method_id": "x",
                            "vasp_parameters": dict(xc="PBE", encut=520, **override),
                        }
                    )


class ExecutionTests(unittest.TestCase):
    def test_second_launch_preserves_first_result(self):
        import sys
        from rnal.jobs import run_claimed

        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            command = [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('sentinel').write_text('first')",
            ]
            run_claimed(p, command)
            with self.assertRaises(FileExistsError):
                run_claimed(p, command)
            self.assertEqual((p / "sentinel").read_text(), "first")


class OriginalStructureTests(unittest.TestCase):
    def test_changed_original_geometry_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            root = p / "tasks"
            task = root / "one"
            task.mkdir(parents=True)
            a = Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.1]], cell=[10] * 3)
            a.info["group"] = "trajectory"
            dump(
                task / "task.json",
                {
                    "original_hash": structure_id(a),
                    "order": [0, 1],
                    "reference_hash": "x",
                    "prepared_positions": a.positions.tolist(),
                    "prepared_cell": a.cell.array.tolist(),
                },
            )
            dump(
                task / "label.json",
                {
                    "converged": True,
                    "reference_hash": "x",
                    "energy": -2.0,
                    "forces": np.zeros((2, 3)).tolist(),
                    "numbers": [7, 7],
                    "positions": a.positions.tolist(),
                    "cell": a.cell.array.tolist(),
                },
            )
            a.positions[1, 2] = 1.2
            write_frames(task / "original.extxyz", [a])
            dump(root / "manifest.json", {"tasks": ["one"], "reference_hash": "x"})
            out = collect(root, p / "collected")
            result = json.loads((out / "collection.json").read_text())
            self.assertEqual(result["accepted"], 0)
            self.assertIn("modified", result["rejected"][0]["reason"])


class SourceGroupTests(unittest.TestCase):
    def test_ungrouped_frames_from_one_file_stay_together(self):
        from unittest.mock import patch
        from rnal.labeling import prepare_vasp

        def fake_input(calc, atoms):
            for name in ["INCAR", "POSCAR", "KPOINTS", "POTCAR"]:
                (Path(calc.directory) / name).write_text("unit-test-only input fixture")

        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            a = Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.1]], cell=[10] * 3)
            b = a.copy()
            b.positions[1, 2] = 1.12
            write_frames(p / "trajectory.xyz", [a, b])
            with patch(
                "ase.calculators.vasp.Vasp.write_input",
                autospec=True,
                side_effect=fake_input,
            ):
                out = prepare_vasp(
                    p / "trajectory.xyz",
                    {
                        "method_id": "unit-test",
                        "vasp_parameters": {"xc": "PBE", "encut": 520},
                    },
                    p / "tasks",
                )
            first = read(out / "frame_00000/original.extxyz")
            second = read(out / "frame_00001/original.extxyz")
            self.assertEqual(first.info["group"], second.info["group"])


class FineTuneSettingsTests(unittest.TestCase):
    frames = IntegrityTests.frames

    def test_finetune_learning_rate_reaches_cli(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            write_frames(p / "data.xyz", self.frames())
            out = prepare_training(
                p / "data.xyz",
                {
                    "backend": "mace",
                    "seeds": [7, 19],
                    "lr": 1e-4,
                    "epochs": 2,
                    "plot": False,
                },
                p / "training",
            )
            args = json.loads((out / "seed_7/training_args.json").read_text())
            self.assertEqual(args["lr"], 1e-4)
            self.assertEqual(args["max_num_epochs"], 2)
            self.assertFalse(args["plot"])


class RunnerTemplateTests(unittest.TestCase):
    def test_default_feature_map_rejects_dedicated_symfunctions(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            write_frames(p / "data.xyz", IntegrityTests.frames(self))
            (p / "input.nn").write_text(
                "nnp_generation 2\nfeature_map_default\nsymfunction_short C 2 O 0.01 0.0 1\n"
            )
            with self.assertRaisesRegex(ValueError, "feature_map_default"):
                prepare_training(
                    p / "data.xyz",
                    {
                        "backend": "runner",
                        "input_nn_template": str(p / "input.nn"),
                        "executable": "/usr/bin/true",
                    },
                    p / "out",
                )


class AssembleRunnerTests(unittest.TestCase):
    def test_assembly_checks_success_and_shared_scaling(self):
        from rnal.training import assemble_runner

        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            train = p / "train"
            train.mkdir()
            dump(train / "manifest.json", {"backend": "runner", "seeds": [7, 19]})
            for seed in [7, 19]:
                d = train / f"seed_{seed}"
                d.mkdir()
                dump(d / "exit_status.json", {"returncode": 0})
                (d / "input.nn").write_text(
                    "nnp_generation 2\nrunner_mode train\nelements C O\ninitialization_method uniform\ndont_use_old_scaling\ndont_use_old_weights\n"
                )
                (d / "scaling.data").write_text("same scaling")
                for z in [6, 8]:
                    (d / f"opt.weights_short.{z:03d}.out").write_text(
                        f"test weights {seed} {z}"
                    )
            assemble_runner(train, p / "committee", "/path/libRuNNer.so")
            self.assertTrue((p / "committee/1/weights_short.006.data").exists())
            self.assertNotIn(
                "dont_use_old_scaling", (p / "committee/input.nn").read_text()
            )
            self.assertNotIn(
                "dont_use_old_weights", (p / "committee/input.nn").read_text()
            )
            self.assertIn(
                "num_committee_members 2", (p / "committee/input.nn").read_text()
            )
            (train / "seed_19/scaling.data").write_text("different scaling")
            with self.assertRaises(ValueError):
                assemble_runner(train, p / "bad", "/path/libRuNNer.so")
