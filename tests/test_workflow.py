import tempfile
import unittest
from pathlib import Path
from ase import Atoms
from rnal.common import force_disagreement, fresh_dir, select_frames, split_groups
from rnal.labeling import validate_reference, restore_label


class WorkflowTests(unittest.TestCase):
    def test_force_disagreement_population_vector(self):
        self.assertEqual(force_disagreement([[[1, 0, 0]], [[3, 0, 0]]]).tolist(), [1.0])
        with self.assertRaises(ValueError):
            force_disagreement([[[float("nan"), 0, 0]], [[0, 0, 0]]])
        with self.assertRaises(ValueError):
            force_disagreement([[[0, 0, 0]]])

    def test_existing_results_are_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            p = fresh_dir(Path(t) / "round")
            (p / "sentinel").write_text("keep")
            with self.assertRaises(FileExistsError):
                fresh_dir(p)
            self.assertEqual((p / "sentinel").read_text(), "keep")

    def test_selection_deduplicates_before_budget(self):
        a = Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.1]])
        a.info.update(force_sigma_max=0.5, group="traj1")
        b = a.copy()
        b.info["force_sigma_max"] = 0.7
        c = a.copy()
        c.positions[1, 2] = 1.2
        c.info["force_sigma_max"] = 0.2
        chosen = select_frames(
            [a, b, c], minimum=0.1, maximum=1.0, budget=2, min_step_gap=0
        )
        self.assertEqual(len(chosen), 2)
        self.assertEqual(chosen[0].info["force_sigma_max"], 0.7)

    def test_group_split_no_trajectory_leakage(self):
        frames = []
        for g in range(5):
            for step in range(3):
                a = Atoms("H", positions=[[g + step * 0.01, 0, 0]])
                a.info.update(group=str(g), step=step)
                frames.append(a)
        train, valid = split_groups(frames, 0.2, 42)
        self.assertFalse(
            {a.info["group"] for a in train} & {a.info["group"] for a in valid}
        )
        self.assertEqual(len(train) + len(valid), 15)
        with self.assertRaises(ValueError):
            split_groups(frames[:3], 0.2, 42)

    def test_missing_dft_settings_and_ionic_md_rejected(self):
        with self.assertRaises(ValueError):
            validate_reference({"vasp_parameters": {}})
        with self.assertRaises(ValueError):
            validate_reference(
                {
                    "method_id": "x",
                    "vasp_parameters": {"xc": "PBE", "encut": 500, "nsw": 10},
                }
            )

    def test_labels_restore_atomic_order_and_reject_unconverged(self):
        a = Atoms("OH2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[10] * 3)
        order = [1, 2, 0]
        payload = {
            "converged": True,
            "energy": -10.0,
            "forces": [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
            "numbers": [1, 1, 8],
        }
        b = restore_label(a, order, payload, "method")
        self.assertEqual(b.arrays["REF_forces"][:, 0].tolist(), [3, 1, 2])
        self.assertEqual(b.info["REF_energy"], -10.0)
        payload["converged"] = False
        with self.assertRaises(ValueError):
            restore_label(a, order, payload, "method")
        payload["converged"] = True
        payload["numbers"] = [8, 1, 1]
        with self.assertRaises(ValueError):
            restore_label(a, order, payload, "method")


if __name__ == "__main__":
    unittest.main()


class ChargeTests(unittest.TestCase):
    def test_charge_metadata_is_not_silently_lost(self):
        from rnal.common import total_charge

        a = Atoms("Ag3")
        a.info["total_charge"] = -1
        self.assertEqual(total_charge(a), -1)
        a.info["charge"] = 0
        with self.assertRaises(ValueError):
            total_charge(a)
