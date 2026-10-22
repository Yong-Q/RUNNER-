from pathlib import Path
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from .common import force_disagreement, require_compute, total_charge


def load_backend(config):
    require_compute()
    kind = config["backend"]
    if kind in ("mace", "polar"):
        import torch
        from mace.calculators import MACECalculator

        torch.set_num_threads(int(config.get("threads", 4)))
        models = [
            torch.load(p, map_location=config.get("device", "cpu"), weights_only=False)
            for p in config["models"]
        ]
        calc = MACECalculator(
            models=models,
            model_type="PolarMACE" if kind == "polar" else "MACE",
            device=config.get("device", "cpu"),
            default_dtype=config.get("dtype", "float64"),
        )
    elif kind == "runner":
        from runnerase_prediction import RuNNerlib

        calc = RuNNerlib(
            path=str(Path(config["potential"]).resolve()),
            runner_library=str(Path(config["library"]).resolve()),
        )
    else:
        raise ValueError("backend must be mace, polar or runner")
    return Committee(calc, kind)


class Committee(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, inner, kind):
        super().__init__()
        self.inner = inner
        self.kind = kind

    def calculate(
        self, atoms=None, properties=("energy", "forces"), system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        if np.any(atoms.pbc) and not np.all(atoms.pbc):
            raise ValueError(
                "This gas workflow supports fully periodic or fully isolated systems only"
            )
        if self.kind == "runner":
            if atoms.info.get("spin", 1) != 1 or np.any(
                atoms.info.get("external_field", [0, 0, 0])
            ):
                raise ValueError(
                    "RuNNer adapter does not support explicit spin or external-field inputs"
                )
            from ase import units
            from runner_interface.generic import call_predict_structure

            charge = total_charge(atoms)
            if atoms.pbc.any() and abs(charge) > 1e-10:
                raise ValueError("Charged periodic RuNNer systems are unsupported")
            es, fs, _ = call_predict_structure(
                self.inner.lib,
                self.inner.ncommittee_members,
                len(atoms),
                atoms.positions / units.Bohr,
                atoms.numbers,
                total_charge=charge,
                lattice=atoms.cell.array / units.Bohr if atoms.pbc.all() else None,
            )
            es = np.asarray(es) * units.Hartree
            fs = np.asarray(fs) * units.Hartree / units.Bohr
        else:
            inp = atoms.copy()
            inp.info["charge"] = total_charge(atoms)
            self.inner.calculate(inp, properties=["energy", "forces"])
            es = np.asarray(
                self.inner.results.get("energy_comm", [self.inner.results["energy"]])
            )
            fs = np.asarray(
                self.inner.results.get("forces_comm", [self.inner.results["forces"]])
            )
        if not np.isfinite(es).all() or not np.isfinite(fs).all():
            raise ValueError("Non-finite model prediction")
        self.results = {
            "energy": float(es.mean()),
            "forces": fs.mean(axis=0),
            "energy_members": es.copy(),
            "forces_members": fs.copy(),
        }
        if len(fs) >= 2:
            self.results["force_sigma"] = force_disagreement(fs)
