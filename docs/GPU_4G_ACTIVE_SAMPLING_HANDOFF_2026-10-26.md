# GPU 4G + active sampling handoff (2026-10-26)

This note records the split between the RUNNER workflow and the GPU 4G
implementation so the two workstreams do not get conflated.

## Current remote tasks

Two single-GPU `queue004` tasks are submitted:

| Task | Purpose | Queue | State |
|---|---|---|---|
| `t-20261026135843-f6bq5` | Plan D expanded sampling + MACE-POLAR distillation | `queue004`, 1×A100-80G, 50 h | queued |
| `t-20261026144930-jw8c7` | RUNNER/4G committee active sampling + 4G distillation | `queue004`, 1×A100-80G, 50 h | queued |

Q4 was fully allocated (64/64, 0 free) when both were submitted, so both are
waiting for a GPU. The RUNNER task entrypoint is
`symmetrix-polar/scripts/remote_4g_al_round.sh`; it selects from the bulk pool
with the c1/c2/c3 committee, merges the selected frames, and fine-tunes the
fixed 4G student for 5 epochs.

## Workstream A: Plan D (MACE-POLAR distillation)

- Student: linear MACE, 128 channels, `L_max=1`, 2 interactions.
- Teacher: `MACE-POLAR-1-M` (FP64 labels `REF_energy` / `REF_forces`).
- Initialisation: MACE-OFF23-medium; this is **not** a PolarMACE-initialised
  student.
- Dataset: `data_20261025_al_full/train.xyz` = 23,376 base frames + 140 dense
  active-selected frames ×10 (six molecules), 24,776 frames total.
- Validation: `data_20261025/valid.xyz`, 3,465 frames.
- Prior accuracy: 1.24 meV/atom / 7.75 meV/Å on the full validation set.
- Dense active-frame audit: 1.2 meV/atom / 19.0 meV/Å; the ρ=0.02 slice is
  32.2 meV/Å, i.e. the highest-density states are still the weak point.
- Speed (5,184 atoms): FP32 direct/JIT trial medians 4.55–5.19 µs/atom/step;
  FP64 generic probe median 598, best 251 µs/atom/step. The project FP64
  policy is not yet satisfied by the generic path.
- Remote config: `bench/plan_d_al_2026-10-26/configs/student_al_full_remote.yaml`.

## Workstream B: RUNNER + GPU 4G

- RUNNER remains the Slurm/VASP committee workflow. It contains RuNNer **2G**
  data conversion; the 4G charge labels / two-stage training are not wired
  into RUNNER.
- The GPU 4G implementation is in `symmetrix-polar/plan_c_4g`:
  local descriptors → χ/J networks → periodic QEq/PME → adjoint force trick.
  It is an independent PyTorch implementation, not a port of the RuNNer2
  Fortran code.
- 4G speed: 13.45–19.26 µs/atom/step (FP64, 5,184 atoms, 72k parameters).
- 4G fixes in the current tree:
  - Gaussian self term included in the QEq operator and `E_elec`;
  - element-resolved Pearson χ/J initialisation;
  - charge surrogate / two-stage training interface;
  - PolarMACE l=0 charges calibrated by `4*pi` for reference labels.
- Fixed 4G result (self-energy + Pearson init, no charge stage):
  bulk 39.4 meV/atom / 77.5 meV/Å; gas 49.0 / 14.9; dense force 293.8 meV/Å
  (from 338.8). This is an ~8–13 % improvement, still well above the 5/30 gate.
- Remaining 4G physics gaps: charge-monopole-only electrostatics vs the
  teacher's l=1 multipoles, thin angular descriptor (8 G4-like functions,
  10 pair channels), and the incomplete charge-training stage.

## Next actions

1. Wait for Q4 headroom and run the queued Plan D task; report valid metrics.
2. Define the RUNNER / 4G active-sampling task entrypoint that combines
   `rnal` proposal/labelling with `plan_c_4g` GPU training; submit it as a
   **second** single-GPU Q4 task only after the first task starts or finishes.
3. For 4G accuracy, implement l=1 atomic multipoles and an element-resolved
   angular basis before another full AL round.

## References

- `symmetrix-polar/docs/results/PLAN_C_4G_STATUS_2026-10-25.md`
- `symmetrix-polar/docs/results/DATASET_REBUILD_2026-10-25.md`
- `symmetrix-polar/docs/research/RUNNER2_TRANSFERABLE_TECHNIQUES.md`
