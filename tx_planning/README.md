# tx_planning — transmitter-side planning on the RF-3DGS scene

RF-3DGS answers "what does the channel look like from a fixed transmitter?".
This folder answers the planner's question instead: **where should the
transmitter go?** It reuses the Sionna 2.1 scene the port built
(`sionna_port/`), runs on the Windows env `rf-sionna-win` (GPU OptiX), and
borrows Aerial Omniverse Digital Twin's data model — static radio units,
a grid of user positions, structured CIR output per (Tx, Rx) — rather than
pictures.

## Stages

| stage | script | status |
|---|---|---|
| 0 | `probe_differentiability.py` — is Sionna's solve differentiable in material and Tx position on this scene? | verified (gradient vs finite difference within 5 %) |
| 1 | `tx_sweep.py` — brute-force ground truth: candidate Tx positions × indoor Rx grid, structured CIRs | verified |
| 2 | `optimize_tx.py` — gradient ascent on a coverage objective, Tx position from Sionna's own gradient | verified |
| 3 | `fit_materials.py` — fit every material's scattering coefficient to PDPs observed from Tx A, score at held-out Tx B | verified |
| 4 | real scenes: geometry from gsplat / 2DGS instead of the Blender model | later |

## Running

```
set PYTHONUTF8=1
cd tx_planning
SCENE=../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1_sionna12.xml

python tx_sweep.py    --scene-xml %SCENE% --tx-grid-step 2 --rx-step 1
python optimize_tx.py --scene-xml %SCENE% --init-from ../output/tx_planning/tx_sweep.npz ^
                      --objective coverage --threshold-db -85 --tx-height 2.0 --rx-step 1
python fit_materials.py --scene-xml %SCENE% --steps 25 --samples 30000
```

`tx_sweep.npz` holds `tx_positions`, `rx_positions`, `gain_db [n_tx, n_rx]`,
`n_paths`, and per-pair ragged CIRs (`cir_a`, `cir_tau`, `cir_aod`, `cir_aoa`
as object arrays keyed by `cir_keys`), with the run config in `meta`. That is
the same content Aerial's `CIRResultsRequest` returns to its RAN simulator.

## What was learned building it (NIST lobby, 60 GHz, depth 1, scattering 0.7)

- **Indoor mask.** The bounding-box grid covers an L-shaped lobby; half the
  points lie outside the building and get zero paths. A point is indoor when a
  ray up hits a ceiling *and* a ray down hits a floor (`scene_common.indoor_mask`).
  On the 4 m smoke grid this agreed with "at least one path" at all 35 points.
- **Batch the receivers.** Sionna's cost is per source, not per receiver. One
  solve with all Rx in the scene gives the same gains as pairwise solves
  (max |Δ| = 0.0 dB) at ~12 Tx/s for 16 Rx versus 0.9 pairs/s — about 200×.
- **Monte-Carlo noise is not the problem.** At a fixed Tx the soft-coverage
  objective varies by ±0.0007 across seeds at 20 k samples (±0.0002 at 100 k).
  The 0.04 jumps seen while optimising are real: at 60 GHz a 0.3 m move flips
  line-of-sight for several grid points, so the landscape is piecewise smooth.
  400 k samples with 59 receivers *breaks* (per-Rx std up to 14 dB) — stay ≤ 100 k.
- **The gradient does not know about walls.** Unconstrained, the optimiser
  walked the Tx through the east wall at x = 8.48 m (no ceiling above → no
  paths → drjit reduce over an empty axis). Each step is now accepted only if
  the new point is indoor with ≥ `--margin` (0.2 m) clearance; a blocked step
  first drops its wall-normal component so the Tx slides along the wall.
- **Sweep, then refine.** Starting the gradient from the sweep's best cell
  (`--init-from`) and sliding along the wall took hard coverage (> −85 dB on a
  2 m grid, 59 points) from 45.8 % to 59.3 % in 25 steps / 27 s.
- **AD flags.** Reverse mode needs `solver.loop_mode = "evaluated"` and
  `dr.set_flag(dr.JitFlag.SpillToSharedMemory, False)` (`scene_common.enable_reverse_mode`).

## Stage 3: learning the materials from one transmitter (`fit_materials.py`)

Setup: each of the 29 used materials gets a hidden scattering coefficient in
[0.2, 0.9]; the "measurement" is the delay-binned PDP (16 × 10 ns, +1 dB
noise) at the 59 indoor grid points from Tx A = (8, −7, 2). Every material
starts at 0.5. The fitted twin is scored at Tx B = (0, −3, 2), which it never saw.

| | RMSE at Tx A | RMSE at held-out Tx B |
|---|---|---|
| all materials at 0.5 | 1.53 dB | 1.09 dB |
| fitted from Tx A | 0.98 dB (= the 1 dB noise) | **0.47 dB** |

- **Total power is the wrong observation.** It reached the same Tx A error, and
  even 0.44 dB at Tx B, with material estimates that were nonsense (a glass at
  0.80 fitted to 0.01, a wood at 0.89 to 0.12): the split between specular and
  diffuse barely changes the total, but it reshapes the PDP (spike vs tail).
- **Adam is the wrong optimiser here.** It moves every parameter at the same
  rate, so the materials the data does not constrain wander at random.
  Normalised gradient descent (step ∝ gradient / largest gradient) moves a
  material in proportion to how much the data cares: with it, the 7 materials
  whose mean gradient is ≥ 2 % of the largest went from mean |error| 0.161 to
  **0.025** (concrete 0.71 → 0.74, metal 0.22 → 0.21, glass 0.80 → 0.85, …),
  and the other 22 stayed within 0.02 of the prior.
- **Identifiability is the real output.** 12 materials receive exactly zero
  gradient from Tx A (never on a depth-1 path); the gradient magnitudes say
  which walls a second transmitter position would have to illuminate — i.e.
  where to measure next. That is the active-measurement question a planner can
  ask that RadioSight's per-AP twin cannot.
