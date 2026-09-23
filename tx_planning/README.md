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
| 3b | `active_measurement.py` — pick the second transmitter position by the twin's own sensitivity, before measuring | verified |
| 3c | `guided_placement.py` — placement by coarse solves plus an analytic line-of-sight guide on the fine grid (DLSS-style reconstruction), `guided_refine.py` — the gradient from its cell | verified (lobby, corridor) |
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
python active_measurement.py --scene-xml %SCENE%
```

The dashboard picks these outputs up with `--planning-dir`
(`planning_panels.py` renders the section; first file name found per kind wins):

```
python sionna_port/dashboard.py output/ablation.json --out output/dashboard.html ^
       --reference-root . --comparison output/comparison.json --planning-dir output/tx_planning
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

## Stage 3b: where to measure next (`active_measurement.py`)

Transmitter A leaves 22 of 29 materials unconstrained. Before measuring
anything, the twin can say what a candidate second position *would* reveal:
the sensitivity ‖∂PDP/∂s_n‖² of that position's PDPs to each material,
estimated with four random projections and one backward pass each
(Hutchinson: E[(J·w)²] = ‖J‖²). A candidate is scored by
Σ_n log(1 + sens_C(n) / (sens_A(n) + floor)) — large for materials A never
saw, saturating for those it already pins down — and the twin is then fitted
from A plus the best candidate, versus A plus the median-scoring one.
Held-out transmitters: (0, −3), (8, −15), (8, 1), i.e. main room and both
corridor ends; six candidates on a 4 m grid.

| fitted from | mean RMSE at the 3 held-out Tx | materials identifiable |
|---|---|---|
| prior (all 0.5) | 1.24 dB | 0 |
| A only | 0.30 dB | 7 |
| A + median-scoring C = (8, −3) | 0.25 dB | 7 |
| **A + chosen C = (0, 1)** | **0.17 dB** | 8 |

The chosen position halves the held-out error; an arbitrary second position
takes off a sixth. The score is computed from the prior alone, so this is
the planner's loop: fit → ask the twin where it is blind → measure there.
The 4 m candidate grid and 25-step fits are the smoke setting; the ranking
is cheap (about 1 s per candidate) next to the fits.

Optimiser note: `adam-rel` (Adam with ε tied to the largest running gradient)
was tried to let weakly-constrained materials converge faster than
normalised gradient descent allows; it keeps the unconstrained ones still
but its sign-like steps overshoot on noisy gradients (identifiable-material
error 0.051 vs 0.025), so `ngd` stays the default.

## Interactive planner (`interactive.py` + `interactive.html`)

The interaction Aerial's digital-twin demo offers — place a radio unit on the
map, run, look at the coverage — on this scene, without Omniverse: a Python
HTTP server keeps the scene and the 239-point indoor receiver grid in memory,
and one HTML page talks to it.

```text
PYTHONUTF8=1 python tx_planning/interactive.py          # then open http://localhost:8765
```

- click the floor plan → one batched solve → coverage map and fraction (0.2 s)
- **Optimise from here** → optimize_tx.py's loop through Sionna's gradient with the wall constraints (5 steps ≈ 2 s)
- **Retrain RRF here** → a 160-position dataset for that transmitter, remapped, and a 2k-iteration `db` fine-tune in WSL;
  measured 45 s + 60 s = **104 s** from click to held-out PSNR and renders

Aerial's UI components are Omniverse Kit extensions and cannot be reused
outside Omniverse; what is borrowed is the interaction and the data model
(static RU × mobile UE grid, CIR per pair). Dr.Jit's JIT flags are per thread,
so the server is single-threaded on purpose; the retrain job only shells out.

## Depth of the planning objective (2026-09-19)

The coverage objective solves at `max_depth 1`. Measured per receiver with
1M samples (`depth_bias_perrx.py`, 59 grid points, 60 GHz, −85 dB), depth 3
adds a median +0.15 dB and at most +2.5 dB (corridor ends), raising coverage
by 2–3 points at every candidate, and it does not change the ranking: the
depth-1 optimum (8.2, −5.05, 2.0) is also the best at depth 3 (0.627 against
0.610 for the optimum found on a depth-2 objective). The many-receiver
solve had a real defect on the way to that answer: with all 59 receivers in
one scene the coverage of one transmitter fell from 59.3 % (20k samples) to
57.6 % (100k) to 44.1 % (400k), and a depth-3 solve at those budgets lost
first-order power (P10 −28 dB) instead of adding bounces. The cause is the
solver's `max_num_paths_per_src` (default 1e6): the candidate paths of 59
receivers overflow it and are dropped, so the path count saturates near
30k and more samples only truncate more (`sampling_cap_check.py`). With
the cap at 1e7 the coverage is 0.593 at every budget and depth. Every
many-receiver solve here now passes `max_num_paths_per_src=10_000_000`;
the optimiser's own runs (20k samples) never reached the cap and stand.

## Guided placement: coarse solves + a free line-of-sight G-buffer (2026-09-23, `guided_placement.py`)

The DLSS structure, applied to the objective map over transmitter positions: the expensive "shading"
(a full Sionna solve) runs on a coarse grid, and a guide that is exact and nearly free at full resolution
reconstructs the fine map. The guide is the line-of-sight term: both ends of the planning scene are single
isotropic V-polarised elements, so a visible pair's LoS power is (λ/4πd)², and visibility for every
(candidate, receiver) pair is one batched Mitsuba `ray_test`. Real-time denoisers' demodulation does the rest:
P_full = P_LoS + P_mp, the multipath remainder P_mp is interpolated from the coarse solves (inverse distance,
in dB), the exact fine-grid P_LoS is added back, and the top 5 cells of the reconstructed map are verified with
full solves.

- The analytic guide matches Sionna's own LoS term on 100.000 % of the lobby's (Tx, Rx) pairs (dB difference
  < 6e-6) and 99.995 % of the corridor's, and costs 0.011 ms per cell against 56 ms for a full solve. A LoS-only
  Sionna solve is not a cheap guide (10 ms, fixed overhead).
- Lobby, 0.5 m fine grid (480 cells), coverage re-scored at 200k samples:

| method | full solves | K = 1 | K = 2 | K = 3 |
| --- | --- | --- | --- | --- |
| fine exhaustive | 480 | 0.485 | 0.678 | 0.770 |
| 1 m exhaustive (the benchmark's) | 128 | 0.469 | 0.661 | 0.762 |
| **guided, 2 m coarse** | **37** | **0.485** | **0.678** | 0.766 |
| plain interpolation, 2 m, no guide | 37 | 0.485 | 0.611 | 0.695 |
| LoS map alone | 5 | 0.460 | 0.665 | 0.757 |
| benchmark gradient (from the 1 m cell) | 128 + 30 | 0.510 | 0.686 | N/A |
| gradient from the guided cell (`guided_refine.py`) | 37 + 30 | 0.502 | 0.674 | N/A |

- The LoS map alone ranks the cells with a rank correlation of 0.98: at 60 GHz coverage is a line-of-sight
  question, which is why the guide works. Greedy K = 3 is not an optimum (plain 1 m interpolation reached 0.791),
  so the K = 3 column is noise-level.
- Corridor (1020 fine cells): fine exhaustive 0.307 / 0.516 / 0.621 (K = 1 / 2 / 3); the benchmark's 1 m sweep
  (253 solves) 0.307 / 0.490 / 0.608; guided 2 m (69 solves) 0.307 / 0.510 / 0.614; plain interpolation 2 m
  0.307 / 0.487 / 0.592; LoS map alone (5 solves) 0.301 / 0.510 / 0.621. Gradient from the guided cell: 0.301 / 0.513
  in 99 solves, against the benchmark gradient's 0.301 / 0.507 in 283. At 8 of the corridor's 312,120 pairs the
  analytic guide sees a grazing ray Sionna does not (0.005 %), so the remainder is clipped at zero there. The
  fine sweep needs Dr.Jit's pool freed every 5 solves (`--flush-every`), or the card runs out of memory.
