# rrf_gsplat — the radio radiance field on gsplat

Stage 2 of the project: test the generation-side improvements on the model
itself. RF-3DGS fine-tunes colour and opacity of a frozen visual 3DGS on
jet-colourmapped spectrum PNGs through the INRIA rasteriser (3 channels,
L1 + SSIM). This directory does the same fine-tune on **gsplat 1.6**, which
rasterises any number of channels, so the target no longer has to be an RGB
picture — and then runs the experiments that this makes possible.

Runs in WSL (`rf-gsplat`, Python 3.10, torch 2.9.1+cu130, gsplat 1.6.0), reading
the repo from `/mnt/c`. Datasets come from `sionna_port/generate_dataset.py` on
Windows, which also writes the float spectra the evaluation needs.

```text
train_rrf.py          the fine-tune; --mode rgb | db | power; evaluation in dB and in PSNR
jet.py                matplotlib's jet and its inverse in torch (to score RGB models in dB)
renormalize.py        remap a dataset's float spectra to PNGs: global percentile / global min-max / per view
summarize.py          output/rrf/*/results.json -> one table (summary.json + markdown)
rrf_panels.py         the dashboard section (sionna_port/dashboard.py --rrf-dir output/rrf)
run_matrix.sh         the experiment matrix, skips finished runs
wsl_run.sh            run one training in WSL with a log
win_gpu_jobs.sh       Windows-side GPU jobs: Tx-B dataset, 2.4 GHz timing, INRIA rasteriser timing
time_tutorial_019.py  the original tutorial pipeline (Sionna 0.19 + TF) timed on this machine
```

## The three colour functions

| `--mode` | what a Gaussian carries | how it composites | loss target |
| --- | --- | --- | --- |
| `rgb` | SH coefficients for an RGB triple | alpha-blend RGB (RF-3DGS) | jet PNG |
| `db` | SH coefficients for one value in [0, 1] = normalised dB | alpha-blend the value | float spectrum, normalised with the dataset's global range |
| `power` | SH → value in [0, 1] → linear power 10^(value·span/10) | alpha-blend **powers**, read back as 10·log10 | same as `db` |

`power` is the composition rule that matches what the spectrum is: contributions
add in power, not in dB and not in colour space. Everything else — geometry,
zeroed SH init, INRIA's learning rates (f_dc 0.0025, f_rest /20, opacity 0.05,
Adam ε 1e-15), λ_SSIM 0.2, one random view per step — is the same for all three.

## One evaluation for every model — physical metrics first

- **What the field decodes to** (`--mode multi`, `eval_encoding.py`): RMSE of
  the departure azimuth and zenith in degrees and of the delay in ns, on the
  held-out pixels a path reaches. This is the metric a beam-management or
  localisation use would feel, and the primary one here.
- **dB RMSE / MAE against the float truth** on the held-out positions. An RGB
  prediction goes back through the jet inverse (4096-entry nearest-neighbour
  LUT; round-trip error ≤ 1.4 % of the span).
- **PSNR / SSIM after jet mapping** — a *compatibility* number, not a training
  target: every model's output (dB, power or multi-channel) is mapped through
  jet and scored against the PNG the way the paper does. No run here optimises
  it; the `db`/`power`/`multi` losses live in the physical domain.
- The split holds out whole receiver positions (160 × 4 yaws = 640 views), as
  the released split does.

## Usage

```bash
# Windows: datasets (float + PNG), then the global-percentile remap
python sionna_port/generate_dataset.py --scene-xml ... --rx-loc-file ... --out-dir RF-3DGS_dataset/regenerated/3dgs_MVDR_100 --spectrum MVDR
python rrf_gsplat/renormalize.py RF-3DGS_dataset/regenerated/3dgs_MVDR_100 RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct --norm global-pct --pct 1 99.99

# WSL: one run, or the matrix
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/.../rrf_gsplat/wsl_run.sh e2_mvdr_db --source RF-3DGS_dataset/regenerated/3dgs_MVDR_100_gpct --mode db
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-22.04 -u ke -- bash /mnt/c/.../rrf_gsplat/run_matrix.sh colour norm ablate txmove

# table + dashboard
python rrf_gsplat/summarize.py
python sionna_port/dashboard.py output/ablation.json --out output/dashboard.html --reference-root . \
       --comparison output/comparison.json --planning-dir output/tx_planning --rrf-dir output/rrf
```

## Results (2026-09-18, RTX 3060; full table in `output/rrf/summary.json`, narrative in `docs/stage2_notes.md`)

**The port reproduces RF-3DGS.** On the released MVDR data, `rgb` reaches
15.97 dB / 0.727 SSIM on the 640 held-out views (published checkpoint: 16.02 /
0.731), identically through the torch SH path and gsplat's CUDA SH.

**Colour function** — regenerated MVDR, global percentile range, 10k iterations:

| mode | PSNR (jet) | SSIM | RMSE dB | it/s | s to PSNR 17 |
| --- | --- | --- | --- | --- | --- |
| rgb (RF-3DGS) | 18.15 | 0.802 | 4.42 | 47 | 69 |
| **db** | **18.75** | **0.818** | **3.91** | 60 | **19** |
| power | 18.63 | 0.812 | 3.97 | 58 | 20 |

Targeting the spectrum instead of its picture takes 11 % off the dB error and
reaches the same quality 3.6× sooner; compositing linear power buys nothing
over compositing dB here. The RGB model also composites colours off the jet
curve (see the strip on the dashboard), which the value modes cannot do.

**Normalisation** — same float spectra, `rgb`:

| data | range | PSNR (jet) | RMSE dB (oracle range) |
| --- | --- | --- | --- |
| MVDR | global | 18.15 | 4.42 |
| MVDR | per image | 11.97 | 7.61 |
| CBF | global | 13.68 | 7.54 |
| CBF | per image (the tutorial's CBF/TCBF) | 13.09 | 11.03 |

Per-image normalisation costs 6 dB PSNR on MVDR even when the evaluation is
handed each test image's true range. The CBF weakness the paper reports has
this in it. Rerun at the released data's own setting (CBF, 2.4 GHz, the
tutorial's materials — verified below): global 13.00 / 9.32 dB, per image
12.49 / 13.35 dB, global with the `db` target 13.39 / 8.54 dB.

**The released data are 2.4 GHz.** Generated at the released datasets' exact
poses (`--poses-from`), the CBF power the tutorial writes to `cbf_power.csv`
matches our 2.4 GHz spectra within 1.6 dB and the 60 GHz ones by 34 dB
(`output/freqcheck/`); the notebook's dataset cells set 2.4e9, the paper says
60 GHz.

**Does it hold under other data?** With the tutorial's own per-material
definitions (`--materials tutorial`, span 83 dB): rgb 17.29 / 5.33 dB, db
18.04 / 4.68 dB. At the tutorial's 2.4 GHz with those materials — the
released data's setting — rgb reaches 16.87 dB (released checkpoint on the
released data: 16.02), db 17.66. On CBF: rgb 13.68 / 7.54, db 13.95 / 7.04.
The dB target wins in every setting tried.

**Ablations** (`db`): SH0 16.75 → SH1 17.43 → SH3 18.75 dB, so view dependence
is where the capacity goes; frozen opacity −0.22 dB; 2k iterations (36 s) 17.45;
160 of 800 positions −0.06 dB, 40 positions −1.3 dB.

**Geometry.** Unfreezing means/scales/quats (`--train-geometry`, INRIA's
lrs with the means at 10× its final lr) takes the same model from 18.75 /
3.91 dB to **21.53 / 2.85 dB** — the largest single gain here. RF-3DGS's
frozen geometry assumes the visual geometry is the radio geometry; that
assumption costs a dB of RMSE. What moved (`diag_displacement.py`): the
means by 6 mm median and 21 cm at most, neither onto new surfaces nor into
free space — the gain is in scales and rotations. And it transfers: the
Tx-A-adapted geometry, frozen and with colours reset, fine-tuned on Tx-B
gives 17.99 dB / 4.58 dB in-range RMSE against 17.86 / 4.82 from the visual
geometry (`--init-from ... --init-geometry-only`); unfreezing on Tx-B itself
gives 19.43 / 3.83. Read as a decomposition: **24 % of the unfreezing gain
is a transferable geometric correction, 76 % is specific to the transmitter
it was fitted on.** That is not a frozen-versus-joint artefact — the adapted
geometry frozen on Tx-A itself, colours refit from zero, reaches 21.55 /
2.81, the same as the joint fit. Two alternatives were tested and fail:
the gain is not the geometry absorbing the dB-domain mixing bias (the
`power` mode, which composites in the linear domain, gains the same
−27.5 %: 3.97 → 2.88), and it is not a generic re-allocation of capacity
that any transmitter would supply (the geometry adapted on a third
transmitter, Tx-C in the main room, carried to Tx-B is *worse* than the
visual geometry: 4.95 against 4.82 in-range RMSE). `diag_shape.py` finds no
directional signature of the transmitter in the change (rotation axes
isotropic to the Tx direction; extent along it +7 % against +3 %
elsewhere), and the visual checkpoint is needle-like rather than disc-like
(s_mid/s_min median 3.5; discs 15 %, needles 36 %), so the surface-normal
test only applies to the disc subset, where alignment gets 5° worse. The
position basis stays Tx-invariant; the shape adaptation carries
transmitter-specific information without being a stretch towards the
transmitter — the measured cost of a colour function with no
incident-direction argument, and an open question whether a degree-1
incident-direction term would recover it. gsplat's MCMC densification with its default
hyper-parameters diverges from this converged start (NaN loss by 4k
iterations at either cap); a low-noise, late-start, relocate-only
configuration is the next thing to try.

**Multi-channel targets** (`--mode multi`, 2.4 GHz, tutorial materials):
path power, AoD azimuth (as cos and sin, so the channel has no seam), AoD
zenith and delay each get a channel. Decoded on the same 20.4M held-out
pixels a path reaches, reported as **median / P90 / RMSE** because the error
has a tail:

| decoded from | AoD azimuth | AoD zenith | delay |
| --- | --- | --- | --- |
| one channel per quantity, azimuth as (cos, sin) | **0.6° / 5.9° / 9.8°** | 0.9° / 12.1° / 14.7° | 2.4 / 12.8 / 8.9 ns |
| one channel per quantity, azimuth as (φ+180)/360 | 1.8° / 11.0° / 19.0° | **0.7° / 7.7° / 11.6°** | **2.3 / 11.5 / 8.2 ns** |
| the tutorial's angle × amplitude RGB | 26.9° / 85.6° / 48.6° | 29.6° / 90.5° / 56.6° | — |

Carrying the azimuth as cos and sin removes the seam at ±180° that the
single channel had (the pinhole resampling interpolated across it, and the
field had to fit a jump that is not physical): azimuth median 1.8° → 0.6°,
P90 and RMSE halved. The typical pixel decodes to within 1° of both angles; the RMSE
is carried by the 3 % of pixels where two comparable paths with opposite
departure angles share a pixel and the encoded mean lands between them (the
same tail exists in the target itself: `diag_encoding_truth.py` puts the
target's median at 0.2° and its RMSE at 8.9° with no field involved). The
tutorial's encoding is bad in the median too: decoding it divides two
learned channels, and δ(a/b)/(a/b) = √((δa/a)² + (δb/b)²) is unbounded as
the amplitude channel → 0. The gain of one channel per quantity is
conditioning, not capacity.

Splat width as a measurement, not a hyper-parameter: sweeping σ over
0.33° / 1° / 2° / 4.3° (1 / 3 / 6 / 13 equirect pixels, (cos, sin) target),
the decoded azimuth error is smallest at **σ\* ≈ 2°** (median 0.57°, P90
5.3°; 0.33° gives 0.93° / 9.7°, 4.3° gives 0.70° / 6.2°). That minimum is
the field's representation bandwidth: θ₃dB = 2.355 σ\* ≈ 4.7°, i.e. the
radiance field resolves angles like an M ≈ 22 array (the paper's M = 10
array has θ₃dB ≈ 10°). Zenith and delay keep improving with smoother
targets. Caveat: the set of pixels a path reaches grows with σ (8.8M →
32M), so the rows are not scored on one support; a fixed-support version
is pending.

**Transmitter moved** (Tx-B at (8.2, −5.4, 2.0), Tx-A's dB range): cold start
reaches PSNR 17 in 1500 iterations, warm start from the Tx-A `db` model in
1250; in `rgb` a warm start *hurts* (2000 → 3250). The cost of one move, as
the full chain (dataset + training + evaluation) on an RTX 3060 12 GB held
exclusively, per the reporting contract in `CLAUDE.md`:

| pipeline | dataset (3200 views unless stated) | fine-tune | evaluation | total |
| --- | --- | --- | --- | --- |
| as published: Sionna 0.19 + TF, INRIA `train.py` 30k→40k | MISSING (the 1.1 s/view CPU figure was taken beside a running GPU job; re-measurement pending) | 272 s incl. its own test pass | (inside) | MISSING |
| as published + engineering fixes: Sionna 2.1 on the GPU, same spectra (2.4 GHz, tutorial materials, M = 10), INRIA `train.py` | 222 s (14.4 views/s) | 272 s | (inside) | ≈ 8.2 min |
| this work, full data: 800 positions, `db`, gsplat, 10k it | 311 s (10.3 views/s, 60 GHz) | 168 s | 35 s | 8.6 min |
| this work, method: 160 positions, `db`, 2k it | 33 s (640 views) | 36 s | 35 s | **≈ 1.7 min** (live page: 104 s measured click-to-render, 2k it, 32-position test) |

Decomposition of the speed-up against the published pipeline: the
**engineering** part (ray tracing on the GPU through Sionna 2.1) takes the
dataset from ~1 h (MISSING until re-measured) to under 4 min; the **method**
part (the dB target converging 3.6× sooner, 160 positions losing 0.06 dB
on Tx-A and 0.4 dB on Tx-B, 2k iterations) takes the rest from 8 min to
under 2. Training alone is not the number: at 2k iterations the evaluation
pass is as long as the training.
