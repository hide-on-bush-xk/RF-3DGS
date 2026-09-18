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
assumption costs a dB of RMSE. gsplat's MCMC densification with its default
hyper-parameters diverges from this converged start (`--densify mcmc
--cap-max 1300000`: 14.7 dB at 1k iterations, 5.3 by 8k), and the cap-at-N
variant hits a CUDA error in the relocation kernel; a low-noise, late-start,
relocate-only configuration is the next thing to try.

**Multi-channel targets** (`--mode multi`, 2.4 GHz, tutorial materials):
path power, AoD azimuth, AoD zenith and delay each get a channel. Decoded on
the same held-out pixels, the per-quantity channels give 19.0° / 11.6° / 8.2 ns
(azimuth / zenith / delay RMSE) where the tutorial's angle × amplitude RGB
encoding decodes to 48.6° / 56.6°. With one channel per quantity the
amplitude no longer needs to be multiplied into the angle at all.

**Transmitter moved** (Tx-B at (8.2, −5.4, 2.0), Tx-A's dB range): cold start
reaches PSNR 17 in 1500 iterations (≈ 27 s), warm start from the Tx-A `db`
model in 1250; in `rgb` a warm start *hurts* (2000 → 3250). Measured end to
end with a 160-position dataset (33 s to generate) the RRF reaches PSNR 17
after 32 s of training cold, 23 s warm — **about one minute per transmitter
move**, at a cost of 0.4 dB in the final PSNR against the 800-position
dataset (17.45 vs 17.86). The tutorial pipeline on this machine takes about an
hour for the dataset alone (CPU, 1.1 s/view).
