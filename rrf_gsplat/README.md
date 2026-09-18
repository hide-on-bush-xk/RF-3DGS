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

```
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
|---|---|---|---|
| `rgb` | SH coefficients for an RGB triple | alpha-blend RGB (RF-3DGS) | jet PNG |
| `db` | SH coefficients for one value in [0, 1] = normalised dB | alpha-blend the value | float spectrum, normalised with the dataset's global range |
| `power` | SH → value in [0, 1] → linear power 10^(value·span/10) | alpha-blend **powers**, read back as 10·log10 | same as `db` |

`power` is the composition rule that matches what the spectrum is: contributions
add in power, not in dB and not in colour space. Everything else — geometry,
zeroed SH init, INRIA's learning rates (f_dc 0.0025, f_rest /20, opacity 0.05,
Adam ε 1e-15), λ_SSIM 0.2, one random view per step — is the same for all three.

## One evaluation for every model

- **dB RMSE / MAE against the float truth** on the held-out positions. An RGB
  prediction goes back through the jet inverse (4096-entry nearest-neighbour
  LUT; round-trip error ≤ 1.4 % of the span).
- **PSNR / SSIM after jet mapping**, so every mode has the number the paper
  reports. A `db` or `power` prediction is mapped through jet first.
- The split holds out whole receiver positions (160 × 4 yaws = 640 views), as
  the released split does.

## Usage

```
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

## Results

See `docs/stage2_notes.md` while the matrix is running; this section is filled
in from `output/rrf/summary.json` when it is done.
