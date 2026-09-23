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
profile_resolution.py one training step split into SH colour / raster / loss+backward / Adam, at six render sizes
check_face_batching.py controls for the speed flags: render() vs HEAD, batched faces vs single, CUDA vs torch SH
dlss_table.py         round-24 table: time split and decoded quality of every r24_* run
mvdr_peaks.py         main-peak direction / power and top-3 peak detection of beamformed predictions
upsample_renders.py   bilinear upsampling of a run's saved renders (the 2D upsampler of the SR comparison)
neural_shading.py     the DLSS-style pieces: residual CNN, guide buffers (geometry, transmitter physics)
label_sr.py           experiment A1: a learned upsampler for 150x100 labels, scored against the true 300x200
check_shading_head.py controls for --head cnn: identity at the start, guide buffers, gradients
head_share.py         how much of a head model's output is the head, on in-range and peak pixels
dlss_ablation.py      rounds 41-43 tables: every head feature's gain (ladder, leave-one-out, seeds)
```

**Speed (2026-09-23, rounds 24–35; details in `docs/stage2_notes.md`).** A step does not get
cheaper below 600×400 (multi: 39–44 ms from 75×50 to 600×400, `profile_resolution.py`): the cost is
the per-Gaussian SH colour (11 ms) and Adam over 81M parameters (18 ms), so rendering at a lower
resolution and upsampling (DLSS super-resolution) cannot speed training here, and training on
150×100 labels costs the strong paths accuracy (power-weighted azimuth median 1.16° → 1.9–2.1°).
What does pay is DLSS's frame-generation idea: the four faces of one receiver position share their
camera centre and hence their SH colour exactly, so `--faces-per-step 4 --lr-scale 2` shades once,
rasterises four faces in one call and takes one Adam step. With `--sh-backend gsplat --eval-group`
(same numbers, CUDA SH) and the parallel I/O now always on:

| multi, 10k views, 640 held-out views | train | train stage wall | az / zen / delay median |
| --- | --- | --- | --- |
| before (1 view/step, torch SH) | 375 s | 511 s | 0.636° / 1.006° / 0.549 ns |
| + engineering (CUDA SH, grouped eval) | 310 s | 426 s | 0.635° / 1.012° / 0.549 ns |
| + 4 faces/step, lr ×2, 2.5k steps | 101 s | 218 s | 0.618° / 0.942° / 0.539 ns |
| + parallel load, background render writes | 101 s | 129 s | (same config) |

db on MVDR: 167 s → 52 s at 18.75 → 18.77 dB PSNR(jet); over three seeds each the main peak's
direction is within seed noise but its power is ~0.5 dB lower (`mvdr_peaks.py`), a real cost.
With the geometry unfrozen (the best MVDR rows): 302 → 97 s at 21.53 → 21.45 dB, same 2.85 dB RMSE
(one seed). Per view the four-face step is 2.5–2.9× faster at 150×100, 300×200 and 600×400 alike. The planner's "Retrain RRF here" training stage
went from 56 s to 21 s at the recorded live transmitter (data + training 104 → 69 s, data generation
now 70 % of it); clicked through the page at (8.2, −5.05, 2) the whole job takes 77 s.

**DLSS-5-style deferred shading (2026-09-23, rounds 39–43; `docs/stage2_notes.md`).** Labels at
150×100 save only 11 % (MVDR) / 18 % (MULTI) of the data generation, and a learned label upsampler
is no better than bilinear at the peaks, so label super-resolution is dropped. `--head cnn` puts a
small residual CNN, shared by every view, after the rasteriser, with the residual bounded at
±6 dB and fed by guide buffers rasterised alongside the colour (`--head-guides geo,phys`,
`--head-latent 4`, `--head-strip`); `--peak-loss 1` adds L1 on each view's top 10 dB. Train it with
`--head-bound-units --head-warmup 250`: without them 3 of 21 heads saturated at the bound in the
first 250 steps and never recovered. On MVDR (3 seeds): SH3 + head 21.52 dB PSNR(jet) vs 18.72 without,
main peak 1.2° closer and 1.0 dB stronger at the true peak, but no more views within 1°, and
the step is 2.4× (300×200) to 5.5× (600×400) slower, so it is a quality option, not a speed-up. Per feature:
the head itself +1.1 dB, the geometry guides +1.9 when added first but redundant with the rest,
the learned latent the only piece that costs when removed (−0.4 dB), transmitter physics and the
four-face ring nothing measurable; width 8 matches width 32. The Gaussians under a head are a worse
field on their own (SH3: 3.5 dB mean error without the head vs 2.7 dB for SH3 trained alone; the head
carries 17–28 % of the output variance), so nothing that renders the Gaussians without the head
should use a head model's parameters.

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

**Where things stand (2026-09-20, measurements frozen at `d359dc9`).** The
paper text is `docs/paper1_draft.md` (v0.6); the round-by-round record is
`docs/stage2_notes.md`; a one-table index of every result after 09-18 is
`docs/ours_vs_rf3dgs.md` section 8. Since the paragraphs below were written:
the delay range term is the Euclidean range (`--delay-range euclid`, the
method's default: median 0.88 to 0.55 ns, P90 5.38 to 4.48, RMSE 5.26 to
4.97); the reconstructed geometry was measured against the scene mesh
(`depth_gs.py` + `depth_gt.py`: lobby 0.117 m median, corridor 0.365 m, both
in front of the surface, which is the delay channel's negative bias); a
second scene (`scene2/`) separated transmitter distance from lit-surface
overlap (the room next door costs 11 dB, the facing room gains 2.5), split
the density crossover by propagation regime, and collapsed the lookup
baseline in a held-out room; seed noise floors were measured for the
transfer benefit (0.03 dB lobby, 0.31 dB corridor) and for the decoded
medians (0.002 to 0.026 deg, 0.006 to 0.011 ns). Two guards were added after
a protocol error: `--save-renders` defaults to every held-out view and
`eval_baselines.py` refuses a partial render set.

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
incident-direction term would recover it. Which Gaussians carry the gain
(`--geometry-subset`, gradients masked to a subset, nothing outside it
moves): the discs alone (11 % of the Gaussians) reach 21.49 / 2.86, the
needles alone (40 %) 21.32 / 2.92, and a *random* 11 % 21.27 / 2.91, a
random 3 % 21.00 / 3.02, a random 1 % 20.72 / 3.14. Shape class is not
the variable: one Gaussian in a hundred, chosen at random, recovers 73 %
of the RMSE gain, so the adaptation is a low-dimensional, distributed
correction rather than a property of surfels or needles, and neither
2DGS nor a needle-specific incident term is singled out by it. Rays are
not single-Gaussian either: the effective number of Gaussians per ray,
N_eff = (Σw)²/Σw² (`diag_neff.py`, Rademacher estimate), has median 5.9
on the frozen model and 7.5 after unfreezing, with under 4 % of pixels
below 2, so the equal gain of dB and linear-power compositing is a
measured fact, not a consequence of one Gaussian owning each pixel.

**How far an adaptation carries** (`transfer_curve.py`, `diag_overlap.py`):
ten source transmitters across the lobby, each adapted for 10k steps, its
geometry frozen on Tx-B with colours reset and refit for 2k steps (a budget
calibrated against the 10k points on A/C/D: same signs, same order; seed
noise 0.03 dB over four seeds). The benefit over the visual geometry is
+0.57 dB at 1.8 m, +0.34 at 4.2 m, +0.29 at 5.8 m along the east side
where Tx-B sits, and between −0.06 and −0.35 dB for every source in the
main hall, the south corridor or the west, whatever its distance (5.3 m
is as useless as 12.8 m). The main result is binary: the four sources on
Tx-B's side of the lobby all help (mean +0.33 dB, worst +0.12), the six
elsewhere all hurt (mean −0.20 dB, best −0.06), and distance does not
separate them. An exponential-plus-plateau fit gives d_c = 5.3 m, but a
1000-resample bootstrap puts its 90 % interval at 2–40 m with 17 % of
resamples resolving no decay at all, so d_c is reported, not concluded;
distance and the illumination overlap of the two transmitters (IoU of
the top-30-dB pixels on the held-out views; benefit correlates −0.81
with distance and +0.85 with the overlap) are collinear and their
coefficients cannot be read separately. The plateau is the firm number:
c = −0.46 dB (bootstrap median −0.50), i.e. geometry adapted elsewhere is
*worse* than the visual geometry. The adaptation is directional, and
stacking ten of them (`diag_delta_pca.py`) shows why: the ten deformations
are nearly orthogonal (mean pairwise cosine 0.18, five principal components
carry 66 % of the energy), only their mean, 26 % of the energy, is shared
(the 24 % transferable part measured earlier), and interpolating a new
transmitter's deformation from its neighbours' coefficients is no better
than using that mean. One adaptation serves a moved transmitter only
where the two light the same surfaces; a multi-transmitter deployment
should keep the visual geometry plus a per-transmitter delta on the 1–3 %
of Gaussians that carry the gain, not one adapted geometry for the floor,
and not a shared low-rank basis taken from independent adaptations.
gsplat's MCMC densification with its default
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

The floor under those numbers (`eval_baselines.py`, same views and
pixels): copying the spectrum of the nearest *training* position (median
0.23 m away, no model) decodes to 1.07° / 131° / 55° azimuth median / P90 /
RMSE, 0.79° zenith median and 0.86 ns delay median; an inverse-distance
blend of the two nearest gives 0.81° / 10.8° / 37°. So the field's median
azimuth gain over a copy is 1.8×, its gain is in the tail (P90 5.9° against
131°), and its delay channel is 3× *worse* than the copy in the median
(2.4 against 0.9 ns, the geometric delay of a 23 cm move). Translated into
beam selection (square θ₃dB codebook, true beam among the k cells nearest
the decoded direction; unweighted | weighted by pixel power): with the
M = 10 array of these datasets (θ₃dB 10.2°) the field reaches top-1 0.68 |
0.70 and top-3 0.84 | 0.91 against the copy's 0.60 | 0.65 and 0.74 | 0.82;
with the 64 × 64 array the paper headlines (1.59°) the σ = 1° field's
strong-path top-1 (0.10) falls *below* the copy (0.23) and only the 0.33°
kernel recovers a lead (top-1 0.19 | 0.19, top-5 0.44 | 0.61 against 0.26 |
0.20). The field is a usable beam selector for the M = 10 array and not a
top-1 selector for a 64 × 64 one; and the downstream task picks the kernel
width (0.33°) that the power-weighted metric picked.

Those floors expose the evaluation protocol rather than the model: a
random 20 % hold-out leaves every test position 0.23 m from a training
one, a density no channel sounder delivers. Two protocol changes fix that.
**Density sweep** (training positions 800 / 160 / 80 / 40 / 20, same
held-out views, copy baseline drawn from the same subset, the field with
the delay decomposition below): the copy's azimuth median grows linearly
with the spacing, 1.07° → 1.68° → 2.55° → 3.72° → 4.68° over 0.23 → 0.84 m,
while the field stays at 0.62° → 0.63° → 0.66° → 0.75° → 0.84° with its P90
near 6.5° throughout (the copy's 131–152°); zenith goes 0.79° → 2.91° for
the copy against 0.96° → 1.24° for the field, delay 0.86 → 3.88 ns against
0.95 → 1.43 ns. All three channels cross between 0.23 and 0.35 m spacing
(azimuth already below 0.23 m, zenith and delay at about 0.25–0.27 m by
linear interpolation): **beyond about 0.3 m of measurement spacing the
field beats nearest-neighbour lookup on every physical channel**, and the
lookup's error grows 4.4× over the sweep where the field's grows 1.3–1.5×.
Twenty positions, 80 images, still give 0.84° median azimuth. The flatness
is the frozen visual geometry carrying the spatial structure, so the
crossover depends on that geometry's quality; a second scene tests it. **Leave-one-region** (`make_region_split.py`, the
south-east corridor end held out, 123 positions, nearest training position
1.43 m median and 2.75 m at most): the field extrapolates to 1.47° / 17.6°
azimuth median / P90 against the copy's 7.38° / 93°, zenith 1.99° against
4.29°, PSNR 14.1 against 16.1 on the random split; delay is the weak
channel there (6.78 against the copy's 5.95 ns median, 23 against 52 ns at
P90). A constant predictor (the training pixels' mean) scores 27.6° / 7.2°
/ 9.5 ns, so none of the three channels is degenerate.

**Delay as analytic range plus learned residual** (`--delay-depth`). A
path's delay is the scatterer's own delay plus the range from the Gaussian
to the receiver over c; the second term is a distance, which a directional
SH colour cannot represent, and it dominates the channel (3 m of depth is
10 ns against a 2.4 ns error). gsplat renders the alpha-composited depth
natively (render mode RGB+D), so the delay channel gets Σ wᵢ dᵢ / c added
in its normalised units and the learned part keeps only the
view-independent scatter delay. Same data, same everything else: delay
median 2.42 → **0.95 ns**, P90 12.8 → 6.6 ns, RMSE 8.9 → 6.9 ns,
power-weighted median 3.54 → 0.40 ns (the copy's 0.44); azimuth, zenith
and power unchanged (0.62° / 0.96° / PSNR 16.14). It is the one change
here forced by a measurement rather than borrowed from a paper.

Carrying the azimuth as cos and sin removes the seam at ±180° that the
single channel had (the pinhole resampling interpolated across it, and the
field had to fit a jump that is not physical). The controlled A/B — the
same dataset, five channels in both, the seamed azimuth carried twice in the
control — gives azimuth median 1.86° → 0.59°, P90 11.7° → 5.9°, RMSE 19.1°
→ 9.8°, while zenith, delay and power are identical to two decimals. So the
seam costs its own channel a factor of three and nothing else: an unfittable
feature in one channel does not leak into the others through the shared
opacity. The typical pixel decodes to within 1° of both angles; the RMSE
is carried by the 3 % of pixels where two comparable paths with opposite
departure angles share a pixel and the encoded mean lands between them (the
same tail exists in the target itself: `diag_encoding_truth.py` puts the
target's median at 0.2° and its RMSE at 8.9° with no field involved). The
tutorial's encoding is bad in the median too: decoding it divides two
learned channels, and δ(a/b)/(a/b) = √((δa/a)² + (δb/b)²) is unbounded as
the amplitude channel → 0. The gain of one channel per quantity is
conditioning, not capacity.

Splat width, swept over 0.33° / 1° / 2° / 4.3° (1 / 3 / 6 / 13 equirect
pixels, (cos, sin) target). The set of pixels a path reaches grows with σ
(8.8M → 32M), so every row is scored twice: on its own support, and on the
fixed support of the σ = 0.33° dataset (8.8M pixels, the kernel cores):

| σ | azimuth, own support | azimuth, fixed support | zenith, fixed | delay, fixed |
| --- | --- | --- | --- | --- |
| 0.33° | 0.93° / 9.7° | 0.93° / 9.7° | 1.63° / 29.4° | 3.3 / 19.1 ns |
| 1° | 0.59° / 5.9° | **0.56° / 3.9°** | 0.68° / 4.2° | 2.3 / 13.0 ns |
| 2° | **0.57° / 5.3°** | 0.64° / 5.3° | **0.64° / 3.2°** | 2.3 / 13.0 ns |
| 4.3° | 0.70° / 6.2° | 0.95° / 9.7° | 0.72° / 3.6° | 2.3 / 12.8 ns |

(median / P90.) On a fixed support the azimuth optimum is σ\* = 1°, not
the 2° the growing support suggested: the pixels a wider kernel adds are
kernel tails one path reaches alone, easy in the median, and they moved
the minimum. Scored support-free instead, every touched pixel weighted by
its own linear power (`--power-weighted`), the azimuth error falls
monotonically down to the sweep's finest kernel, σ = 0.33° (the equirect
grid step): median 0.84°, P90 2.15°, against 1.5° / 5.7° at 1° and
1.1° / 7.8° at 4.3°; zenith prefers wide kernels (RMSE 2.6° at 2°) and
delay is flat. So the optimum is a property of the metric's weighting,
not of the field: by pixel count it sits at 1–2° because smoothing helps
the many weak pixels, by power it sits at the grid limit because a narrow
kernel keeps strong paths separate. The "σ\* measures the representation
bandwidth" reading is dropped. What can be said with its metric attached:
by power, the field reproduces strong-path azimuth under a 0.33° kernel
(finer than a 64 × 64 UPA's 1.59° beamwidth); by pixel count, the interior
optimum is 1–2°. The paper's 5.94° median beam-pointing error is a
different quantity (an M = 10 beam pointed by the field), not a comparison
point.

**Transmitter moved** (Tx-B at (8.2, −5.4, 2.0), Tx-A's dB range): cold start
reaches PSNR 17 in 1500 iterations, warm start from the Tx-A `db` model in
1250; in `rgb` a warm start *hurts* (2000 → 3250). The cost of one move, as
the full chain (dataset + training + evaluation) on an RTX 3060 12 GB held
exclusively, per the reporting contract in `CLAUDE.md`:

| pipeline | dataset (3200 views unless stated) | fine-tune | evaluation | total |
| --- | --- | --- | --- | --- |
| as published: Sionna 0.19 + TF (CPU: no OptiX in WSL, no GPU for TF 2.15 on Windows), INRIA `train.py` 30k→40k | 0.9 h (1.0 s/view: 0.4 s paths + 0.6 s MVDR, exclusive; the authors' GPU box would be faster — MISSING) | 272 s incl. its own test pass | (inside) | ≈ 1 h on this machine |
| as published + receiver batching only (Sionna 0.19 + TF on the CPU, the tutorial's per-view loop replaced by batched receivers, nothing else) | MISSING (never measured in isolation) | 272 s | (inside) | MISSING |
| as published + engineering fixes: Sionna 2.1 on the GPU (OptiX), the port's cached angle grids and torch MVDR, same spectra (2.4 GHz, tutorial materials, M = 10), INRIA `train.py` | 222 s (14.4 views/s) | 272 s | (inside) | ≈ 8.2 min |
| this work, full data: 800 positions, `db`, gsplat, 10k it | 311 s (10.3 views/s, 60 GHz) | 168 s | 35 s | 8.6 min |
| this work, method: 160 positions, `db`, 2k it | 33 s (640 views) | 36 s | 35 s | **≈ 1.7 min** (live page: 104 s measured click-to-render, 2k it, 32-position test) |

Decomposition of the speed-up against the published pipeline: the
**engineering** part (ray tracing on the GPU through Sionna 2.1) takes the
dataset from 0.9 h on this machine's CPU to under 4 min; the **method**
part (the dB target converging 3.6× sooner, 160 positions losing 0.06 dB
on Tx-A and 0.4 dB on Tx-B, 2k iterations) takes the rest from 8 min to
under 2. Training alone is not the number: at 2k iterations the evaluation
pass is as long as the training.
