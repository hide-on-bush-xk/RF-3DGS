# What Determines the Quality of a Radio Radiance Field? A Controlled Measurement Anchored on RF-3DGS

*Draft v0.2, 2026-09-19. Every number is taken from `docs/stage2_notes.md` and `output/rrf/*`; `[MISSING: …]` marks a number the experiments have not produced yet. All numbers are on one scene (the NIST lobby of RF-3DGS) unless a scene-2 row is given. Timings follow the reporting contract in `CLAUDE.md` (full chain, GPU model and exclusivity, medians after warm-up, two resolutions).*

## Abstract

Radio radiance fields (RRFs) fit a 3D Gaussian scene to spatial spectra so that the spectrum at an unvisited receiver can be rendered. Recent work changes the representation (planar Gaussians, bidirectional spherical harmonics, receiver-conditioned radiance) and reports the PSNR of the rendered spectrum. We ask a prior question: with the representation fixed, what determines reconstruction quality, and does the number reported measure it? Anchored on a reproduction of the published RF-3DGS checkpoint (15.97 dB against the published 16.02 dB), we vary one factor at a time on the same scene, simulator and evaluation. Three findings. (i) Cross-view consistency of the training data dominates. Normalising each spectrum image separately costs 6.2 dB PSNR; drawing the ray tracer's diffuse paths per view instead of per position costs 32 % of the decoded power error on a path-level target, and nothing at all on a beamformed MVDR target (18.69 / 3.94 against 18.75 / 3.91 dB), which bounds the claim to the target type. Both costs exceed every representation change we measured, including unfreezing the geometry (−27 % RMSE). (ii) The geometry gain is distributed, low-dimensional and transmitter-specific: a random 1 % of the Gaussians recovers 73 % of it; ten geometries adapted to ten transmitters are nearly orthogonal (five principal components carry 66 % of their energy); and a geometry adapted on one transmitter helps a second only when both light the same surfaces (+0.33 dB mean on the same side of the lobby) and hurts otherwise (−0.20 dB mean, plateau −0.46 dB), whatever the distance. (iii) PSNR is blind to the physics: a conductivity unit error that turns every wall into a perfect conductor, a 2.3× error in the multi-bounce power share, a 34 dB carrier mismatch between the released data and the paper, and the kernel width that decides beam-selection accuracy all leave PSNR unchanged. We therefore score decoded angles and delays against the floor those numbers lacked (nearest-measurement lookup) and the ceiling (the target's own encoding). The field beats lookup on azimuth, zenith and delay once measurements are more than about 0.3 m apart in the lobby and 0.5 m in a second, corridor scene (the spacing is set by the scene's spatial correlation length, not by the wavelength), under two subsampling rules, and extrapolates into a held-out corridor end at 1.6° azimuth against 7.4° for lookup. One change forced by these measurements, rendering the delay's range term from the Gaussian depth and learning only the residual, takes the delay error from 2.42 to 0.88 ns median and from 8.9 to 5.3 ns RMSE with the other channels unchanged.

## 1. Introduction

A radio radiance field represents the spatial spectrum a receiver would measure at any position as the alpha-composited emission of a set of 3D Gaussians whose positions come from a visual reconstruction of the room. RF-3DGS introduced the idea on a simulated NIST lobby; RF-PGS, BiWGS, GSRF and RxGS have since changed what a Gaussian stores (planar primitives, bidirectional spherical harmonics, receiver-conditioned radiance) and how it is composited, and each reports the peak signal-to-noise ratio, sometimes the mean absolute error in dB, of the rendered spectrum on held-out receivers.

Three things are missing from that literature, and this paper supplies them.

- **An anchor.** No published RRF result reproduces another. RF-PGS reports an RF-3DGS baseline of 14.22 dB on the same lobby; the RF-3DGS release reaches 16.02 dB on its own data; the two used different data sizes and configurations, so neither the 20.61 dB of RF-PGS nor any number of ours is comparable to them. We reproduce the released checkpoint on the released data with our trainer to 15.97 dB, bit-identical between two spherical-harmonic implementations, and compare only against that.
- **Control of the data pipeline.** Alpha compositing assumes one physical field observed consistently from every view. We show that the published pipeline violates that assumption in two independent places, per-image normalisation and per-view Monte-Carlo sampling of diffuse paths, and that each costs more than any representation change we measured (§4).
- **A floor and a ceiling for the physical quantities.** PSNR of a colour-mapped spectrum does not see whether the decoded angle of departure or the delay is right, and we exhibit four independent errors it does not see (§6). We evaluate the decoded azimuth, zenith and delay against nearest-measurement lookup, two-neighbour interpolation and a constant predictor, and against the target's own encoding error (§7). The floor turns out to be strong: at the published sampling density, copying the nearest measurement decodes zenith and delay better than the learned field, and only the tail and the sparse-sampling regime separate them.

One algorithmic change follows from the floor, not from a prior paper. A path's delay is the scatterer's delay plus the Gaussian-to-receiver range over the speed of light; the range is a distance that a directional spherical-harmonic colour cannot represent. Rendering it from the composited depth and learning only the residual takes the delay error from three times worse than lookup to level with it (§7.5).

**Scope.** One lobby, two carrier frequencies (2.4 and 60 GHz), two material sets, one array (a 10-element uniform planar array, as in the released data), one visual checkpoint. A second scene, a corridor with six closed rooms and a hall, was built to replicate three claims with predictions written before the runs; its results are `[MISSING: scene 2 rows, §9]`.

## 2. Related work

Only papers whose arXiv page and numbers were opened and checked are cited with numbers (`docs/literature_check.md`, verification table); a number we could not find on the page is marked UNVERIFIED and not used.

**RF-3DGS** (the anchor) trains INRIA 3DGS on jet-coloured CBF, TCBF and MVDR spectra of a Sionna-simulated NIST lobby, four 90° pinhole faces per receiver position, with the Gaussians' positions frozen from a visual reconstruction. The released datasets are at 2.4 GHz (§6, instance 2) although the paper's narrative is 60 GHz. **RF-PGS** (arXiv 2508.16849) replaces the Gaussians by planar primitives and a "fully structured" radiance, reports 20.61 dB PSNR against 14.22 dB for RF-3DGS as an average over several configurations of the same lobby, trains in 3 min 53 s, computes time of flight purely from geometry and reports no delay error; its code is announced "upon acceptance". **BiWGS** (arXiv 2510.26166) stores bidirectional spherical harmonics for a 6D channel-knowledge map on a 6 GHz Sionna scene with up to three scattering orders, training on nine transmitter positions and testing on a tenth, and reports errors of 3.68 to 6.70 dB. **RxGS** (arXiv 2605.24290) separates receiver-independent geometry from receiver-dependent directional radiance, reaches 4.92 dBm MAE at unseen receivers against 9.7–11.6 dBm for per-receiver baselines, and trains 7–45× faster. **GRaF** (arXiv 2502.05708) generalises across scenes. A **differentiable ray tracer on Gaussians** (arXiv 2605.07781) does not learn a field at all and traces on the visual reconstruction. **GS-CG** (arXiv 2607.21099) updates a channel-gain map by keeping a frozen reference set of Gaussians and tuning a small additional set, a structure our §5 supports from measurement. OctCGS, XFreq-GS, GSRF, GS-IR, RadioSight and WEDT were verified at title level only. **WiNeRT** (ICLR 2023) reports time-of-flight errors below 0.33 ns with a neural ray tracer, the only delay number in this literature we could verify.

None of these works reports a naive baseline, a decoded physical quantity with its floor, a reproduction of another method, or a check of the data pipeline's cross-view consistency. That is the gap.

## 3. Anchor, pipeline and protocol

### 3.1 Scene, simulator, targets

The scene is the RF-3DGS NIST lobby mesh, ray-traced with Sionna 2.1 on the GPU (OptiX) instead of the tutorial's Sionna 0.19 on TensorFlow. Materials are either the tutorial's per-object ITU materials with the conductivity's frequency unit corrected (§6, instance 1) or a uniform scattering coefficient of 0.7 (the setting of the transmitter experiments, §5). Receivers follow the released route (800 positions), four 90° faces per position. Two kinds of target are used:

- **Beamformed spectra**, as published: CBF or MVDR spatial spectra of a 10-element uniform planar array (M = 10, 3GPP TR 38.901 pattern), ported from the tutorial's TensorFlow to torch with the angle grid cached per camera model; one dB channel, or the jet-mapped RGB of the original.
- **Path-level five-channel target** (MULTI): each ray-traced path is splatted with a Gaussian kernel (σ = 3 px = 1.0° unless stated) into five channels: path power in dB, cos and sin of the departure azimuth, departure zenith, delay. This is the target on which angles and delays are decoded (§7).

### 3.2 Trainer

gsplat 1.6 rasterises any channel count; the model keeps the RF-3DGS structure (SH degree 3 colours, opacities, geometry frozen from the visual checkpoint unless stated), trains 10k steps on the dB target with an L1 loss, and holds out 20 % of positions as whole positions (640 test images). The published trainer (INRIA) and ours agree on the anchor to 0.05 dB (Table 1).

### 3.3 Anchor

**Table 1. Anchor C0: released data, released visual checkpoint, RGB jet target.**

| run | PSNR (jet) | SSIM | time (10k steps) |
| --- | --- | --- | --- |
| released checkpoint, evaluated | 16.02 | 0.731 | — |
| our trainer, torch SH (c0) | 15.97 | 0.727 | 763 s |
| our trainer, gsplat CUDA SH (c0b) | 15.97 (bit-identical to c0) | 0.727 | 226 s |
| INRIA train.py, same settings, its own evaluation | 16.82 | — | 272 s |

Every number in this paper is relative to C0 or to a run on our regenerated data trained with the same trainer. Numbers from other papers are quoted but never placed in the same table.

### 3.4 Evaluation

Decoded azimuth, zenith and delay are computed on the pixels a path reaches (the "hit" support, power above the −150 dB kernel floor), per pixel, and reported as median / P90 / RMSE. The tail (a few percent of pixels where two paths overlap) dominates the RMSE and the median is the headline; both are always given. Power-weighted variants weight each pixel by its linear power. The dB RMSE inside the dataset's range and the PSNR after jet mapping are reported as compatibility numbers. Three splits: held-out positions (20 %, random over the route), a contiguous held-out region (§7.4), and a training-density sweep with the lookup baseline drawn from the same subset (§7.3).

### 3.5 Reporting

Timings give the full chain (generation + training + evaluation), the GPU (RTX 3060, 12 GB, shared between Windows and WSL) and whether it was exclusive; a period during which another process shared the card voided every timing taken then, and those were re-measured. Every baseline appears as "as published" and "as published + engineering fixes"; a "tutorial + receiver batching only" row was not measured and is listed as MISSING (Table 12). Seed noise floor of the transfer benefit: 0.03 dB over four seeds. The criterion of every experiment in §4–§7 was written down before the run; a prediction that failed is reported as failed (§7.5, §7.4).

**Data versions.** The generator changed twice during the work (per-view seeding → per-position seeding → kernel truncation at −150 dB). Every MVDR / CBF result was checked to be insensitive to the seeding change (§4, third row), so those series were not re-baselined; every path-level result in §7 is on the per-position-seeded, truncated generator.

## 4. Cross-view consistency dominates

Alpha compositing assumes one physical field observed consistently from every view. The published pipeline breaks that in two independent places, and each was measured with everything else held fixed.

**Table 2. Three instances of cross-view (in)consistency.**

| violation | what it breaks | cost (same scene, same everything else) |
| --- | --- | --- |
| per-image min/max normalisation of the spectrum (the tutorial's CBF / TCBF practice) | radiance consistency: one physical power maps to different pixel values in different views | MVDR, RGB target: PSNR 18.15 → 11.97 (−6.2 dB), RMSE 4.42 → 7.61 dB; CBF: 13.68 → 13.09, RMSE 7.54 → 11.03 dB; CBF at 2.4 GHz: RMSE 9.32 → 13.35 dB (+43 %) |
| per-view Monte-Carlo sampling of diffuse paths (the tutorial seeds once and draws the scattering subset per call, so the four faces of one position see four path sets) | geometric consistency | path-level target, in-range power RMSE 12.67 → 8.57 dB (−32 %) |
| the same seeding change on the beamformed MVDR target | — | 18.75 / 3.91 → 18.69 / 3.94 dB: nothing (noise floor 0.03 dB) |

The per-image normalisation numbers are the best case for the published practice: the evaluation was given the oracle range, which at deployment is not in the image at all. The third row bounds the claim: an MVDR spectrum is an array statistic over all sampled paths (a covariance inverse), and different diffuse subsets give nearly the same statistic; a path-level splat is the path set itself. Cross-view consistency therefore costs according to the target type, not the representation.

For scale, the representation changes measured on the same MVDR data are: dB target instead of jet RGB, +0.6 dB PSNR and −11 % RMSE (18.15 → 18.75 / 4.42 → 3.91; reaching 17 dB in 19 s instead of 69 s); SH degree 0 → 3, +2.0 dB; training the opacities, +0.2 dB; unfreezing the geometry, +2.8 dB (§5). The seeding effect on the path-level target (−32 %) is larger than the geometry effect (−27 %) on the same kind of number. The 12.9 dB power RMSE of our first path-level dataset and the 8.6 dB of the current one differ almost entirely by the seeding rule (12.67 → 8.57 with the encoding, channel count and range held fixed); encoding accounts for 0.02 dB.

The released RF-3DGS tutorial carries the second violation by construction. RF-PGS's pipeline is not public, so whether it does cannot be said; BiWGS and RxGS report Sionna data without stating the seeding rule.

## 5. Geometry adaptation: distributed, low-dimensional, transmitter-specific

RF-3DGS freezes the Gaussians' positions, scales and rotations at the visual reconstruction. Letting them move is the largest single representation gain we measured, and its structure decides how a multi-transmitter deployment should store it.

### 5.1 Unfreezing

**Table 3. Unfreezing the geometry (MVDR 60 GHz, dB target, 10k steps, 1,014,142 Gaussians).**

| geometry | PSNR (jet) | SSIM | RMSE dB | notes |
| --- | --- | --- | --- | --- |
| frozen (RF-3DGS) | 18.75 | 0.818 | 3.91 | 168 s |
| unfrozen (means, scales, rotations) | 21.53 | 0.868 | 2.85 | 302 s; −27 % RMSE |
| unfrozen, then frozen with colours refit from zero | 21.55 | — | 2.81 | not a joint-training artefact |
| same on per-position-seeded data | 18.69 → 21.47 | — | 3.94 → 2.89 | gain reproduced on consistent data |
| unfrozen with MCMC densification (gsplat defaults) | 5.25 | 0.417 | 28.2 | diverges from a converged geometry; untuned, listed as a negative result |

Means move 6 mm (median); the gain is in scales and rotations. In linear-power compositing the gain is the same (−27.5 % against −27.1 % in dB), so it is not the geometry compensating a dB-domain mixing bias; each ray averages 6–8 Gaussians (N_eff median 5.9 frozen, 7.5 unfrozen, Rademacher estimate), so the two compositing rules are not trivially identical either.

### 5.2 Which Gaussians carry it

**Table 4. Unfreezing subsets (Tx-A, dB, 10k steps; gradients masked outside the subset; masks verified to leave the rest untouched).**

| subset | Gaussians | PSNR (jet) | RMSE dB | share of the full gain |
| --- | --- | --- | --- | --- |
| none | 0 | 18.75 | 3.91 | 0 |
| random 1 % | 10k | 20.72 | 3.14 | 73 % |
| random 3 % | 30k | 21.00 | 3.02 | 84 % |
| random 11 % | 111k | 21.27 | 2.91 | 94 % |
| discs only (11 %; s_mid/s_min > 3) | 111k | 21.49 | 2.86 | 99 % |
| needles only (40 %; s_max/s_mid > 3) | 404k | 21.32 | 2.92 | 93 % |
| all | 1.01M | 21.53 | 2.85 | 100 % |

A random 11 % is within 0.05 dB of the discs and a random 1 % recovers 73 % of the gain: the correction is distributed and redundant, not the property of a primitive class. This test does not select 2DGS-style surfels or degree-1 incoming-direction encodings as the key variable. The needles lie in the surface plane (their long axis is 2.8° from the tangent plane, 92 % within 20°), and adaptation tilts them slightly (to 5.2°).

### 5.3 How far an adapted geometry travels

Adapting on transmitter A and evaluating on transmitter B (5.8 m away) decomposes the gain: B's in-range RMSE is 4.82 dB with the visual geometry, 4.58 with A's adapted geometry and 3.83 with its own, so 24 % of the gain transfers and 76 % is transmitter-specific. A third transmitter's geometry carried to B is worse than the visual one (4.95), so the transferable part is not generic capacity.

**Table 5. Ten source transmitters, transfer to Tx-B (adaptation 10k steps; transfer 2k steps, calibrated against 10k: same signs, same order; seed noise 0.03 dB). Reference on Tx-B at 2k steps: visual geometry 5.54, B's own adaptation 4.78 dB.**

| source | zone | distance to B | in-range RMSE | benefit | strong-pixel IoU with B | dB correlation with B |
| --- | --- | --- | --- | --- | --- | --- |
| D | east corridor, west of B | 1.77 m | 4.97 | +0.57 | 0.432 | 0.634 |
| E | east corridor | 3.23 m | 5.42 | +0.12 | 0.417 | 0.492 |
| N | east hall | 4.18 m | 5.20 | +0.34 | 0.357 | 0.520 |
| L | hall south | 5.29 m | 5.63 | −0.09 | 0.358 | 0.482 |
| A | east corridor, north end | 5.81 m | 5.24 | +0.29 | 0.388 | 0.457 |
| O | hall centre | 6.76 m | 5.59 | −0.06 | 0.338 | 0.374 |
| H | south corridor | 7.42 m | 5.80 | −0.27 | 0.317 | 0.420 |
| C | hall centre, 2 m high | 8.54 m | 5.89 | −0.35 | 0.215 | −0.142 |
| M | hall south-west | 9.41 m | 5.71 | −0.17 | 0.283 | 0.294 |
| K | north-west corner | 12.81 m | 5.83 | −0.29 | 0.262 | 0.145 |

The four sources on B's side of the lobby (x ≥ 5 m) all help (+0.12 to +0.57, mean +0.33 dB); the six elsewhere all hurt (−0.06 to −0.35, mean −0.20 dB), whatever the distance: a source 5.3 m away in the hall is as useless as one 12.8 m away. An exponential-plus-plateau fit gives a correlation length of 5.3 m, but its bootstrap 90 % interval is 2 to 40 m and 17 % of resamples find no decay at all; the plateau, −0.46 dB (bootstrap median −0.50), is the firm number. Benefit correlates −0.81 with distance, +0.85 with the overlap of strongly lit surfaces and +0.77 with the correlation of the two transmitters' dB spectra; distance and overlap are collinear on this route and neither explains the benefit alone.

### 5.4 The parameter-space view

Stacking the ten adapted geometries as deformations of the visual one (log-scales, rotation vectors, displacements, each block normalised to unit RMS): five principal components carry 66 % of the energy (the spectrum is flat, so no low-rank basis emerges from independent adaptations); the mean deformation carries 26 % of the energy, matching the 24 % measured functionally; the pairwise cosine is 0.18 on average, 0.37 at most, and decreases with distance (correlation −0.62), with the four most similar pairs all east-side neighbours. Predicting a held-out transmitter's deformation by interpolating its neighbours' coefficients is no better than using the mean (relative error 0.9–1.0 against 0.93) and the nearest neighbour's deformation alone is worse (1.18).

### 5.5 Consequence

The plateau, the orthogonality and the 1 % result say the same thing: a per-transmitter adaptation is a small, local, directional correction that is a liability elsewhere. A multi-transmitter deployment should keep the visual geometry and a per-transmitter delta on 1–3 % of the Gaussians, and never serve a building from one adapted geometry; reuse should be decided by lit-surface overlap (IoU above about 0.36 here), not by radius. Whether a jointly trained shared basis would recover more than the 24 % is untested.

## 6. The metrics do not see the physics (four instances)

1. **Materials.** The tutorial's conductivity formula divides the frequency by 1e-9 instead of 1e9, so every ITU material has a conductivity of 1e16 to 1e24 S/m: every wall is a perfect conductor. Reproduced as published, the lobby returns 6,753 paths; with the unit corrected, 104,443; with uniform scattering, 311,372. The total received power differs by 1 dB (−80.8 / −79.7 / −79.9 dB) and PSNR is unaffected, but the multi-bounce share (power beyond first order) goes from 13.7 % mean (median 7.7 %, max 40.6 %) to 31 % mean (max 98.7 %) at 2.4 GHz.
2. **Carrier.** The released datasets are at 2.4 GHz: on the released receiver poses, the CBF power the release records matches our 2.4 GHz computation within 1.6 dB (std 4.2, correlation 0.96) and the 60 GHz one by 34.2 dB. The paper's 60 GHz narrative holds for its PSNR (the spectrum's shape is frequency-normalised by the array) and not for its physics.
3. **Depth-1 twins.** The power a first-order ray tracer misses is invisible to PSNR. At 60 GHz with uniform scattering it is 5.7 % (median over 40 positions) but 23–41 % at the corridor ends; at 2.4 GHz with corrected materials 45–82 % at the worst four positions; as published, 97–100 % there. For transmitter placement it does not change the ranking of candidates (per-receiver solves at 1M samples: depth 3 adds +0.15 dB median, +1.1 to +1.3 dB at the worst decile, +2.5 dB at most, and 2–3 points of coverage; the depth-1 optimum stays the best at depth 3) but it changes the coverage numbers where coverage matters most.
4. **Kernel width.** The splat width σ that minimises the decoded error is a property of the metric. On a fixed support (the σ = 1 px hit pixels), azimuth is best at σ = 3 px = 1.0° (0.56° / 3.9° median / P90) and degrades either side; by pixel count on each dataset's own support the minimum drifts to 2° because a wider kernel adds easy pixels. Power-weighted, azimuth is best at the grid limit σ = 1 px = 0.33° (0.84° / 2.15°) and worsens monotonically with σ, while zenith prefers the widest kernel (median 0.61° at 13 px, RMSE 2.6° at 6 px). The asymmetry is the scene's: 77 % of arriving power sits within ±5° elevation and 98 % within ±15°, so the azimuth axis is crowded and a wide kernel pollutes neighbours, while the zenith axis is sparse and a wide kernel only smooths. The downstream task decides: beam selection on strong paths (§7.6) picks the power-weighted answer and with it σ = 1 px.

## 7. Decoded physical accuracy with a floor and a ceiling

### 7.1 Encoding and ceiling

**Table 6. Decoded error of the path-level target (640 held-out images, hit pixels), encodings compared.**

| representation | azimuth | zenith | delay |
| --- | --- | --- | --- |
| tutorial's angle × amplitude RGB (AoD3), decoded by channel ratio | 48.6° RMSE | 56.6° RMSE | no channel |
| one channel per quantity, azimuth as a single wrapped angle (same dataset, same five channels) | 1.86° / 11.7° / 19.1° | 0.96° / 11.8° / — | 2.44 / 12.8 / — ns |
| one channel per quantity, azimuth as (cos, sin) | 0.59° / 5.85° / 9.8° | 0.94° / 12.1° / 14.7° | 2.42 / 12.8 / 8.9 ns |
| the target's own encoding (ceiling) | 0.20° | — | — |

The seam at ±180° costs its own channel 3.2× at the median and 2× at P90 and RMSE, and nothing else: zenith, delay and power agree within 0.02 between the two encodings, so a feature one channel cannot fit does not leak into the others. Separating the channels makes gsplat's arbitrary channel count pay: the angle RMSE falls to a fifth of the tutorial's encoding.

### 7.2 The floor

**Table 7. Naive baselines on the same 640 held-out images (nearest training position 0.23 m median, 0.36 m P90). Median / P90 / RMSE.**

| predictor | azimuth | zenith | delay (ns) |
| --- | --- | --- | --- |
| constant (circular mean azimuth, mean zenith, mean delay of the training pixels) | 27.6° / 68° / 49° | 7.2° / 20.7° / 14.1° | 9.5 / 25.6 / 15.7 |
| nearest training position, same yaw, copied (no training) | 1.07° / 131° / 55° | **0.79°** / 90° / 38° | **0.86** / 44 / 20 |
| two nearest, inverse-distance interpolation | 0.81° / 10.8° / 37° | 1.01° / 61° / 33° | 1.06 / 32 / 17 |
| field (RRF, cs, σ = 3 px) | **0.59°** / **5.85°** / **9.8°** | 0.94° / **12.1°** / **14.7°** | 2.42 / **12.8** / **8.9** |

The constant predictor shows that no channel is degenerate. At the published density the field's median gain over copying is 1.8× on azimuth, it loses on the zenith median by 0.15° and on the delay median by 3× (a 23 cm move is 0.77 ns of geometric delay; the learned delay channel is worse than that). The field's gain is in the tail: copying decodes to garbage wherever the path set changes between neighbours (P90 131° against 5.85°), the field is continuous. A headline of 0.59° without this table would mean nothing.

### 7.3 Density

**Table 8. Training-density sweep. Same 640 held-out images; copy drawn from the same training subset; subsets even along the route; field with the delay decomposition of §7.5 (accumulated-depth variant). Medians; field P90 in the last column.**

| training positions | nearest training position, median / P90 / max | azimuth copy / field | zenith copy / field | delay (ns) copy / field | field P90 azimuth / delay | field PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| all 640 (the 800-position route minus the held-out 160) | 0.23 / 0.36 / 0.60 m | 1.07° / **0.62°** | **0.79°** / 0.96° | **0.86** / 0.95 | 6.3° / 6.6 ns | 16.14 |
| 160 | 0.35 / 0.57 / 0.77 m | 1.68° / **0.63°** | 1.32° / **0.93°** | 1.38 / **0.97** | 6.4° / 6.5 ns | 16.08 |
| 80 | 0.51 / 0.76 / 1.08 m | 2.55° / **0.66°** | 1.82° / **0.97°** | 2.07 / **1.05** | 6.4° / 6.7 ns | 16.00 |
| 40 | 0.66 / 1.05 / 1.46 m | 3.72° / **0.75°** | 2.905° / **1.13°** | 3.12 / **1.25** | 7.0° / 7.3 ns | 15.72 |
| 20 | 0.84 / 1.57 / 2.29 m | 4.68° / **0.84°** | 2.913° / **1.24°** | 3.88 / **1.43** | 6.8° / 7.4 ns | 15.37 |

Lookup degrades linearly with spacing (azimuth 1.07° → 4.68°, ×4.4 from 0.23 to 0.84 m); the field is nearly flat (0.62° → 0.84°, ×1.3; delay 0.95 → 1.43 ns). By linear interpolation the crossovers are below 0.23 m for azimuth (the field already wins at the densest sampling), about 0.27 m for zenith and 0.25 m for delay: beyond about 0.3 m spacing the field beats lookup on all three physical channels. The field's tail does not depend on density (P90 azimuth 6–7° at every level; copy 131–152°). Twenty positions, eighty images, give a 0.84° median azimuth.

The route passes the same areas several times, so an even subsample along it is already close to spatially uniform (the nearest-training distance grows 3.65× for 32× fewer positions, 640 → 20, against 5.7× for a uniform 2D thinning). A farthest-point subsample in space, the strongest lookup a subset of that size can get, gives the same numbers within 0.2° and 0.15 ns at 80 and 40 positions (copy azimuth 2.53° / 3.81°, field 0.69° / 0.76°; delay copy 2.01 / 2.98 ns, field 1.02 / 1.16 ns), so the crossover does not depend on the subsampling rule.

The flatness is the frozen visual geometry carrying the spatial structure; the crossover is therefore a property of that geometry's quality as much as of the scene, which is why §9 reports the visual reconstruction quality of both scenes. The crossover is a geometric quantity, not a wavelength one: the channels encode power, angles and delay, not phase, so there is no λ/2 oscillation, and the crossover is set by how fast the multipath structure changes with position (the density of shadow boundaries). It should be roughly stable across carrier frequency and move with the scene's feature scale; on the corridor scene it moves to about 0.5 m (Table 14).

### 7.4 Extrapolation

**Table 9. Held-out region: the south-east corridor end (x > 4, y < −7), 123 positions / 492 images, training on the other 677 positions; nearest training position 1.43 m median, 2.35 m P90, 2.75 m max. Median / P90 / RMSE.**

| predictor | azimuth | zenith | delay (ns) |
| --- | --- | --- | --- |
| constant | 60.0° / 80° / 63° | 9.1° / 23° / 18° | 13.4 / 36 / 20 |
| nearest copy | 7.38° / 93° / 52° | 4.29° / 98° / 48° | 5.95 / 52 / 29 |
| two-neighbour interpolation | 5.85° / 88° / 45° | 6.30° / 94° / 45° | 6.95 / 49 / 28 |
| field, delay learned directly | **1.47°** / 17.6° / 20.6° | **1.99°** / 27.0° / 20.4° | 6.78 / 23.2 / 14.6 |
| field, delay = residual + depth / c | 1.60° / 18.9° / 21.2° | 2.08° / 28.2° / 20.6° | **3.07** / **14.0** / **11.8** |

With no training point within 1.4–2.8 m, the field's azimuth median is 1.47° against 7.38° for lookup (5×), zenith 1.99° against 4.29°, P90 17.6° against 93°. From interpolation to extrapolation the field degrades 2.5× and lookup 7×; PSNR falls from 16.1 to 14.1. The published claim of "extrapolation to unvisited locations" is supported at this distance on this scene. Delay is the channel that suffers: learned directly it is worse than lookup (6.78 against 5.95 ns) and only its tail wins; with the decomposition of §7.5 it becomes the strongest channel relative to lookup (3.07 ns). This region is also the one with the highest multi-bounce share (45–82 % missing at depth 1 with corrected materials), so the held-out set is the hardest part of the training data and the result is conservative.

### 7.5 Delay as analytic range plus learned residual

The floor showed the learned delay 3× worse than copying a neighbour. A path's delay is τ = τ_scatter + |μ − p| / c: the scatterer's delay plus the Gaussian-to-receiver range over c. The range is a distance, not a direction; a directional SH colour cannot represent it, and it dominates the channel (3 m of depth is 10 ns). gsplat renders the composited depth natively, so the delay channel receives Σ wᵢ dᵢ / c from the rasteriser and the learned channel only has to carry the view-independent τ_scatter. The change is twenty lines (a render mode and a division by c in the channel's normalised units).

**Table 10. Delay decomposition (same data, same training; 640 held-out images).**

| range term | signed mean error | delay median / P90 / RMSE (ns) | power-weighted median / P90 | pixels with \|e\| > 5 ns (share negative) | azimuth median / P90 | zenith median / P90 |
| --- | --- | --- | --- | --- | --- | --- |
| none (learned directly) | −1.85 ns | 2.42 / 12.8 / 8.93 | 3.54 / 9.28 | 27.4 % (65 %) | 0.59° / 5.85° | 0.94° / 12.1° |
| accumulated depth Σ wᵢ dᵢ | −1.10 ns | 0.95 / 6.62 / 6.93 | 0.40 / 2.06 | 12.8 % (69 %) | 0.62° / 6.29° | 0.96° / 12.8° |
| expected depth Σ wᵢ dᵢ / α (default) | −0.64 ns | **0.88 / 5.38 / 5.26** | 0.39 / 2.02 | 10.7 % (67 %) | 0.63° / 6.86° | 1.00° / 14.1° |
| nearest copy (0.23 m), for reference | — | 0.86 / 44 / 20 | 0.44 / 1.01 | — | 1.07° / 131° | 0.79° / 90° |

Median 2.42 → 0.88 ns (2.8×), P90 12.8 → 5.4, RMSE 8.9 → 5.3; the power-weighted median 3.54 → 0.39 ns, level with lookup's 0.44. Power, azimuth and zenith are unchanged within 0.04° at the median (PSNR 16.13 → 16.14); the angle tails pay 0.6° / 1.3° at P90 for the expected-depth variant because the shared opacities now also serve the range term. In extrapolation (Table 9) the decomposition removes 3.71 ns of median error against 1.47 ns in interpolation, but the ratio (2.2×) is smaller than in interpolation (2.5×) because the learned residual degrades out of the training region while the analytic term does not. We had predicted the ratio would be larger; the mechanism we had written down concerned the absolute quantity, and the criterion was written in the wrong unit; both are reported.

Two predictions about the remaining tail were tested and one failed. The tail is not multi-path mixing: stratified by paths per pixel, single-path pixels (55 % of pixels) carry 55 % of the squared error after decomposition (median 0.86, RMSE 4.67 ns) while pixels with six or more paths have the lowest RMSE (3.54 ns); the convex-combination bias that explains the azimuth tail does not explain the delay tail. The tail is negative (65–69 % of the pixels with more than 5 ns error predict too short), consistent with the accumulated depth Σ wᵢ dᵢ being shortened by (1 − α) where the pixel's opacity is below one (10 m at α = 0.95 is −1.7 ns); normalising to the expected depth halves the bias (−1.10 → −0.64 ns) and removes a quarter of the RMSE. The residual bias points at low-weight Gaussians in front of the surface (a median depth would test it; gsplat has no kernel for it) and at the residual channel's own fit. The density and region tables use the accumulated-depth variant; the two differ by 0.07 ns at the median.

RF-PGS computes time of flight purely from the geometry and reports no delay error; BiWGS and RxGS do not encode delay; WiNeRT reports below 0.33 ns with a neural ray tracer on a different scene. The analytic-plus-residual delay is, to our knowledge, new for a Gaussian radiance field.

### 7.6 Beam selection

**Table 11. Top-k beam selection accuracy with a square θ₃dB codebook (θ₃dB = 101.5° / M); the true beam is among the k cells nearest the decoded direction. Unweighted | power-weighted over pixels.**

| dataset | array / θ₃dB | method | top-1 | top-3 | top-5 |
| --- | --- | --- | --- | --- | --- |
| σ = 3 px | M = 10 / 10.2° | nearest copy | 0.60 \| 0.65 | 0.74 \| 0.82 | 0.76 \| 0.90 |
| | | field | **0.68 \| 0.70** | **0.84 \| 0.91** | **0.87 \| 0.94** |
| | 64 × 64 / 1.59° | nearest copy | 0.22 \| 0.23 | 0.41 \| 0.47 | 0.50 \| 0.52 |
| | | field | 0.28 \| 0.10 | 0.50 \| 0.22 | 0.59 \| 0.31 |
| σ = 1 px | M = 10 / 10.2° | nearest copy | 0.38 \| 0.49 | 0.52 \| 0.77 | 0.56 \| 0.89 |
| | | field | **0.55 \| 0.87** | **0.72 \| 0.96** | **0.76 \| 0.97** |
| | 64 × 64 / 1.59° | nearest copy | 0.10 \| 0.10 | 0.20 \| 0.17 | 0.26 \| 0.20 |
| | | field | **0.19 \| 0.19** | **0.36 \| 0.45** | **0.44 \| 0.61** |

On the array the data were generated for (M = 10) the field is a usable selector: 9–10 points of top-1 over lookup, power-weighted top-3 of 0.91 (σ = 3) or 0.96 (σ = 1). On the 64 × 64 array the paper headlines, the σ = 3 field loses to lookup on strong paths (power-weighted top-1 0.10 against 0.23: the strong pixels' median azimuth and zenith errors of about 1.5° exceed the 1.59° cell), the σ = 1 field recovers (0.19 against 0.10; top-5 0.61 against 0.20), and top-1 stays at 19 %: not a top-1 selector on that array. This connects instance 4 of §6 to a task: by pixel count σ = 3 is better, for beam selection σ = 1 is, and the task chose the metric and with it the kernel. The published pointing error of 5.94° was obtained on M = 10 data, where it is 0.58 of a beamwidth; on the 64 × 64 array it would be 3.7 beamwidths.

## 8. Cost under the contract

**Table 12. Full chain for one transmitter on an exclusive RTX 3060 (medians, warm-up excluded).**

| chain | generation (3200 views) | training (10k steps) | evaluation | total |
| --- | --- | --- | --- | --- |
| as published: Sionna 0.19 + TensorFlow on the CPU, INRIA trainer, jet RGB | 1.0 s / view → 0.9 h | 272 s | — | ≈ 1 h |
| as published + engineering fixes: Sionna 2.1 on the GPU (2.4 GHz, 14.4 views/s), INRIA trainer | 222 s | 272 s | — | ≈ 8.2 min |
| as published + receiver batching only (Sionna 0.19) | MISSING | 272 s | — | MISSING |
| this work, 800 positions (60 GHz MVDR, 10.3 views/s; gsplat, dB target) | 311 s | 168 s (rgb 211 s) | 35 s | ≈ 8.6 min |
| this work, 160 positions (10k steps there cost −0.06 dB PSNR), trained to 17 dB | 33 s | 32 s (to 17 dB; 10k steps 184 s) | 35 s | ≈ 1.7 min |

Rasteriser step (render + L1 + backward to SH and opacity, 1.01M Gaussians, median of 30): INRIA against gsplat 5.6 vs 4.4 ms at 300 × 200, 12.0 vs 5.1 at 600 × 400, 31.3 vs 9.5 at 1200 × 800, 113.2 vs 25.5 ms at 2400 × 1600. The single-face 300 × 200 number is not extrapolated (kernel launch and Python dominate there); at the two large resolutions the ratio is 3.3× and 4.4×. The speed-up of the chain decomposes as: engineering (Sionna 2.1 on the GPU instead of 0.19 on the CPU) takes the chain from ≈ 1 h to ≈ 8.2 min (≈ 7×); method contributes 272 → 168 s at equal step count (gsplat with the dB target, 1.6×), 17 dB reached in 19 s instead of 69 s, and a dataset of 160 instead of 800 positions at −0.06 dB (generation 311 → 33 s), which together take the chain to ≈ 1.7 min at the 17 dB quality point (≈ 5× more). The two rows are not at equal quality and are not added into one factor. RF-PGS reports 3 min 53 s on the same lobby for a different data size; a Gaussian ray tracer needs no training at all.

The many-receiver planner solve used for transmitter placement was found to be truncated by the solver's default path cap (1e6 paths per source): with 59 receivers its coverage fell from 0.593 to 0.441 as the sample budget rose from 20k to 400k; with the cap at 1e7 it is 0.593 at every budget and depth. Coverage of 239 receivers at 50k samples takes 0.084 s (median of 5); five optimisation steps 1.75 s.

## 9. A second scene `[MISSING]`

To separate topology from the visual geometry's quality, a procedural corridor scene was built with the same pipeline: a 28 m corridor 2.4 m wide, three rooms of 5 × 4.5 m on each side with 0.9 × 2.1 m door openings, and a 10 × 8 m hall; ITU materials (plasterboard, concrete, ceiling board) with uniform scattering 0.7; textured walls so that the visual reconstruction has something to fit; a receiver route at 0.2 m spacing (574 positions) and thirteen transmitters including three same-room pairs. Only three claims are replicated, with the predictions written before any run:

1. Zone structure is harder than in the lobby: the lit-surface overlap between rooms is near zero, the exponential fit fails outright, and the binary separation (same room / adjacent room / opposite room / corridor / hall) is sharper than Table 5's.
2. The density crossover moves to a smaller spacing than 0.23–0.35 m: more occlusion, a shorter spatial correlation length, lookup fails earlier.
3. Extrapolation into a held-out whole room is worse than into the held-out corridor end of Table 9.
4. The two consistency magnitudes (per-image normalisation, per-view seeding) keep their signs; their sizes may differ.

**Table 13. Stage-1 visual reconstruction quality of both scenes (the crossover depends on it).**

| scene | Gaussians | resolution | PSNR on training views | PSNR on held-out views |
| --- | --- | --- | --- | --- |
| lobby, released checkpoint | 1,014,142 | 1600 × 900 | 39.4 dB (809 views; median 39.8, P10 35.9, min 29.4) | none exist (trained on every frame) |
| corridor, our checkpoint (30k steps) | 582,363 | 800 × 450 | 26.0 dB (821 views, against the 64-spp targets it was trained on) | 25.3 dB against the 64-spp targets (their Monte-Carlo noise floor is 27.2 dB); 28.8 dB against clean 512-spp re-renders of the same 43 poses (median 29.2, P10 25.0, min 20.4) |

The two rows are not the same protocol: the lobby checkpoint was trained on every clean Blender frame and has no held-out views; the corridor was trained on 64-spp Monte-Carlo renders, so its held-out PSNR against those targets is capped by their noise, and the clean-target number is the one to read. The first two corridor reconstructions (17–18 dB, 62–74k Gaussians) failed because the camera frame handed to the renderer was rotated 180° about the optical axis relative to the poses the trainer reads (Mitsuba's camera x axis points left); the corrected dataset trains normally.

**Table 14. Corridor density sweep (MULTI, 2.4 GHz, transmitter at the corridor's middle, delay decomposition; 452 held-out images = 113 positions × 4; copy from the same training subset). Medians; P90 in parentheses for the first and last rows.**

| training positions | nearest training position, median / P90 / max | azimuth copy / field | zenith copy / field | delay (ns) copy / field | field PSNR |
| --- | --- | --- | --- | --- | --- |
| all 451 | 0.47 / 0.64 / 1.05 m | 1.85° (159°) / **1.84°** (15.2°) | 2.99° (95°) / **2.23°** (22.4°) | **1.63** (40) / 1.85 (8.5) | 23.80 |
| 225 | 0.57 / 0.89 / 1.08 m | 2.72° / **1.76°** | 3.20° / **2.21°** | 2.22 / **1.82** | 23.50 |
| 113 | 0.80 / 1.28 / 1.93 m | 4.33° / **2.07°** | 4.32° / **2.41°** | 3.64 / **2.36** | 23.37 |
| 56 | 1.10 / 2.12 / 2.70 m | 6.81° / **3.06°** | 12.74° / **2.85°** | 9.70 / **3.81** | 22.57 |
| 28 | 1.33 / 2.82 / 3.71 m | 4.45° (170°) / **3.57°** (30.1°) | 7.77° / **4.54°** | 6.93 (52) / **4.14** (15.8) | 21.35 |

The spacing axis is the measured nearest-training distance in both scenes: the corridor route has a nominal 0.2 m step, but the generator applies the tutorial's ±0.5 m horizontal and −1.0 / +0.3 m vertical jitter to every position, so the densest corridor level is 0.47 m against the lobby's 0.23 m. The structure of Table 8 repeats: lookup degrades roughly linearly with spacing (azimuth 1.85° → 4.5–6.8°, zenith 3.0° → 7.8–12.7°, delay 1.6 → 6.9–9.7 ns from 0.47 to 1.33 m), the field slowly (1.84° → 3.57°, 2.23° → 4.54°, 1.85 → 4.14 ns), and the field's tail is its advantage (P90 azimuth 13–30° against 159–172°; the corridor's azimuth is bimodal, so copying a neighbour whose dominant direction is the opposite one costs 180°). The 28-position copy row is better than the 56-position one; that is the sampling noise of a 28-position subset and is reported as measured.

**Prediction 2 failed.** The crossovers are at about 0.47 m for azimuth (a tie at the densest level), 0.50 m for delay and below 0.47 m for zenith, that is, at a *larger* spacing than the lobby's 0.23–0.35 m, not a smaller one. Both curves moved. Lookup is better in the corridor at equal spacing (1.85° at 0.47 m against the lobby's 2.55° at 0.51 m): a corridor is a waveguide whose multipath structure hardly changes under translation along its axis, and the door openings that the prediction's mechanism relied on are a small part of the route. The field is worse in the corridor (1.84° against 0.62° azimuth, 2.23° against 0.96° zenith, 1.85 against 0.95 ns delay at the densest level); the candidates are the visual geometry's quality (Table 13: 582k Gaussians and 28.8 dB against 1.0M and 39.4 dB, which is why both stage-1 rows are reported) and the corridor's long-range paths, and they were not separated. What holds in both scenes is the form of the statement: beyond a scene-dependent spacing, about 0.3 m in the lobby and 0.5 m in the corridor, the field beats lookup on all three physical channels, and the spacing is set by the scene's spatial correlation length, the mechanism written before the run with the wrong sign.

`[MISSING: Tables 5 and 2 replicated on the corridor (zone structure on room_S2, the two consistency magnitudes); the extrapolation claim (prediction 3) is not in the queue.]`

## 10. Limits

One lobby with two carriers, two material sets and one array; the corridor scene is pending. All data are simulated; the consistency result is about the simulator's sampling, and a measured dataset has its own inconsistency (calibration drift across a survey) that we did not model. MCMC densification is untuned and listed as a negative result; end-to-end differentiability to materials was not attempted; a jointly trained shared deformation basis is untested; the delay residual's remaining 4.7 ns single-path RMSE has no mechanism yet; the M = 64 spectra of the original paper's headline array were not regenerated (the ported beamformer caches an [M², H, W] grid and exceeds 12 GB). The transmitter-transfer timings were taken on a shared GPU and are not reported.

## 11. Conclusion

In the RF-3DGS setting, how consistently the data pipeline observes the field across views matters more than any change to the representation for path-level targets and not at all for beamformed ones; geometry adaptation is a small, distributed, transmitter-specific correction that must not be shared across zones; the metrics in use see none of this, and decoded physical accuracy reported against a lookup floor does. The one algorithmic change that survived this protocol, the analytic delay range, came from the floor, not from a paper.

## Tables and figures (planned)

T1 anchor; T2 consistency; T3 unfreezing; T4 subsets; T5 transfer by source (with F1: benefit against distance, zone-coloured, both budgets, bootstrap band); F2 deformation spectrum and cosine matrix; T6 encodings; T7 floor; T8 density (with F3: copy and field against spacing, both subsampling rules); T9 region; T10 delay decomposition (with F4: signed error histogram and paths-per-pixel strata); T11 beam selection; T12 cost; T13 stage-1 quality of both scenes.
