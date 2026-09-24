# T4: the radio radiance field against traditional channel models (plan, 2026-09-24)

Ke (2026-09-23/24): earlier papers never compare with the traditional channel models (3GPP TDL / CDL); compare in
one scene, draw conclusions, ideally with a stochastic dataset; later: a detailed comparison with channel datasets.

## What is compared, and against what truth

Truth: the Sionna RT spectra of `3dgs_MVDR_100_gpct` (NIST lobby, 60 GHz, Tx (6.905, 0, 0.287), 10 x 10 UPA,
MVDR over 0.1 ns delay taps), held-out positions = the 160 test positions x 4 faces (640 views) every RRF model
of rounds 41-45 was scored on. Measured data (NIST's 60 GHz lobby campaign, used by RF-3DGS and RF-PGS) is not
public; asking for it is Ke's decision.

Predictors of a held-out view:

| id | predictor | site-specific? | uses the truth's training positions? |
| --- | --- | --- | --- |
| RRF | the trained radio radiance fields (SH3 base, SH3 + head, ...) | yes | yes (trained on them) |
| NN | the spectrum of the nearest training position, same face | yes | yes |
| LOS | one free-space direct path (FSPL, the 38.901 element patterns), no blockage | geometry only | no |
| InH-open / InH-mixed | 3GPP TR 38.901 v19.2 indoor-office system-level channel (Sionna 2.1) at the exact Tx / Rx coordinates, LOS direction from the geometry, 10 realisations | no | no |
| (later) Q-D | NIST quasi-deterministic model (qd-realization, MATLAB) on the lobby's geometry | geometry + stochastic diffuse | no |

InH is generated as a CIR with Sionna's PanelArray set to the ray tracer's receive array (10 x 10, lambda/2,
38.901 element, V pol) and transmit element (1 x 1, 38.901, V), each face's yaw as the UT orientation; the element
axis is permuted to rf_spectra's ordering; path loss and shadow fading on; delays shifted by the LOS distance / c
(38.901 delays are relative). The spectra go through the same `merge_paths_to_time_grid` + `mvdr_spectrum` as the
truth. Caveat to report: a 38.901 cluster's 20 rays share one delay, so the delay-tap covariance has rank about
the number of clusters (~20) < 100 elements -- MVDR needs diagonal loading for InH (and LOS), which the truth did
not use; the loading (1e-3 of tr(R) / M^2) is stated with every number.

## Metrics (three levels)

1. Per view (`mvdr_peaks.py`, as for the RRF models): main-peak direction error and <= 1 deg share, power at the
   true peak, main-peak power error, top-3 detection, false-peak rate. Levels: InH / LOS carry their own absolute
   level (Tx power and path-loss conventions differ from the ray tracer), so the level-dependent metrics are also
   given after one global offset (the median over views of truth max - prediction max); the direction metrics
   are level-free.
2. Distribution over the scene: RMS delay spread, azimuth / zenith spread of arrival, K-factor, path gain vs
   distance -- from the RT paths (recomputed at the test positions), from InH's rays, and from the spectra where
   only spectra exist; KS and Wasserstein distances to the RT distributions.
3. A downstream task: receive-beam selection (the 10 x 10 DFT codebook) chosen on each predictor, gain achieved
   on the truth's channel -- the "so what".

## Feasibility notes (2026-09-24, 01:45)

- **Q-D** (`wigig-tools/qd-realization`, MATLAB R2026a is installed): its input is an AMF XML CAD (mm), a material
  library and node position files; it traces by the method of images. The lobby mesh has 707,537 triangles in 254
  shapes (Blender-modelled; many spheres and small cubes), far beyond what the method of images handles. A Q-D run
  needs a simplified planar lobby (walls, floor, ceiling, the large cuboids; order-1 reflections as in our RT);
  its MPCs then go through the same response -> MVDR as InH. Deferred behind InH.
- **CDL** (38.901 CDL-A..E, Sionna): a fixed cluster table per profile, the same for every position; as a
  link-level reference, CDL-D / E (LOS) with the mean AoA set to the geometric LOS direction and the delay spread
  scaled to InH's median. Weaker than InH by construction (no geometry beyond the LOS angle); added if time allows.
- **Level 2** needs the RT paths at the test positions (not stored by the generator): t6_incoherent_mvdr.py
  already re-solves 20 positions with the dataset's seeds; the LSPs (DS, ASA, ZSA, K) come from the same solve.

## Measured datasets found (2026-09-24 night; none has the NIST lobby)

| dataset | what | usable for |
| --- | --- | --- |
| Scazzoli, de Santis, Linsalata, Santucci, Spagnolini, Magarini, "Indoor 60 GHz Radio Channel Dataset Enabling Digital Twin Construction" (arXiv 2605.05824), IEEE DataPort DOI 10.21227/bv69-n602 | 60 GHz, RFSoC + Sivers 2 x 8 phased array; 350 Rx points on a 1.95 x 3.60 m grid in an office room, Tx in the corridor through a doorway; at every point the full 63 x 63 Tx x Rx beam sweep (21 azimuths -54..+54 deg x 3 elevations -18 / 0 / +18 deg) with RSS and power delay profiles | a measured per-position beamspace map: beam-selection task (level 3) and spatial interpolation (NN / 38.901 InH / a learned field) on real data. No 3D model released (a photo and a figure of a reconstruction), so a 3DGS radiance field would first need a visual capture of the room |
| Rastorgueva-Foi et al., "Millimeter-wave Radio SLAM" (IEEE JSAC 2024, arXiv 2312.13741), IEEE DataPort open access (free login) | 60 GHz bistatic OFDM radar with 5G PRS, Tampere University Kampusareena ground floor (a large hall); raw I/Q (129 GB) and post-processed per-path AoA / AoD / ToA / power (48 KB), MATLAB scripts | level 2 on measured paths: 38.901 InH's delay and angle spreads against a measured hall at 60 GHz. No floor plan or 3D model listed |
| NIST 60 GHz lobby (Gentile et al., context-aware channel sounder, EuCAP 2024; the data RF-3DGS and RF-PGS used) | the scene of our dataset | the only same-scene measurement; not public -- asking NIST is Ke's decision |

Neither public dataset can replace the NIST lobby for a same-scene comparison of the radio radiance field; both can
check the stochastic side (is InH's spread realistic at 60 GHz indoors) and give a measured beam-selection task.

## Pre-registered expectations (written before any run)

- Level 1: RRF and NN far better than InH on direction (InH's clusters are random around the LOS direction);
  LOS very good on direction at LOS positions, bad at NLOS ones. The risk worth knowing: LOS may beat the RRF on
  the main-peak direction at LOS positions (the RRF's median is 4-5 deg).
- Level 2: InH's generic distributions off the lobby's (e.g. its delay spread); the RRF's closer.
- Controls: (a) analytic -- a single known ray through the InH -> spectrum path must peak at that ray's direction
  (< 1 deg with the 10 x 10 array's beamwidth); (b) at the positions the truth calls LOS, InH's main peak should
  be near the geometric direct direction whenever its K-factor is large (the element-order mapping check).

### Implementation and pass criteria (written 2026-09-24 ~02:00, before any T4 run)

Scripts: `sionna_port/t4_baselines.py` (control, los, nn, inh_open / inh_mixed x 10; InH also writes each view's
channel LSPs), `sionna_port/t6_incoherent_mvdr.py --positions 0` (re-solves every held-out position with the
dataset's seeds: P1's test, plus the ray-traced LSPs and the tap covariances), `sionna_port/t4_score.py` (all
levels for every predictor).

Estimators, identical for the ray tracer and InH (per view, element patterns included): path gain = sum of the
element-averaged path powers; RMS delay spread; first-tap ratio = power within 0.1 ns of the earliest path over
the rest (a K-factor proxy that needs no path labels). Spectral spread = power-weighted RMS angle from the main
peak over the pixels within 20 dB of it (same estimator on every spectrum). Beam-gain loss (level 3) = the true
channel's best matched-beam power minus that of a beam steered to the predicted main peak (a^H R a / |a|^2, R
the ray-traced tap covariance).

Pass criteria of the pipeline itself (a failure stops the comparison until explained):

1. Control (a): peak error <= 0.5 deg through PanelArray's positions and through rf_spectra's steering (pixel
   pitch about 0.3 deg).
2. t6's coherent re-computation reproduces the stored truth: max |difference| <= 0.01 dB on every view (same
   seeds, same solver); if not, the RT stats and covariances are not the truth's channel.
3. Geometry: at views whose geometric transmitter direction is inside the image, the truth's own main peak lies
   within 2 deg of it in >= 60 % (a lobby in line of sight); below 30 % the pixel <-> world mapping is suspect.
4. InH convention (control (b)): at those LOS-dominant views, InH-open's main peak within 2 deg of the LOS
   direction in >= 50 % (38.901 InH LOS K-factor about 7 dB puts the direct ray on top); near 0 % = mirrored
   phase or element order.
5. Level 3 sanity: a beam steered to the truth's own MVDR peak loses <= 1 dB (median) against the best beam.

Smoke first (`t4_baselines.py --limit 4`: 4 held-out positions spread along the route, end points included, all
four faces each, plus the position whose strongest face is weakest; `t6_incoherent_mvdr.py --positions 3`, which
adds the weakest position and any with an empty face). Its criteria, besides 1 and 2 above: every render finite
and [200, 300]; InH's faces of one position show the same cluster delays (max difference < 1e-12 s, spatial
consistency; if not, each face is its own realisation and that is stated); run time about seconds per view for
the ray tracer (1e6 samples) and well under a minute per InH realisation. The smoke only checks the plumbing:
its handful of views says nothing about which predictor is better.

Expectations about the outcome (not pass criteria):

- Level 2: the ray tracer's delay spread (order-1 reflections in one lobby) below InH's median (38.901 InH LOS at
  60 GHz: lgDS mean -7.71, i.e. about 20 ns); the first-tap ratio higher for the ray tracer.
- Level 3: LOS within 1 dB of the RRF at LOS-dominant views; the RRF better elsewhere; InH worst.

## Results (round 47, 2026-09-24 02:50-03:10; `rrf_gsplat/rounds/win_round47_t4.sh`, `output/rrf/t4_scores*.json`)

640 held-out views of `3dgs_MVDR_100_gpct`; RRF rows are 3-seed means, InH rows 10-realisation means; "aligned" =
after one global level offset per predictor (InH / LOS carry their own absolute level). Scored against the stored
labels; against the float64 recomputation (below) every row moves by < 1.5 points and the ranking is the same.

| predictor | main-peak dir. median | <= 1 deg | beam loss on the true spectrum, median | top-3 detected | at the true peak | RMSE dB | beam-gain loss on the true channel, median / P90 (level 3) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| float64 recomputation of the truth (the labels' own noise) | 0.38 deg | 67.5 % | 0.28 dB | 59.6 % | -0.51 | 1.60 | 0.02 / 6.7 dB |
| NN: the nearest training position's truth, same face | **1.29 deg** | **39.7 %** | **2.59 dB** | 45.1 % | -2.81 | **2.62** | **0.19 / 8.8 dB** |
| RRF SH3 + shading head (B6) | 4.38 deg | 19.6 % | 4.18 dB | 67.6 % | -7.06 | 2.75 | 1.62 / 15.4 dB |
| RRF SH3 (r45_base) | 5.56 deg | 19.8 % | 4.42 dB | 68.8 % | -8.08 | 3.87 | 2.62 / 16.8 dB |
| LOS only (geometry) | 8.17 deg | 11.4 % | 8.89 dB | 12.8 % | -15.0 (aligned) | 13.2 (aligned) | 3.53 / 20.5 dB |
| 3GPP InH-open | 12.7 deg | 8.6 % | 9.96 dB | 11.5 % | -9.3 (aligned) | 18.2 (aligned) | 6.67 / 25.3 dB |
| 3GPP InH-mixed | 23.6 deg | 2.7 % | 16.9 dB | 6.7 % | -7.7 (aligned) | 20.8 (aligned) | 14.3 / 30.3 dB |

Split by views: of the 160 views whose geometric transmitter direction is in the image, 59 have their true main
peak within 2 deg of it ("LOS-dominant"). There every site-aware predictor is near-perfect (beam-gain loss: RRF
0.06, B6 0.03, LOS 0.10, NN 0.16, InH-open 0.13 dB; RRF direction 0.49 deg); everywhere else (581 views) RRF 3.49,
B6 2.25, LOS 4.83, NN 0.21, InH-open 8.32 dB.

Level 2 (the same estimators on the ray tracer's and InH's channels, 640 views): RMS delay spread RT 3.89 ns vs
InH-open 33.5 / InH-mixed 23.9 ns (KS 0.76 / 0.71); first-tap ratio RT -15.3 dB vs -10.8 / -17.0 dB (KS 0.37 /
0.34); path gain RT -106.6 vs -106.2 / -114.8 dB (KS 0.22 / 0.40). Spectral spread (spectra): truth 17.3 deg, NN
16.5, B6 31.2, RRF 33.4, InH 47-50, LOS 1.1.

### Criteria (written before the smoke) and what happened

1. Control (a): **failed in its first version** (2.22 deg through PanelArray vs 0.19 deg through rf_spectra): the
   control drew directions outside the 90 x 67.4 deg face, whose peaks sat on the border. Fixed to in-image pixel
   directions, the same for both paths: 0.00 deg both. Recorded, the range not changed.
2. t6's coherent recomputation reproduces the stored truth to 0.01 dB: **failed** -- by up to 35 dB. Cause found:
   the generator's MVDR ran in complex64 on a covariance with cond ~3e11, and the ray tracer returns the same paths
   in a different order on every solve; two float32 solves of one view differ by up to 17 dB, two float64 ones by
   6e-5 dB. Fixed in the generator (commit 2d12e13, default complex128); the stored labels were left as they are,
   and every predictor is scored against both.
3. Geometry (truth's main peak within 2 deg of the transmitter direction in >= 60 % of the LOS-in-view views):
   **failed, 37 %** (59 / 160), above the 30 % "mapping suspect" line. The mapping itself is verified independently
   (the LOS baseline's peak lands on the geometric pixel within 0-0.37 deg; InH-open's peak is there in 89 % of
   the LOS-dominant views). The expectation was wrong about the physics: this dataset is diffuse-dominated (the
   first 0.1 ns tap carries -15.3 dB of the power, scattering coefficient 0.7), so the direct path is the main
   MVDR peak in only a third of the views that see the transmitter.
4. InH convention (InH-open's peak within 2 deg of the LOS at the LOS-dominant views >= 50 %): **passed, 88.6 %**.
5. Level-3 sanity (a beam steered to the truth's own MVDR peak loses <= 1 dB): **passed, 0.03 dB**.

Smoke criteria: renders finite and [200, 300] (yes); InH faces of one position share their cluster delays (4.6e-13
s over the live clusters -- the first diagnostic compared zero-power padding slots index by index and reported
477 ns; fixed); run time (InH 3 s per realisation of 640 views, t6 263 s for 640 solves).

Expectations: the ray tracer's delay spread below InH's (yes, 3.9 vs 24-34 ns); its first-tap ratio higher than
InH's (no: -15.3 vs -10.8 / -17.0 dB); LOS within 1 dB of the RRF at LOS-dominant views (yes, 0.10 vs 0.06 dB);
the RRF better elsewhere (yes, 3.49 vs 4.83 dB); InH worst (yes).

### What this says

- Against the traditional models the site-specific field wins everywhere it can: 2-4x smaller direction error and
  4-12 dB less beam-gain loss than 3GPP InH, which only reproduces the path gain (KS 0.22) and none of the lobby's
  angular or delay structure (its delay spread is 6-9x the lobby's).
- The baseline no earlier paper reported beats the field: the nearest training position's own label (receiver
  positions are a few tens of cm apart along the route) has a 4x smaller direction error and 14x smaller beam-gain
  loss than the SH3 field. The field is as poor on its own training views as on held-out ones (round 48,
  `diag_train_fit.py`), so this is underfitting, not generalisation.
- The stored MVDR labels' main peaks are only 67.5 % within 1 deg of their own noise-free recomputation: that is
  the ceiling of "<= 1 deg" against these labels, not 100 %.
