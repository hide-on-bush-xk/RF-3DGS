# Where the radio radiance field loses its peaks, and technical paths beyond the current tricks

Prepared 2026-09-24 (night) for Ke's deep dive. Every number is from `docs/stage2_notes.md` / `output/rrf/*`
unless marked as an estimate. Hypotheses are marked **H**; each comes with the cheapest experiment that would
refute it.

## 0. The state in one table (MVDR 60 GHz, NIST lobby, 640 held-out views)

| model | PSNR(jet) | main-peak direction median | <= 1 deg | power at the true peak | top-3 detection |
| --- | --- | --- | --- | --- | --- |
| SH3, frozen geometry (B1, 3 seeds) | 18.72 | 5.55 deg | 19.5 % | -8.07 dB | 68.8 % |
| SH3 + shading head (B6, fixed head, 3 seeds) | 21.52 | 4.38 deg | 19.6 % | -7.06 dB | 67.6 % |
| SH3 + peak loss (B4) | 18.03 | 5.59 deg | 19.5 % | -4.96 dB | 74.4 % |
| geometry unfrozen (round 37) | 21.53 | 5.16 deg | 15 % | -6.65 dB | 45 % |
| label-level control: 150x100 bilinear of the truth | -- | 0.37 deg | 77 % | -2.39 dB | 73 % |

Every representation change so far (SH order, a CNN head, guide buffers, a latent, geometry training, 4x the
views per step) moves PSNR by up to 3 dB and leaves "<= 1 deg" at 15-20 %. The peak is lost somewhere all of them
share. Tonight's ablations (rounds 43-45) do not change that: the head's features, the per-Gaussian backward, the
exact SSIM and LM are about how fast / how smooth the fit is, not about what the model can represent.

## 0b. What the night established (rounds 47-49; details in `docs/t4_channel_model_plan.md`, `stage2_notes.md`)

Five facts that re-rank everything below:

1. **The labels are partly noise.** The dataset's MVDR ran in complex64 on a covariance with cond ~3e11; the
   solver returns its paths in a different order every time, so the weak directions are rounding noise. Against a
   float64 recomputation of the same 640 held-out views the stored labels differ by up to 35 dB (per-view median
   0.8 dB), and their main peak is within 1 deg of the noise-free one in only **67.5 %** of views -- the ceiling of
   "<= 1 deg" against the current labels. Fixed in the generator (2d12e13); regenerating the benchmark is open.
2. **A lookup beats the field.** The nearest training position's label: direction median 1.29 deg (field 5.56),
   <= 1 deg 39.7 % (19.8 %), beam-gain loss on the true channel 0.19 dB (2.62). 3GPP InH is far worse than both
   (12.7 deg, 6.7 dB); geometric LOS in between (8.2 deg, 3.5 dB).
3. **The field underfits; it does not fail to generalise.** On its own training views it is as poor as on held-out
   ones (<= 1 deg 17.8 vs 18.4 %). One position alone is fitted almost exactly (0.14 dB RMSE, top-3 100 %) by the
   frozen Gaussians; 16 positions 59 %; 160 positions 19 %; 640 positions 18 % at 2.5k steps and 21.7 % at 20k
   steps (while held-out PSNR rises 18.8 -> 20.0 dB). Capacity, not schedule: what fails is sharing one colour
   function per Gaussian across many receiver positions.
4. **Most main peaks are diffuse hotspots, not the direct path.** The first 0.1 ns tap holds -15 dB of the power
   (scattering coefficient 0.7); of the 160 views that see the transmitter only 59 have the direct path as the
   main peak -- and there the field is already right (0.49 deg). P0 (an emitter at the transmitter) is moot.
5. **P1's ceiling is not what binds.** The incoherent MVDR (what a power-rendering field plus an MVDR layer could
   reach at best) keeps the main peak within 1 deg of the coherent float64 one in 65.8 % of views (at the peak
   -0.07 dB). The field is at ~20 %; the ceiling of the incoherent representation is ~66 %.
6. **Sparse training does not rescue the field (round 49).** Trained on K positions and compared with NN from the
   same K positions, on the 640 held-out views (field / NN):

   | positions | direction median | <= 1 deg | beam-gain loss median (P90) | RMSE |
   | --- | --- | --- | --- | --- |
   | 640 | 5.68 / 1.29 deg | 18.4 / 39.7 % | 2.75 (16.6) / 0.19 (8.8) dB | 3.85 / 2.62 dB |
   | 160 | 5.43 / 1.89 deg | 18.3 / 25.2 % | 2.73 (17.2) / 0.37 (8.8) dB | 3.91 / 3.21 dB |
   | 40 | 7.21 / 3.16 deg | 15.5 / 13.0 % | 3.25 (19.6) / 1.08 (10.5) dB | 4.74 / 4.62 dB |
   | 10 | 12.64 / 7.11 deg | 11.4 / 6.2 % | 5.98 (23.9) / 3.31 (16.6) dB | 6.33 / 6.87 dB |

   By the reading written before the run (the field must win on direction median or beam-gain loss at some K), NN
   is better at every density down to 10 positions. The field wins only secondary metrics at K <= 40 (<= 1 deg
   share, RMSE at 10); its top-3 detection (~70 % at every K vs NN's 14-45 %) is inflated by its many local maxima
   (81 % of its predicted peaks are false) and is not a win.
7. **Clean labels do not rescue it (round 50).** Regenerated in float64 (3dgs_MVDR_100_f64_gpct, same poses,
   range and split; the float32 labels had per-view maxima up to 10 dB above the true one), the field trained and
   scored on clean labels is unchanged within seed range (5.58 vs 5.59 deg, <= 1 deg 20.2 vs 18.6 %). NN on the
   clean labels: 1.16 deg, 40.3 %.
8. **More SH bands help the capacity benchmark, modestly (round 51, one seed).** 160 positions, training views:
   <= 1 deg 16.6 / 18.0 / 19.2 / 23.9 % and main-peak median 5.60 / 5.75 / 4.95 / 3.74 deg for SH1-4 (held-out
   PSNR 17.6 / 18.2 / 18.8 / 19.4). Monotone, but SH3 -> SH4 is +4.7 points, between the two lines written before
   the run: no conclusion yet.
9. **Sharp lobes do not lift it either (round 52, one seed).** SH3 + spherical-Gaussian lobes (`--lobes`): 1 lobe
   23.9 %, 2 lobes 24.1 %, 1 sharp lobe (kappa 100, ~6 deg) 25.6 % <= 1 deg on the same benchmark (main peak
   3.32 deg, at the true peak -5.95 dB) -- about one more SH band's worth, below the 28.9 % written before the
   run as the line for pursuing P2. A sharper directional function per Gaussian is not what is missing.

What this implies: the bottleneck is how a Gaussian's value may vary with the receiver position. SH3 gives each
Gaussian 16 numbers as a smooth function of the direction to the receiver; one position is representable, many are
not. Whether more angular bandwidth (P2) is enough, or the value must depend on the receiver position itself (not
only the direction -- a diffuse hotspot's MVDR level depends on what else the array sees, M1), is the question the
next experiments should separate: P2 (sharper lobes) vs a position-conditioned colour (a per-Gaussian latent
decoded with the receiver position, the NeRF^2 / shading-head idea moved into the Gaussians).

## 1. What the model is asked to represent

For receiver position r, pixel direction u, the dataset's value is the MVDR pseudo-spectrum of the 10 x 10 array:

    P_MVDR(u; r) = 1 / ( a(u)^H R(r)^-1 a(u) ),   R(r) = sum_l x_l x_l^H,   x_l = sum_{p in tap l} g_p a(u_p)

(p = paths of the ray tracer, g_p their complex gains, taps of 0.1 ns as snapshots). The radiance field renders

    I(u; r) = sum_i c_i(d_i(r)) alpha_i(u) T_i(u),   c_i = SH_3 coefficients . Y(d_i)

and is trained so that I(u; r) = 10 log10 P_MVDR(u; r) (normalised).

Three mismatches follow directly from these two lines.

**M1 -- the target is not additive.** P_MVDR is a nonlinear function of the whole path set: adding a path changes
R^-1 and therefore the value in every direction (MVDR nulls interferers, sharpens peaks, and its floor depends on
the rank of R). Alpha compositing is (locally) additive in the emitters. The model has to fake an operator that
couples all directions with a sum over a few Gaussians per pixel. **H1:** a large part of the peak error comes
from representing a pseudo-spectrum instead of the physical angular power it is computed from.

**M2 -- the angular resolution of an emitter's view dependence.** c_i varies with the receiver direction
d_i = normalize(mu_i - r) only through SH of degree <= 3: band-limited to features of about pi / (L + 1) = 45 deg.
A specular reflection is sharp in exactly that variable: the reflection point on a wall lights a receiver only
near the mirror direction of the transmitter. With SH3 the wall's Gaussians emit the reflected power into a 45 deg
cone, so the rendered peak moves with the receiver in the wrong way and is spread. **H2:** the <= 1 deg share is
capped by the emitters' angular bandwidth, not by the number or placement of the Gaussians.
Evidence already in hand (rounds 41 / 43, 3 seeds each, same data and schedule): halving the band limit from SH1
(about 90 deg) to SH3 (45 deg) moved the main-peak direction median from 5.9-6.9 to 5.4-5.7 deg and the <= 1 deg
share from 17.0-19.2 % to 18.1-21.1 % -- a real but small trend. If H2 holds, going from 45 deg to the few degrees
of a specular lobe is a much larger step than SH1 -> SH3 and could still pay off; if the trend stays this flat,
bandwidth is not the limiter and M1 / M3 are.

**M3 -- what places the peak in the image.** In a render, the peak's pixel direction is where the brightest
Gaussian projects. For the direct path that is a Gaussian near the transmitter; for a single-bounce path it is the
reflection point on the surface. The visual reconstruction puts Gaussians where there is visual texture, and the
frozen geometry cannot move them to where the radio energy comes from. Unfreezing the geometry raised PSNR by 2.8 dB
and lowered top-3 detection from 68 % to 45 % (round 37): the optimiser moved Gaussians to fit pixels, not paths.

## 2. Candidate paths, ranked by expected gain per unit of work

**Ranking after the night (0b).** The order below was written before rounds 47-49; this is the order they suggest:

1. **Regenerate the MVDR labels in float64** (the generator's default since 2d12e13; ~5 min of generation per
   dataset, then re-run the baselines). Every comparison until then carries up to tens of dB of label noise in the
   weak directions and a 67.5 % ceiling on "<= 1 deg". Cheapest, and a prerequisite for judging anything else.
2. **P1 -- render power, apply the MVDR as a known layer.** Two independent hints point at the target's
   non-additivity (M1) rather than at the Gaussians: on the additive per-path encodings (MULTI, `stage2_notes.md`
   "波束选择准确率") the same kind of field **beat** copying the nearest training position -- M = 10 beam-selection
   top-1 0.68 vs 0.60 (sigma 3) and 0.55 vs 0.38 (sigma 1), decoded-azimuth P90 5.85 vs 131 deg -- while on MVDR the
   copy wins at every density (0b.6); and the incoherent-MVDR ceiling (65.8 % <= 1 deg) is three times where the
   field is.
3. **Position-conditioned colour (new, P7)**; sharper lobes (P2) were tried in round 52 and bought only ~1 SH band
   (0b.9). the capacity sweep (0b.3) is the fast
   benchmark for both -- train on 160 positions and score the training views (`diag_train_fit.py`); SH3 gives
   19 %, one position alone ~100 % top-3. P7: a small per-Gaussian latent decoded together with the receiver
   position (not only the direction) by a shared MLP, the shading head's idea moved into the Gaussians. P2: 1-2
   spherical-Gaussian lobes per Gaussian, optionally mirror-tied.
4. **Report the NN baseline everywhere.** A radio radiance field has to beat the lookup of its own training labels
   to claim anything about peaks; on this dataset it does not, at any density.
5. P0 is moot (0b.4); P3-P5 stay as the heavier physics routes.

### P0. An explicit emitter at the transmitter  (addresses M3 for the direct path; the cheapest test)

If most views' main peak is the direct path, its pixel is fixed by geometry: the transmitter's direction. The
frozen visual reconstruction has no compact Gaussian there (whatever Gaussians sit near the antenna were placed
for visual texture, and their footprints are large), so the rendered peak lands where those Gaussians project,
spread and offset. **H0:** at the views whose main peak is the direct path, the RRF's direction error comes from
the missing compact emitter. **Test:** add one Gaussian at the transmitter position (isotropic, a few cm, opacity
near 1, SH trainable, geometry frozen like the rest; occlusion by the scene's Gaussians then models blockage),
retrain the SH3 db configuration, and compare the direction error at the LOS-dominant views (t4_score.py splits
it). Depends on T4's geometry check: the share of views whose truth main peak lies within 2 deg of the geometric
transmitter direction. If that share is small, P0 cannot matter much.

### P1. Render the physics, apply the array processing as a known layer  (addresses M1)

Let the Gaussians emit angular power density (and a delay), render the per-direction power S(u; r) (additive, alpha-
composited as now), and compute the spectrum with the same processing as the truth, differentiably:

    R_hat(r) = sum_pixels S(u; r) a(u) a(u)^H dOmega(u) + sigma^2 I       (incoherent, per delay tap if delays are rendered)
    P_hat(u; r) = 1 / (a(u)^H R_hat^-1 a(u)),   loss on 10 log10 P_hat vs the truth

Gradient (standard matrix calculus): dP/dR = P^2 R^-1 a a^H R^-1, so dL/dS(u') = sum_u dL/dP(u) P(u)^2
|a(u)^H R^-1 a(u')|^2 dOmega(u'). Cost per view: one 100 x 100 inverse and a [H W, 100] projection -- cheap next to
the rasteriser. The model then only has to put the power in the right directions; the sharpening, nulling and
floor are exact.
**First experiment (no training):** take the RT path lists of 20 test positions, build R from the paths
incoherently (drop the phases: sum_p |g_p|^2 a a^H, per tap) and compare its MVDR with the truth's coherent one on
mvdr_peaks. If the incoherent MVDR keeps the main peak within 1 deg in most views, P1 is sound; if not, the
coherence of paths within one tap matters, and no rendered power (per tap or not) carries it -- see below.
Risk: the MVDR's floor depends on rank / snapshots; the incoherent R is full rank with sigma^2 -- the floor must
be calibrated (one global sigma^2).

**What the first experiment actually measures (derivation).** With x_l = sum_{p in l} g_p a_p,

    R_coh = sum_l x_l x_l^H = sum_p |g_p|^2 a_p a_p^H  +  sum_l sum_{p != q in l} g_p g_q^* a_p a_q^H
          =        R_inc                              +  (cross terms of paths sharing a 0.1 ns tap)

Three consequences:

1. Delay resolution does not help an incoherent model: per-tap incoherent covariances R_l = sum_{p in l} |g_p|^2
   a_p a_p^H sum to R_inc again. Rendering per-tap power adds nothing over rendering total power per direction;
   only the complex amplitudes (phases) of paths in one tap carry the rest.
2. Over random phases the cross terms have zero mean: R_inc = E[R_coh]. The truth is one realisation of a
   speckle pattern whose phases change over lambda / 2 = 2.5 mm at 60 GHz (a path length difference within one
   tap is < 3 cm). No radiance field that renders power can learn that pattern from positions metres apart.
3. Hence MVDR(R_inc) vs the truth, which t6_incoherent_mvdr.py measures, is the ceiling of every incoherent
   radiance field on these labels -- RF-3DGS's, ours, and P1's. If that ceiling is low, the <= 1 deg numbers of
   section 0 are partly label speckle, and the remedy is on the label side (report MVDR(E[R]) as the target, or
   average over a small receiver neighbourhood) or a coherent field (complex amplitudes, as NeRF^2 renders).

**Coverage.** R needs every arrival direction, but the four faces (90 x 67.4 deg each at 300 x 200) miss
elevations beyond +-33.7 deg -- floor and ceiling bounces near the receiver. P1 must render the top and bottom
too (a cube map) or accept that those paths only enter through the element pattern's back lobe.

**Gradient cost.** With w(u) = dL/dP(u) P(u)^2, dL/dS(u') = dOmega a(u')^H G a(u') with G = R^-1 (sum_u w(u)
a(u) a(u)^H) R^-1: O(H W M^2) per view (6e8 multiply-adds at 300 x 200 x 4 faces, M^2 = 100) -- a few ms on the
GPU; torch.linalg.solve's autograd produces exactly this without forming the H W x H W matrix.

### P2. Sharper view dependence per Gaussian  (addresses M2)

Replace or augment SH3 with a few spherical Gaussian lobes per emitter:
    c_i(d) = sum_k w_ik exp(kappa_ik (m_ik . d - 1))       (or anisotropic SG as in Spec-Gaussian, NeurIPS 2024)
Lobe width 1/sqrt(kappa) is learnable down to a few degrees; 1-2 lobes per Gaussian cost 5-10 parameters (SH3 is
16). A physically informed variant: tie the lobe direction to the mirror direction of the transmitter at the
Gaussian (m = reflect(tx - mu, n)), so only kappa and w are learned -- the specular lobe then moves with the receiver
correctly by construction. **First experiment:** SH1 + one mirror-tied lobe vs SH3, 2.5k x 4, the same metrics;
the <= 1 deg share is the number to watch.

### P3. Structure the paths (RF-PGS's "fully structured" decomposition)  (addresses M1 + M3)

Per path: FSPL(total length) x interaction gain(direction), with the geometry of the path computed from the Gaussians
and the known transmitter (image-source for one bounce). The direct path is then exact, single bounces are placed
by geometry, and only the interaction gains are learned. RF-PGS reports 20.61 vs 14.22 PSNR for RF-3DGS on this
scene (their data regime; not comparable to our 18.7 without their split). Heavier to build (a path enumerator
over Gaussians), strongest physical prior.

### P4. Supervise with the ray tracer's paths, not only its images  (addresses M3)

In simulation we have every path's AoA, delay and power. A loss that asks the rendered S(u; r) to put each path's
power at its AoA (or asks the Gaussian nearest each reflection point to carry it) gives the geometry the signal the
image loss does not. Only for simulated data -- but it would show whether M3 is the limit.

### P5. Differentiable ray tracing on the Gaussians (arXiv 2605.07781, noted in docs/literature_check.md)

Do not learn a radiance field; trace paths against the visual Gaussians with learned materials. Highest physical
fidelity, the biggest departure from the current codebase.

## 3. Mathematical facts to have at hand

- SH of degree L resolves angular features of about pi / (L + 1): 45 deg at L = 3, 18 deg at L = 9 (100 coefficients).
- A 10 x 10 half-wavelength UPA's broadside beamwidth: about 2 / M rad = 11 deg (Bartlett); MVDR resolves finer
  when R is well conditioned. The peak metrics' 1 deg bin is well inside that.
- The render is linear in the SH coefficients when geometry and opacity are frozen (used by lm_optim.py): the
  colour fit is a (weighted) linear least-squares problem in 16 N unknowns; its conditioning is set by how many
  views see each Gaussian and from how many directions -- a Gaussian seen from a narrow cone cannot fix its degree-3
  terms (the unconstrained directions extrapolate).
- MVDR is scale-covariant: P(c R) = P(R) / c, so a global level error is one offset; peak direction and relative
  powers are invariant.

## 4. What tonight's work changes for these paths

- The exact SSIM (`--cudnn-tf32 off`) and the frozen-geometry backward (`GSPLAT_BWD_NO_GEOM`) make every
  experiment above faster and its loss correct; they are infrastructure.
- The traditional-model comparison (docs/t4_channel_model_plan.md) will say where the RRF stands against a
  geometric LOS predictor and 3GPP InH on the same peak metrics -- if LOS beats the RRF on direction at LOS
  positions, P1/P3 (put the physics in) are the priority over P2.
