# Sionna 2.x port of the RF-3DGS dataset generator

The project's simulation tutorial (`RF-3DGS-tutorial.ipynb`, distributed through
the authors' Google Drive) targets **Sionna 0.19 with TensorFlow**. Sionna RT was
rewritten on Dr.Jit and Mitsuba 3 for 2.x, so none of the path-tracing calls
survive unchanged. This directory is that pipeline expressed against **Sionna
2.1 with PyTorch**.

```
rf_spectra.py        array processing (pure torch, no Sionna import)
generate_dataset.py  scene, path solving, spectra, COLMAP + PNG + float export
test_rf_spectra.py   checks for the array processing
```

## Running it

```bash
python generate_dataset.py \
    --scene-xml   ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_lobby_v1.0/NIST_lobby_V1.1.xml \
    --rx-loc-file ../sionna_tutorial/RF-3DGS_Sionna_simulation_tutorial/NIST_rx_loc.txt \
    --out-dir     ../RF-3DGS_dataset/regenerated/3dgs_MVDR_100 \
    --spectrum MVDR --num-positions 800
```

The output layout matches what `scene/dataset_readers.py` expects, except that
`train_index.txt` / `test_index.txt` still have to be written separately.

## API mapping

| Sionna 0.19 (TensorFlow) | Sionna 2.1 (PyTorch / Dr.Jit) |
| --- | --- |
| `scene.compute_paths(...)` | `PathSolver()(scene=scene, ...)` |
| `reflection=True` | `specular_reflection=True` |
| `scattering=True` | `diffuse_reflection=True` |
| `num_samples=1e6` | `samples_per_src=1_000_000` |
| `scat_keep_prob=0.1` | **no equivalent** — the rewritten solver samples diffuse paths itself |
| `scat_random_phases=False` | **no equivalent** |
| `paths.normalize_delays = False` then `paths.cir()` | `paths.cir(normalize_delays=False, out_type="torch")` |
| `sionna.rt.antenna.tr38901_pattern` | `sionna.rt.antenna_pattern.v_tr38901_pattern` (+ `PolarizedAntennaPattern` for slant/polarisation) |
| `sionna.channel.cir_to_time_channel` | not used; delay binning is done directly in `merge_paths_to_time_grid` |
| `tf.Variable([yaw, 0, 0])` for orientation | plain list, `Receiver(..., orientation=[yaw, 0, 0])` |

`paths.cir(out_type=...)` accepts `"numpy"`, `"drjit"`, `"torch"`, `"tf"` and
`"jax"`, which is why the rest of the pipeline can stay in torch end to end.

## Deliberate departures from the tutorial

**One normalisation for every spectrum type.** The tutorial normalises
inconsistently: `generate_3dgs_MVDR/AoD/Delay/MPC_spectrum` establish a global dB
range from a first pass over two hand-picked probe positions, while
`generate_3dgs_CBF/TCBF_spectrum` call `jet_colormap_convert`, which rescales
*each image* by its own min and max — their `per_view_normalization` argument is
accepted and then never read. Per-view normalisation destroys photometric
consistency across views, which is the assumption every radiance-field method
rests on. Here a first pass collects all spectra and the range comes from the
data, identically for every type.

This matters for interpreting the paper: RF-3DGS reports CBF and TCBF collapsing
to about 5 dB PSNR and attributes it to interference and missing geometric
structure. Those are also the only two spectra generated with per-view
normalisation. The confound is testable — regenerate CBF with a global range and
retrain.

**The float spectrum is kept.** The tutorial's only output is an 8-bit jet PNG,
so the dB values are gone by the time training starts. `--save-float` (on by
default) writes `spectra_float/NNNNN.npy` next to each PNG, and
`generation_meta.json` records the dB range used for the colormap, so the
quantisation becomes reversible and a future float-domain loss has something to
train on.

**Per-view work is hoisted.** Steering vectors and the array manifold depend only
on the camera model, not the receiver pose, but the tutorial rebuilds them inside
`CBF_spectrum` / `MVDR_spectrum` for all 3200 views. `ArrayGrid.build` computes
them once.

**The singular-covariance case is checked.** MVDR needs at least M² independent
delay taps. The tutorial says so in a comment; `torch.linalg.inv` does not raise
on a rank-deficient matrix, it returns inf/nan, so `mvdr_spectrum` refuses the
case outright and offers `diagonal_loading` instead.

## Where it runs

| | Sionna | backend | scene load | status |
| --- | --- | --- | --- | --- |
| **Windows, conda py3.12** | 2.1.0 | CUDA / OptiX | 0.3 s | **works, use this** |
| WSL2, conda py3.12 | 2.1.0 | LLVM (CPU) | - | broken, see below |
| WSL2, conda py3.10 | 1.2.2 | LLVM (CPU) | 2.0 s | works, slower |

**Sionna 2.1 brought no API changes this port cares about.** `PathSolver.__call__`,
`Paths.cir`, `PlanarArray.__init__`, `Receiver.__init__` and `v_tr38901_pattern`
all have byte-identical signatures in 1.2.2 and 2.1.0, so nothing needed
rewriting. 2.1 additionally exposes `Paths.cfr()` and `Paths.taps()`, which is
where a wideband target would come from. Note 2.x requires Python >= 3.11; on a
3.10 environment pip silently installs 1.2.2 instead.

**Sionna 2.1 does not run on the CPU backend here.** Dr.Jit 1.5.0 bundles LLVM
15.0.7 and emits the `fmaximum` intrinsic, which LLVM 15 cannot lower on x86
(that landed in LLVM 17/18), so a path solve dies with `LLVM ERROR: Cannot
select: f32 = fmaximum`. Sionna 1.2.2 ships Dr.Jit 1.3.1, which does not emit it.
This is upstream, not a misconfiguration.

**OptiX is unavailable under WSL2** with driver 616.92:
`/usr/lib/wsl/lib/libnvoptix.so.1` is a 14 KB loader stub and the full library is
absent from the Windows driver store, so Dr.Jit reports "could not find symbol
optixQueryFunctionTable". Windows has the real `nvoptix.dll`, which is why the
GPU path works there and not in WSL. Between the two constraints, **Sionna runs
on Windows and gsplat runs in WSL** -- gsplat cannot build under VS 2026, and
Sionna cannot ray trace under WSL.

Two Windows-only requirements:

```powershell
$env:PYTHONUTF8 = 1      # Sionna opens the scene XML with the locale encoding
python ascii_meshes.py <scene>.xml     # once, see below
```

## Preparing the scene

The published Blender scene does not load in Sionna 1.2+ for three separate
reasons. `fix_scene_xml.py` handles the first two:

```bash
python fix_scene_xml.py NIST_lobby_V1.1.xml NIST_lobby_V1.1_sionna12.xml --frequency-ghz 60
```

1. **Blender's duplicate suffixes.** Materials are named `mat-itu_plasterboard.001`,
   and Sionna derives the ITU type by stripping `mat-` and `itu_`, so it looks up
   `plasterboard.001` and fails.
2. **Materials that are not ITU materials at all.** `custom_plastic`,
   `custom_leather` and `custom_cloth` were assigned electromagnetic properties
   by a semantic descriptor that was never published. They are mapped to ITU
   materials of roughly the right permittivity -- placeholders, not the authors'
   values. The script also rejects a mapping whose ITU properties are undefined
   at the scene frequency, which is easy to hit: `plywood` and `brick` stop at
   40 GHz, so neither can be used at 60.
3. **Mesh filenames.** The archive stores UTF-8 names without setting the UTF-8
   flag, so both bsdtar and Python's zipfile decode them as CP437 and write
   mojibake to disk while the XML still references the real names. Re-extract
   with `name.encode("cp437").decode("utf-8")`.

## What is verified

`test_rf_spectra.py` passes under both Python 3.10 and 3.12: the angle grid
matches the pinhole model and the 90 degree FoV, steering vectors have unit
modulus, delay binning sums paths into the right bins, and both beamformers peak
at the true direction of a synthetic single-path arrival.

`smoke_sionna.py` passes against the real NIST lobby scene on Windows with
Sionna 2.1.0 and GPU ray tracing, and on WSL with 1.2.2 on the CPU. It settles
what the port rested on:

* **Element ordering agrees with Sionna**, to within the array's own ambiguity
  (below). Sionna reports the line-of-sight arrival at theta 86.26, phi 27.12
  degrees; the beamformer peaks at theta 86.00 and at one of phi 27 / 153.
* **`v_tr38901_pattern` returns a single `Complex2f`**, not 0.19's
  `(c_theta, c_phi)` pair. Unpacking it as a pair silently takes its real and
  imaginary parts. Fixed in `element_gain_fn`.

### The array cannot tell phi from 180 - phi

`_element_offsets` places the elements in the y-z plane, so the steering vector
depends only on `(sin(theta) sin(phi), cos(theta))`. For the two directions above
the steering vectors differ by 9.7e-07 and their normalised inner product is
1.000000: they are the same vector. A planar array has no way to separate front
from back, so the full-sphere spectrum has two equal peaks and `argmax` picks one
arbitrarily -- which is why the same code reported the direct peak on one run and
the mirror on the next. `smoke_sionna.py` now accepts either and checks the
ambiguity explicitly.

This is worth knowing beyond the test. Each pinhole view spans only +-45 degrees
of azimuth, so a path's mirror image lands in a *neighbouring* view rather than
the same image -- the dataset's four yaw angles tile the azimuth circle, so every
arrival can appear as a ghost 180 degrees away. What suppresses it is the element
pattern's front-to-back ratio, and the two spectra do not treat that the same
way: `cbf_spectrum` beamforms with the bare `steering` vectors while
`mvdr_spectrum` uses `manifold`, which carries the TR 38.901 element gain. That
asymmetry is inherited from the tutorial, where `CBF_spectrum` calls
`steering_vector` and `MVDR_spectrum` calls `array_manifold_vector`.

`generate_dataset.py` still has not been run end to end.

## Calibrating the path count

The paper reports "more than 300,000 MPCs per Tx-Rx pair" at 60 GHz. A depth-1
solve straight out of the box gives **8**, and a spectrum built from single-digit
path counts is not a spectrum at all -- it is the array's point spread function,
a regular sidelobe lattice with no scene structure in it.

`samples_per_src` turns out to be irrelevant: 100k and 10M give the same 8 paths.
`calibrate_paths.py` isolates the actual cause -- **ITU materials load with
`scattering_coefficient = 0`**, so `diffuse_reflection=True` produces nothing and
every path is specular. Turning `diffuse_reflection` off changes the count by
zero.

| scattering_coefficient | depth 1 | depth 2 | depth 3 |
| --- | --- | --- | --- |
| 0.0 (Sionna default) | 8 | 20 | 43 |
| 0.3 | 57,619 | 94,220 | 130,269 |
| 0.5 | 159,684 | 262,935 | 363,155 |
| **0.7** | **312,683** | 522,365 | 722,643 |

0.7 at depth 1 reproduces the paper's figure closely, so that is the default in
`Config`. The authors' 0.19 pipeline must have set this somewhere -- most likely
in the semantic material descriptor that was never published, which is the same
missing piece that leaves `custom_*` materials unmapped.

## Throughput

On a desktop RTX 3060 under Windows with GPU ray tracing, 20 receiver positions
(80 views, MVDR, 312k paths each) take **8 seconds** -- about 10 views per second,
so the full 800-position dataset is roughly **5 minutes**. The 80 spectra are
distinct (mean pairwise correlation 0.298, no duplicates).

Regenerated MVDR spectra are qualitatively what the released ones look like:
smooth angular power maps with localised hot spots, rather than the array
lattice. They are not numerically comparable yet -- the normalisation range
differs (this port takes it from the data, the tutorial from two probe
positions), and the released spectra have sharper peaks, so some combination of
`time_interval_ns`, the element pattern and `synthetic_array` still needs
matching before a like-for-like evaluation.

## All six spectra

`rf_spectra.py` now covers the whole released set, in two families:

* **CBF, TCBF, MVDR** beamform on the pinhole grid. TCBF is CBF with a Hann
  taper over the array -- lower sidelobes, wider main lobe.
* **MPC, Delay, AoD** do not beamform at all. Each path is splatted onto an
  equirectangular grid at its angle of arrival with a Gaussian kernel, and the
  result is resampled through the same pinhole model the beamformed spectra use,
  so every panel in a row shares one camera. Delay colours the splat by
  normalised delay, AoD by the departure angles.

The tutorial loops over every path in Python to do that splatting, which is
minutes for 300k paths; this scatters them in one `index_add_`.

Against the released ground truth the projection spectra line up structurally
almost pixel for pixel -- the silhouettes of the pillar and the wall match --
which is the check that the splatting, the resampling and the pose recovery all
agree. Brightness differs because the normalisation does: this port takes its
range from the data, the tutorial from two probe positions.

## Cheap views with a moving lattice

`per_view_seed` (on by default) advances the solver seed per view. The reasoning
is that a low `samples_per_src` leaves sampling structure in each spectrum, and
reusing one seed makes that structure identical in every view -- the one kind of
error a multi-view fit cannot average away, because it looks like a consistent
feature of the field.

`test_sampling_accumulation.py` measures it against a 4M-sample reference at the
same pose, with 50k samples per view:

| views averaged | lattice moves | lattice fixed |
| --- | --- | --- |
| 1 | 1.690 dB | 1.691 dB |
| 4 | 1.577 dB | 1.691 dB |
| 8 | **1.472 dB** | 1.691 dB |

The fixed-lattice column is flat, which is the mechanism confirmed: averaging
eight identical residuals changes nothing. Moving the lattice does accumulate,
but only by 1.15x over eight views where independent noise would give sqrt(8) =
2.83x. So most of the 1.69 dB is bias, not variance -- a low-sample solve
systematically misses weak paths, and moving the lattice does not fix that. The
technique buys back the variance half; the bias half still costs samples.
