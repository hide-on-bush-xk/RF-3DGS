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

## Running it in WSL2

Two environment facts, both discovered the hard way:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib
python smoke_sionna.py --scene-xml <scene>_sionna12.xml     # defaults to the CPU variant
```

**Sionna 2.x needs Python >= 3.11.** On a 3.10 environment pip silently installs
1.2.2 instead. That turned out not to matter: every API this port uses --
`PathSolver`, `paths.cir(out_type="torch")`, `Paths.vertices`,
`v_tr38901_pattern` -- has the same signature in 1.2.2 as in the 2.1 docs, so the
port runs unchanged. Moving to 2.x means a new interpreter and a gsplat rebuild,
and buys nothing until Sionna PHY is needed.

**OptiX is unavailable under WSL2 with driver 616.92**, so ray tracing runs on
the CPU. `/usr/lib/wsl/lib/libnvoptix.so.1` is a 14 KB loader stub and the full
library is nowhere in the Windows driver store, so Dr.Jit fails with "could not
find symbol optixQueryFunctionTable". `sionna.rt` picks `cuda_ad_mono_polarized`
whenever CUDA is present and then dies, so the scripts set
`llvm_ad_mono_polarized` before importing it. Beamforming still runs on the GPU
through torch; only the path solve is on the CPU.

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

`test_rf_spectra.py` passes on Windows with torch 2.9.1: the angle grid matches
the pinhole model and the 90 degree FoV, steering vectors have unit modulus,
delay binning sums paths into the right bins, and both beamformers peak at the
true direction of a synthetic single-path arrival.

`smoke_sionna.py` passes in WSL against the real NIST lobby scene, which settles
the two assumptions this file used to list as open:

* **Element ordering agrees with Sionna.** With a single line-of-sight path,
  Sionna reports an arrival at theta 86.26, phi 27.12 degrees and the beamformer
  peaks at theta 86.00, phi 27.00 -- inside the one-degree grid spacing. So
  `_element_offsets` matches how `PlanarArray` lays out its elements.
* **`v_tr38901_pattern` returns a single `Complex2f`, not a `(c_theta, c_phi)`
  pair.** 0.19 returned the pair and the tutorial kept `c_theta`; unpacking the
  1.2 return value the same way silently yields its real and imaginary parts.
  Fixed in `element_gain_fn`.

`generate_dataset.py` still has not been run end to end.

## Known discrepancy with the released dataset

A depth-1 solve with diffuse reflection and 100k samples per source produced
**8 paths**, against the "more than 300,000 MPCs per Tx-Rx pair" the paper
reports for the same scene. 0.19's `scat_keep_prob` and the rewritten solver's
diffuse sampling are not the same mechanism, and `samples_per_src` will have to
be raised considerably to get a comparable path count. Until that is calibrated,
spectra regenerated here are not comparable to the released ones in absolute
terms.
