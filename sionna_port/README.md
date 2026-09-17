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

## What is verified and what is not

`test_rf_spectra.py` passes on Windows with torch 2.9.1: the angle grid matches
the pinhole model and the 90° FoV, steering vectors have unit modulus, delay
binning sums paths into the right bins, and both beamformers peak at the true
direction of a synthetic single-path arrival.

`generate_dataset.py` is **not executed yet** — it needs Sionna 2.1, which needs
Mitsuba 3 and a Linux GPU environment. It compiles, and the API calls follow the
2.1 documentation, but two assumptions want checking on the first real run:

1. that Sionna's `PlanarArray(num_rows=M, num_cols=M)` orders its elements the
   same way `_element_offsets` does — if the CBF peak lands in the wrong place,
   this is the first thing to look at;
2. that `v_tr38901_pattern` returns `(c_theta, c_phi)` in that order, matching
   0.19's `tr38901_pattern`.

Also unresolved: 0.19's `scat_keep_prob=0.1` thinned diffuse paths by a factor of
ten, and there is no 2.x equivalent. Expect a different path count, and therefore
a different absolute dB range, than the released dataset.
