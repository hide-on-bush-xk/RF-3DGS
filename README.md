# RF-3DGS: radio radiance fields on gsplat

This repository started as a fork of the published RF-3DGS code (IEEE TWC 2026,
[arXiv 2411.19420](https://arxiv.org/abs/2411.19420)). The published code now sits unchanged in
[original_rf3dgs/](original_rf3dgs/), with its own README; everything else is this project's.

## Layout

| folder | what | environment |
| --- | --- | --- |
| [rrf_gsplat/](rrf_gsplat/) | the current pipeline: the radio radiance field on gsplat ([train_rrf.py](rrf_gsplat/train_rrf.py)), its losses, the LM optimiser and the shading head; evaluation ([mvdr_peaks.py](rrf_gsplat/mvdr_peaks.py) and the table scripts); `check_*.py` (controls), `diag_*.py` (one-off diagnostics); [rounds/](rrf_gsplat/rounds/) holds every experiment queue, one script per round | `rf-gsplat-win` (Windows, the gsplat fork), `rf-gsplat` (WSL, stock gsplat 1.6.0) |
| [sionna_port/](sionna_port/) | the dataset generator on Sionna 2.1 ([generate_dataset.py](sionna_port/generate_dataset.py), [rf_spectra.py](sionna_port/rf_spectra.py)), the dashboards, and the comparison with 3GPP channel models (`t4_*.py`, `t6_*.py`) | `rf-sionna-win` |
| [tx_planning/](tx_planning/) | transmitter placement on the same scene | `rf-sionna-win` |
| [scene2/](scene2/) | the second scene, a procedural corridor with rooms | `rf-sionna-win`, `rf-3dgs` |
| [docs/](docs/) | the lab notebook ([stage2_notes.md](docs/stage2_notes.md)), plans, paper drafts, [tech_paths.md](docs/tech_paths.md) | |
| [original_rf3dgs/](original_rf3dgs/) | the published code, unchanged apart from the move: INRIA 3DGS `train.py` / `render.py` / `metrics.py`, `scene/`, `utils/`, the CUDA `submodules/`, the SIBR viewer, the upstream README. Used for the "as published" rows only | `rf-3dgs` |
| `RF-3DGS_dataset/`, `output/` | data and results (not tracked) | |

The original code runs from the repository root as before, with its new path: `python original_rf3dgs/train.py -s ...`
(its imports resolve next to the script; `diff_gaussian_rasterization` and `simple_knn` are installed into `rf-3dgs`,
not editable, so the move does not affect them).

## Environments

| env | where | for |
| --- | --- | --- |
| `rf-3dgs` | Windows | the original code (INRIA rasteriser, simple-knn) |
| `rf-gsplat-win` | Windows | `rrf_gsplat/`: gsplat 1.6 built from the fork at `../gsplat_win` (branch `rf-win`: MSVC fixes, the per-Gaussian backward, the frozen-geometry backward) |
| `rf-gsplat` | WSL Ubuntu 22.04 | stock gsplat 1.6.0, the untouched reference |
| `rf-sionna-win` | Windows | Sionna 2.1 with OptiX: `sionna_port/`, `tx_planning/` |

The CUDA build recipe (VS 2026, CUDA 13.4) is in [sionna_port/README.md](sionna_port/README.md) and
[docs/ours_vs_rf3dgs.md](docs/ours_vs_rf3dgs.md).

[LICENSE](LICENSE) is INRIA's Gaussian-Splatting license, which covers the code derived from it.
