# Scene 2: a corridor with closed rooms

Procedural scene built to separate transmitter distance from lit-surface overlap (a door is binary visibility, a wall binary occlusion) and to replicate the lobby's claims with predictions written before the runs. Plan: a 28 m corridor 2.4 m wide, three 5 × 4.5 m rooms on each side with 0.9 × 2.1 m door openings, a 10 × 8 m hall.

| file | what |
| --- | --- |
| `make_corridor.py` | writes `corridor/`: the Sionna scene (`corridor_sionna.xml`, ITU materials), the visual scene (`corridor_visual.xml`, textured diffuse walls, area lights), meshes, textures, `rx_route.txt` (0.2 m step, interleaved across spaces), `tx_positions.json` (13 transmitters), `layout.json` |
| `render_visual.py` | renders the visual dataset with Mitsuba in the Blender / NeRF-synthetic layout the INRIA trainer reads |
| `eval_clean_test.py` | re-renders the held-out poses at high spp and scores the trainer's renders against clean targets |
| `make_manifest.py` | manifest with a sha256 per file (`--check` verifies a dataset against it) |
| `win_scene2_fix.sh`, `win_scene2_rf.sh`, `win_scene2_extra.sh`, `win_scene2_rerender.sh` | the queues: stage 1, the three RF experiments, the follow-ups (room hold-out, seeds), the full-test-set re-renders |
| `analyze_scene2.py` | the tables: density (overall and per space class), zone structure with lit-surface overlap, consistency |

## What is tracked and what is not

The rendered frames (`visual_dataset/`, 563 MB) and the 3DGS checkpoints (`visual_trained*/`) are derived data and are not in git. Tracked instead: the scene, the renderer, its exact arguments, and `visual_dataset_manifest.json` (910 files, sha256 each).

The render is seeded (Mitsuba `seed = frame index`, random poses from `numpy.random.default_rng(0)`), so a re-render reproduces the frames bit for bit only with the same Mitsuba version (3.x, `cuda_ad_rgb`) and the same GPU code path; with another version the frames differ by Monte-Carlo noise (64 spp, about 25 dB between two renders of one view). **The manifest verifies a copy; it does not make the render reproducible across versions.**

```bash
PYTHONUTF8=1 python scene2/render_visual.py --scene scene2/corridor --out scene2/visual_dataset --width 800 --height 450 --spp 64 --extra 300 --route-every 4
PYTHONUTF8=1 python scene2/make_manifest.py --dataset scene2/visual_dataset --out scene2/visual_dataset_manifest.json --check
```

The corridor checkpoint every scene-2 RF result depends on (`visual_trained/chkpnt30000.pth`, 582,363 Gaussians, sha256 `bac6515b3d2f2ea1…`) is backed up outside the repository disk: `D:\RF-3DGS_backup\scene2_visual_trained\` (with the point cloud and `cfg_args`) and OneDrive `RF-3DGS_backup\scene2_chkpnt30000.pth`. The lobby checkpoint (`4a1023d5b72ca54a…`) is on `D:\RF-3DGS_backup\lobby_visual_trained\`.

## Camera convention

Mitsuba's camera frame is x left, y up, z forward; a Blender pose is passed with its x and z columns negated. The first two datasets flipped y and z (a 180° rotation about the optical axis) and could not be fitted (17–18 dB, 62–74k Gaussians). Two semantic checks fix the handedness (ceiling lights in the upper half of the frame; an off-axis corridor mouth on the camera's left); a mirrored pose lowers PSNR without zeroing it, so the number alone cannot locate the error.
