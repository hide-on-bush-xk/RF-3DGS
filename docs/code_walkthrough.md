# 代码导览:从起点到每个模块的输入/输出

> 写给 Ke:仓库现在有三条独立的流水线,共用一个场景和一个视觉 checkpoint。每一节按"起点 → 关键函数 → 输入/输出"写,
> 文件名可点击。环境:`rf-sionna-win`(Windows,Sionna 2.1 GPU)跑生成与规划;WSL `rf-gsplat` 跑 gsplat 训练;
> `rf-3dgs`(Windows)只跑原版 `train.py`。

## 0. 全景

```text
Blender 场景 XML ──fix_scene_xml/ascii_meshes──▶ NIST_lobby_V1.1_sionna12.xml
                                                        │
      ┌─────────────────────────────────────────────────┼──────────────────────────────────┐
      ▼                                                 ▼                                  ▼
[A] 数据集生成 sionna_port/generate_dataset.py   [C] Tx 规划 tx_planning/*.py        (原版) 教程 notebook
      │  images/*.png  spectra_float/*.npy                │  tx_sweep.npz, optimize_tx.json,
      │  sparse/0/{cameras,images}.txt                    │  fit_materials.json, active_measurement.json
      │  generation_meta.json                             │
      ▼                                                   │
[B] RRF 训练 rrf_gsplat/train_rrf.py  (或原版 train.py)   │
      │  results.json  rrf_state.pt  renders/             │
      ▼                                                   ▼
[D] 汇总与 dashboard  rrf_gsplat/summarize.py + sionna_port/dashboard.py(--rrf-dir / --planning-dir)
      └─▶ output/dashboard.html
```

视觉几何来自 `RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth`(INRIA 3DGS 训练 30k 步的 checkpoint,
1,014,142 个高斯),[B] 的所有训练都从它出发,几何默认冻结。

---

## A. 数据集生成 —— `sionna_port/generate_dataset.py`

**起点** `main()` → `argparse` → `Config(**vars(args))` → `generate(cfg)`。

| 步骤 | 函数 | 输入 | 输出 |
| --- | --- | --- | --- |
| 场景 | `build_scene(cfg)` | `cfg.scene_xml`, `frequency`, `M`, `materials` | Sionna `Scene`:`load_scene(merge_shapes=True)`,`tx_array` 1×1、`rx_array` M×M(tr38901);材料要么统一 `scattering_coefficient`,要么 `tutorial_materials.apply()` |
| 位姿 | `load_rx_locations(path, rng)` 或 `read_pose_groups(images_txt)` | NIST 路线文件(mm)+ 抖动;或别的数据集的 `images.txt` | `[(rx_loc, [yaw...])]`,默认每位置 4 个 yaw = `VIEW_YAWS` |
| 角网格 | `ArrayGrid.build(M, width, height, fov, element_gain_fn)`([rf_spectra.py](../sionna_port/rf_spectra.py)) | 相机模型 | 每个像素的 (θ, φ)、导向矢量 `steering [M², H·W]`、带单元增益的 `manifold` —— 只算一次 |
| 求解 | `solve_paths(solver, scene, cfg, view_index)` | `Receiver(position, orientation=[yaw,0,0])` 加进场景;`PathSolver()(scene, max_depth, samples_per_src, los, specular_reflection, diffuse_reflection, refraction, synthetic_array, seed=cfg.seed+view_index)` | Sionna `Paths`:`a`(复增益)、`tau`、`theta_r/phi_r`(AoA)、`theta_t/phi_t`(AoD)、`valid` |
| 谱 | `spectrum_for_paths(paths, grid, cfg, yaw)` | `Paths` | 见下表;返回 `[H,W]` 或 `[C,H,W]` 的 torch 张量(dB 或原始量) |
| 指标 | `channel_metrics(paths, bandwidth_hz)`([rf_metrics.py](../sionna_port/rf_metrics.py)) | `Paths` | K 因子、RMS 时延扩展、相干带宽、角扩展、相干增益、容量… |
| 写盘 | pass 2 | 所有谱 + 全局范围 | `images/NNNNN.png`(jet)、`spectra_float/NNNNN.npy`、`sparse/0/cameras.txt`、`images.txt`、`generation_meta.json` |

`spectrum_for_paths` 按 `cfg.spectrum` 分两族:

| 类型 | 调用链 | 数学 |
| --- | --- | --- |
| `CBF` | `paths_to_response(paths, time_interval_ns)` → `h[t, m]`(按时延分箱的逐单元复响应)→ `cbf_spectrum(h, grid)` | $P(\theta,\phi)=\sum_t \lvert \mathbf a^H \mathbf h_t\rvert^2$ |
| `MVDR` | 同上 → `mvdr_spectrum(h, grid, diagonal_loading)` | $P = 1/(\mathbf a^H \mathbf R^{-1}\mathbf a)$,$\mathbf R=\sum_t \mathbf h_t\mathbf h_t^H$;秩亏时拒绝 |
| `MULTI` | `multichannel_spectrum_equirect(paths)` → `equirect_to_perspective(eq, w, h, fov, yaw)` | 按 AoA 把每条路径高斯核散射到等距柱状网格(`equirect_splat`,一次 `index_add_`),4 通道:功率 dB、AoD 方位(圆均值)、AoD 天顶、时延 —— 都是功率加权均值,**不乘幅度** |
| `AOD3` | `aod_spectrum_equirect(paths)` → 同样重采样 | 教程的编码:R = amp·θ/180,G = amp·(1−(φ+180)/360),B = amp,再 `10log10(+150)` |

位姿约定(和教程一致,`euler_to_quaternion([yaw,0,0])`):`images.txt` 存的是 c2w 四元数和 `t = −R_c2w·rx`,
所以反算 `rx = −Rᵀt`(`read_pose_groups` 就是这么做的)。

**命令**(Windows,`rf-sionna-win`,`PYTHONUTF8=1`):

```bash
python generate_dataset.py --scene-xml ../sionna_tutorial/.../NIST_lobby_V1.1_sionna12.xml \
    --rx-loc-file ../sionna_tutorial/.../NIST_rx_loc.txt --out-dir ../RF-3DGS_dataset/regenerated/3dgs_MVDR_100 \
    --spectrum MVDR --num-positions 800 [--frequency 2.4e9] [--materials tutorial] [--tx x y z] [--poses-from images.txt]
```

配套:

- [renormalize.py](../rrf_gsplat/renormalize.py):`spectra_float` → 新的 `images/`(`--norm global-pct|global-minmax|per-view`,`--range`),
  浮点用 NTFS 硬链接;逐图归一化时写 `per_view_ranges.csv`。这是 E1(归一化)实验的数据来源。
- [tutorial_materials.py](../sionna_port/tutorial_materials.py):`apply(scene, frequency_hz, variant="asis"|"fixed")`,按材料名把
  每个物体换成 notebook cell 6 的定义(as-is 保留其 1e-9 的单位错误)。
- [fix_scene_xml.py](../sionna_port/fix_scene_xml.py) / [ascii_meshes.py](../sionna_port/ascii_meshes.py):一次性把 Blender 导出的
  XML 变成 Sionna 2 能读的(材料名、ITU 有效性、网格文件名 ASCII、shape id)。

---

## B. RRF 训练 —— `rrf_gsplat/train_rrf.py`

**起点** `main()`。一次运行 = 一个数据集目录 + 一种 `--mode` → `output/rrf/<name>/{results.json, rrf_state.pt, renders/, train.log}`。
外层用 [wsl_run.sh](../rrf_gsplat/wsl_run.sh)(单次)或 [run_matrix.sh](../rrf_gsplat/run_matrix.sh)(按组、跳过已完成)在 WSL 里跑。

| 步骤 | 函数 | 输入 | 输出 |
| --- | --- | --- | --- |
| 色标范围 | 读 `generation_meta.json` | `spec_min_db/spec_max_db`;`multi` 模式读 `channels` + `channel_ranges` | `vmin, span`(归一化值 ↔ dB 的映射) |
| 相机 | `read_colmap_text(sparse/0)` | `cameras.txt`(PINHOLE fx fy cx cy)、`images.txt` | `{name: (viewmat 4×4 w2c, K 3×3, w, h)}` |
| 划分 | `ensure_split(source)` | `train_index.txt/test_index.txt`(没有就按**位置**留出 20% 写出) | 名字列表 |
| 数据 | `load_views(source, names, views, device, want_float)` | PNG、`spectra_float/*.npy` | `{"rgb": uint8 [n,3,H,W], "float": fp16 [n,(C,)H,W], "viewmats", "Ks", ...}` 全部驻留 GPU |
| 模型 | `RRF(ckpt, mode, channels, sh_degree, device, train_opacity, train_geometry)` | checkpoint 的 `xyz/scaling/rotation/opacity`(detach!) | `params` = ParameterDict:`means, scales(log), quats, opacities(logit,[N]), sh0 [N,1,C], shN [N,K−1,C]`;SH 清零 |
| 前向 | `model.render(viewmat, K, w, h, span_db)` | 一张视图 | `[C,H,W]`:rgb 走 gsplat 内置 CUDA SH(`sh_degree=3`);db/power/multi 走 `colours()`(`sh_basis` 线性映射)→ `rasterization(..., sh_degree=None)`;power 模式先 `10^(v·span/10)` 再合成再 `10log10` |
| 损失 | `l1_loss` + `ssim`(INRIA 的 `utils/loss_utils`),λ=0.2;`multi_loss` 对非掩码通道只在功率 > 2% 的像素上算 L1 | 预测 vs `target(i)` | 标量 |
| 优化 | 每个参数一个 `Adam(eps=1e-15)`:sh0 0.0025、shN /20、opacities 0.05;几何(若解冻)means 1.6e-5×scale、scales 5e-3、quats 1e-3 | | |
| 致密化 | `--densify mcmc`:`MCMCStrategy(cap_max, refine_start 500, refine_stop 0.8·iters, refine_every 100)`,`step_pre/post_backward` 钩子 + gsplat 的 opacity/scale 正则 | `model.params`、`optimizers`、`model.last_info` | 参数张量被就地替换(所以模型全部通过 `params[...]` 读) |
| 评估 | `evaluate(model, data, idx, span, vmin)` / `evaluate_multi(...)` | 留出视图 | `psnr_rgb`(预测经 `jet_rgb` 后对 PNG)、`ssim_rgb`、`rmse_db`/`mae_db`(对浮点真值)、`rmse_db_in_range`(真值裁到色标范围)、multi 的逐通道 RMSE(方位角 wrap-aware,单位 deg/ns) |

`--mode` 四种:`rgb`(= RF-3DGS,jet 三通道)、`db`(单通道归一化 dB)、`power`(线性功率合成)、`multi`(每个浮点通道一个)。
`--init-from` 热启动读别的 run 的 `rrf_state.pt`(`load_state` 兼容旧布局)。

**和原版的对应**:[train.py](../train.py) 的 `training()` 用 `Scene(dataset)` + `GaussianModel.restore(checkpoint)`,冻结
`_xyz/_scaling/_rotation`、清零 `_features_dc/_features_rest`,每步 `render(viewpoint_cam, gaussians, pipe, bg)`
(`gaussian_renderer/__init__.py` → `diff_gaussian_rasterization`),同样的 L1+SSIM。`train_rrf.py` 的 rgb 模式在发布数据上
复现到 15.97 vs 16.02 dB,所以后面所有实验只改被测变量。

其它工具:

- [jet.py](../rrf_gsplat/jet.py):`jet_rgb(v)`、`jet_inverse(rgb)`(4096 点最近邻 LUT)——让 RGB 模型也能用 dB 打分。
- [summarize.py](../rrf_gsplat/summarize.py):所有 `results.json` → `summary.json` + markdown 表(含到 PSNR 阈值的步数/秒数)。
- [eval_encoding.py](../rrf_gsplat/eval_encoding.py):把 MULTI 和 AOD3 两种表示的渲染都解码成 AoD 角度,在同一批留出视图上比 RMSE。
- [bench_resolution.py](../rrf_gsplat/bench_resolution.py) / [profile_step.py](../rrf_gsplat/profile_step.py):光栅器一步的耗时(gsplat vs INRIA,按分辨率)。
- [time_tutorial_019.py](../rrf_gsplat/time_tutorial_019.py):exec 教程 notebook 自己的 cell,在 0.19 上计时。
- `win_*.sh` / `run_*.sh`:实验队列(Windows 侧生成 → WSL 侧训练),每个都是"等 GPU → 生成 → 重映射 → 复制划分 → 训练"。

---

## C. Tx 规划 —— `tx_planning/`

共用 [scene_common.py](../tx_planning/scene_common.py):`load_radio_scene(xml, frequency, scattering, rx_elements)`、
`make_solver(reverse_mode)`(反向模式要 `loop_mode="evaluated"`;`enable_reverse_mode()` 关掉 drjit 的 SpillToSharedMemory)、
`rx_grid(x_range, y_range, z, step)`、`indoor_mask(scene, points)`(向上打到天花板且向下打到地板)、`clearance(scene, p)`。

| 脚本 | 起点 → 关键函数 | 输入 | 输出 |
| --- | --- | --- | --- |
| [tx_sweep.py](../tx_planning/tx_sweep.py) | 候选 Tx(`--tx` 或 `--tx-grid-step`)× 室内 Rx 网格;`solve_tx()` 把**所有 Rx 放进一次 solve**,从 `paths.cir()` / `theta_*` / `phi_*` / `valid` 拆出每个 Rx 的路径 | 场景、候选 | `tx_sweep.npz`:`gain_db [n_tx,n_rx]`、`n_paths`、每对 ragged CIR(`cir_a/tau/aod/aoa`) |
| [optimize_tx.py](../tx_planning/optimize_tx.py) | `objective()`:一次 solve → 每 Rx 功率 → soft coverage $\frac1N\sum\sigma((P_i-T)/w)$;`dr.enable_grad(tx.position)`、`dr.backward`;Adam 上升;每步 `indoor_mask`+`clearance`≥margin,被挡先去掉墙法向分量 | `--init-from tx_sweep.npz` | `optimize_tx.json`(每步位置/目标/覆盖率/梯度) |
| [fit_materials.py](../tx_planning/fit_materials.py) | `MaterialTwin`:`measure(tx, values)`(时延分箱 PDP + 噪声)、`loss_and_grad(values, tx, obs)`(PDP 的 MSE,对每种材料的 `scattering_coefficient` 反传)、`fit(observations)`(ngd/adam/adam-rel)、`sensitivity(values, tx)`(Hutchinson)、`identifiable(history)` | 合成真值 | `fit_materials.json`(truth/fitted/identifiable/history) |
| [active_measurement.py](../tx_planning/active_measurement.py) | 候选 C 的灵敏度得分 $\sum_n \log(1+\lVert J_n(C)\rVert^2/(\lVert J_n(A)\rVert^2+\epsilon))$ → 用 A+C 拟合 vs A+中位候选,在多个留出 Tx 打分 | `MaterialTwin` | `active_measurement.json` |
| [planning_panels.py](../tx_planning/planning_panels.py) | `collect(dir)` → `render(found)` | 上面四个文件 | dashboard 的 "Transmitter-side planning" 一节(热力图 + 曲线 + 表) |

---

## D. 评价与 dashboard —— `sionna_port/`

- [rf_metrics.py](../sionna_port/rf_metrics.py):`channel_metrics(paths, bandwidth_hz)` → `ChannelMetrics.as_dict()`;`frequency_response(paths, fc, bw)`(`paths.cfr()`);`spectrum_stats(all_db)`(8-bit 量化代价)。
- [run_ablation.py](../sionna_port/run_ablation.py):s × depth 扫描 → `output/ablation.json`(`runs[*].metrics_mean`)。
- [comparison_grid.py](../sionna_port/comparison_grid.py) → `comparison.json`;[comparison_table.py](../sionna_port/comparison_table.py) 渲染"行 = 真值 / RF-3DGS 输出 / 我们,列 = optical + 六种谱"的交叉表;[render_optical.py](../sionna_port/render_optical.py) 用 Sionna 的相机在同一位姿渲染场景。
- [dashboard.py](../sionna_port/dashboard.py):`build(data)` 把 `reference_panels`、`comparison_table`、`rrf_panels`、`planning_panels` 四节拼成一页;CLI:

```bash
python sionna_port/dashboard.py output/ablation.json --out output/dashboard.html --reference-root . \
       --comparison output/comparison.json --planning-dir output/tx_planning --rrf-dir output/rrf
```

---

## E. 一次实验的完整调用链(以 `e2_mvdr_db` 为例)

1. Windows:`generate_dataset.py --spectrum MVDR --num-positions 800` → `regenerated/3dgs_MVDR_100/`(60 GHz,统一 s=0.7,每视图换种子)。
2. Windows:`renormalize.py 3dgs_MVDR_100 3dgs_MVDR_100_gpct --norm global-pct --pct 1 99.99` → 新 PNG + 硬链接的浮点 + meta 里的范围 −175.75…−101.52。
3. WSL:`wsl_run.sh e2_mvdr_db --source .../3dgs_MVDR_100_gpct --mode db` → `train_rrf.py`:读 meta 范围 → `read_colmap_text` → `ensure_split`(按位置留出 160 个)→ `load_views` → `RRF(... mode="db", channels=1)` → 10k 步(每步随机一张训练图,`render` → L1+SSIM → 三个 Adam)→ 每 1000 步在 64 张留出图上 `evaluate` → 结束时 640 张全评 → `results.json`。
4. Windows:`summarize.py` → `summary.json`;`dashboard.py --rrf-dir output/rrf` → `rrf_panels.render()` 画柱状图/曲线/表。

Tx 移动的那条:`generate_dataset.py --tx 8.2 -5.4 2.0 --num-positions 160` → `renormalize.py --range -175.75 -101.52`(用 Tx-A 的范围)→
`train_rrf.py --init-from output/rrf/e2_mvdr_db/rrf_state.pt`(热启动)。

---

## F. 想动手改,从哪里切入

| 想改的东西 | 改哪里 |
| --- | --- |
| 新的谱/新的通道 | `rf_spectra.py` 加一个 `xxx_spectrum_equirect` 或基于 `paths_to_response` 的函数;`generate_dataset.spectrum_for_paths` 加一个分支;`channel_ranges` 定范围 |
| 高斯携带的量 / 合成规则 | `train_rrf.py` 的 `RRF.render()`(`--mode` 分支)和 `target()` |
| 损失 | `train_rrf.py` 主循环里 `loss = ...` / `multi_loss` |
| 相机模型(fisheye) | 生成端:`compute_angle_matrices`/`ArrayGrid.build`(角网格)与 `equirect_to_perspective`;训练端:`rasterization(camera_model="fisheye", with_ut=True)` |
| 几何是否可学、致密化 | `--train-geometry` / `--densify mcmc`,`lrs` 字典 |
| 规划目标(SINR、吞吐…) | `optimize_tx.py` 的 `objective()`;梯度不用改 |
| 材料模型 | `tutorial_materials.TUTORIAL` 表或 `scene_common.load_radio_scene` |
