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

---

## 更新(2026-09-20):09-18 之后新增的模块与约定

全景图多了一条流水线和一层评估工具;冻结点 `d359dc9`(之后只改措辞)。

```text
[E] 场景 2  scene2/make_corridor.py ──▶ corridor/{corridor_sionna.xml, corridor_visual.xml, meshes, textures, rx_route.txt, tx_positions.json, layout.json}
                 │                                      │
                 ▼ scene2/render_visual.py (Mitsuba)    ▼ [A] generate_dataset.py --scene-xml corridor_sionna.xml --rx-loc-file rx_route.txt
      visual_dataset/ (Blender 布局, 不入库, manifest 入库)          s2_MULTI_corrM / s2_MVDR_tx<name>(+ _gpct)
                 ▼ 原版 train.py --eval 30k                                   │
      visual_trained/chkpnt30000.pth (582,363 高斯; 盘外备份)  ──────────────▶ [B] train_rrf.py --checkpoint … (队列 scene2/win_scene2_*.sh)
                                                                              ▼
[F] 评估层  eval_baselines.py(常数 / 最近邻 / 两近邻 / 场;top-k 波束;--layout 按空间;渲染不全拒绝打分)
            eval_encoding.py(编码对比,--support / --power-weighted)  delay_signed.py  delay_oracle.py
            depth_gs.py(WSL,期望/累积深度)+ depth_gt.py(Sionna mesh 首次命中)  diag_delay_paths.py  diag_overlap.py  diag_delta_pca.py  diag_neff.py
            transfer_curve.py(--tag 2k,自助法)  make_region_split.py  scene2/analyze_scene2.py(密度按 regime、区域结构 + 点亮表面 IoU、一致性)
            rrf_panels.py 新卡片:双场景密度交叉、场景 2 转移曲线
```

### A′ 生成端的变化(`sionna_port/generate_dataset.py`)

| 项 | 现在 | 为什么 |
| --- | --- | --- |
| MULTI 通道 | **5 通道**:功率 dB、cos(AoD 方位)、sin(AoD 方位)、AoD 天顶、时延;`channel_ranges(all_db, kind, power_floor_db)` 只在 hit 像素(功率 > 核底)上取百分位 | 单通道方位在 ±180° 有接缝,自身中位误差 3.2× |
| 采样 seed | **按位置**(四个 yaw 共用一次 solve 的路径集合);`--seed-per-view` 恢复教程的按视角 seed | 按视角 seed 让四个面看到四套漫散射路径:路径级目标功率 RMSE +48%,波束谱不敏感 |
| 核截断 | `power_floor_db = −150`,核尾巴低于它的像素记为无路径 | 核尾巴污染 spec_min 与 hit 掩码 |
| 零路径位姿 | 波束谱分支写常数 `EMPTY_VIEW_DB = −300`,`generation_meta.empty_views` 计数 | 场景 2 房间 Tx 在 depth 1 下 46–48% 的位姿无路径,之前进程直接崩 |
| `--tx-list NAME:x,y,z` | 一个进程内共享场景 / 求解器 / 角网格换 Tx | 省每进程 16 s 的启动,吞吐不变(干净卡上 A/B:23.2 vs 22.4 views/s) |
| 位置抖动 | 教程的 ±0.5 m(x, y)、−1.0 … +0.3 m(z)保留 | 所以"间距轴"一律是实测的最近训练距离,不是路线的名义步长 |

**MULTI 是什么、不是什么**:每像素是落入该像素的路径的**一阶矩**(功率和、功率加权圆均值方位、均值天顶、均值时延),两条等功率路径给同一标签;
不是多峰 AoD、PDP、相干叠加或相位,不是 CSI/CIR。功率算子 `_path_arrays` 是 (Σ_m |a_m|)²,只在各阵元幅度相同(合成阵列、同元方向图)时等于单阵元功率乘常数。

### B′ 训练端的变化(`rrf_gsplat/train_rrf.py`)

| 标志 | 作用 | 用在哪 |
| --- | --- | --- |
| `--delay-depth` | 时延通道 = 学习残差 + 渲染深度 / c(`render_mode="RGB+D"/"RGB+ED"`,在通道归一化单位上相加) | 时延中位 2.42 → 0.95(D)→ 0.88(ED)ns |
| `--delay-depth-mode {D,ED}` | 累积深度 Σw·d 或期望深度 Σw·d/α;ED 是方法默认 | 带符号偏差 −1.10 → −0.64 ns |
| `--delay-range {z,euclid}` | 深度 × sec θ_pixel(欧氏距离而非相机 z);euclid 是方法默认 | 中位 0.88 → 0.55,P90 5.38 → 4.48,RMSE 5.26 → 4.97(判据 < 4.5 未过) |
| `--max-train-views N --subset-mode {route,fps}` | 训练位置子集:沿训练列表等间隔 / 空间最远点采样 | 密度扫描;两种规则交叉点相同 |
| `--geometry-subset {all,needles,discs,random} --geometry-fraction --geometry-seed` | 只解冻某类 / 随机比例的高斯(梯度按掩码清零) | 随机 1% 拿到 73% 的解冻增益 |
| `--train-geometry` | 解冻 means/scales/quats(lr 1.6e-5×scale / 5e-3 / 1e-3) | −27% RMSE;均值只动 6 mm,增益在尺度与旋转 |
| `--init-from state.pt [--init-geometry-only]` | 热启动 / 只取几何(颜色清零) | 转移曲线、区域结构 |
| `--no-eval` | 适配 run 不做测试通道 | 转移曲线的源适配 |
| `--iterations 0 --init-from` | 零步热启动 = 只重渲、重评估 | 把 300 张渲染补到全部 452 张 |
| `--save-renders` | **默认 −1 = 全部留出图**;`eval_baselines.py` 遇到渲染不全**拒绝打分**(`--allow-partial` 并记 `partial: true` 例外) | 第四次协议错误后加的护栏 |
| `--seed` | 数据顺序与初始化的 seed | 噪声底:转移收益 0.03 dB(大堂)/ 0.31 dB(走廊);解码中位 0.002–0.026°、0.006–0.011 ns |

落盘的 `renders/*.npy` 是**物理单位**(时延 ns,角度为 cos/sin 与归一化天顶),`eval_baselines.decode()` 直接读——`delay_oracle.py` 第一版把它当归一化单位,三个 run 全零变化,已修。

### F 评估层(哪个脚本回答哪个问题)

| 问题 | 脚本 | 输出 |
| --- | --- | --- |
| 场比朴素基线好多少(地板)、波束选择 top-k | `eval_baselines.py --multi run --truth ds [--max-train-views N] [--layout layout.json]` | `baselines_<run>.json`:constant / nearest / interp2 / rrf 的中位 / P90 / RMSE / 功率加权;`by_space` 按走廊 / 房间 / 厅 |
| 编码之间的解码误差、σ 扫描 | `eval_encoding.py [--support] [--power-weighted]` | 同支撑集 / 功率加权的中位 / P90 / RMSE |
| 时延的带符号误差 | `delay_signed.py` | 均值 / 中位 / 功率加权均值 / \|e\|>5 ns 占比与其中负的比例 |
| 几何对 mesh 有多准 | `depth_gs.py`(WSL,ED 与 D 深度)→ `depth_gt.py`(Sionna `scene.mi_scene.ray_intersect`,相机 z) | 深度 RMSE / 中位 / 带符号均值;范围项缺口 (range − z)/c |
| 测试时换真值深度会怎样 | `delay_oracle.py` | 三个 run 都变差,偏移 ≈ 几何平均偏差:残差已在训练时补偿几何,测试时替换算两遍 |
| 尾巴是不是多径 | `diag_delay_paths.py --runs …` | 按每像素路径数分层的时延误差与平方误差占比 |
| 转移收益与距离 / 点亮表面重叠 | `transfer_curve.py --tag 2k`、`diag_overlap.py`、`scene2/analyze_scene2.py` | 曲线 JSON(仪表盘卡片)、IoU 与相关 |
| 整段区域留出 | `make_region_split.py --xmin … --ymax …` | 硬链接的姊妹数据集 + 新的 train/test 索引 |

### E 场景 2 的约定

- 路线文件按空间交错写(任何前缀都铺满平面),名义 0.2 m 步长;生成器抖动后最密档最近训练距离 0.46 m。
- Mitsuba 相机帧 x 左、y 上、z 前:Blender 位姿翻 x、z 列(不是 y、z);两个语义判据定手性(灯在上半、走廊口在相机左侧)。
- 渲染集不入库(manifest + 固定 seed),checkpoint 盘外备份(`D:\RF-3DGS_backup`、OneDrive)。
- 队列 `scene2/win_scene2_rf.sh` 的 gpct 判断用 "images 非空",不用 "目录存在"。

### 队列与门(所有 `win_round*.sh` / `scene2/win_scene2_*.sh`)

- 门:没有游戏进程 且 GPU < 15% 持续 2 min(历史上 19:15–20:19 与 03:02–04:10 两段游戏占卡,期间所有计时作废)。
- `run()` 按 `results.json` 跳过;`gen()` 按 `generation_meta.json` 跳过;bash 边跑边读脚本,**运行中的脚本不能改**。
- 计时契约:只报独占卡上的中位数;共享卡上的 run 只取质量数。
