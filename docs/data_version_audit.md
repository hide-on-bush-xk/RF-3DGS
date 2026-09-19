# 数据版本审计(2026-09-19):每个实验跑在哪个版本的生成器上

按 Ke 的要求:实验 × 数据生成版本(seed 方式 / 四面派生 / 归一化 / 通道数),标出同版本内比较、跨版本比较、需要在干净数据上重做的结论。
来源:`git log` 对 `sionna_port/generate_dataset.py`、`rf_spectra.py`、`rrf_gsplat/renormalize.py`;每个数据集 `generation_meta.json` 的写入时间;每个 run 的 `results.json.config.source`。

## 生成器版本

| 版本 | 提交 | 时间 | seed | 四面来源 | 备注 |
| --- | --- | --- | --- | --- | --- |
| **G1** | 5ebfcc9 | 09-17 15:44 | **按视角**(seed = 42 + 视角序号) | 每个视角自己 solve(波束族必须;投影族当时也是) | "move the lattice per view" |
| **G2** | 69e47a3 | 09-18 09:14 | **按位置**(seed = 42 + 位置序号) | 投影族(MULTI/AOD3)一次 solve 派生四面;波束族每视角 solve 但共用位置的 seed | 修复跨视角不一致 |
| **G3** | 261d753 | 09-18 18:20 | 同 G2 | 同 G2 | 投影族溅射在 −150 dB 截断;`channel_ranges` hit 门限同步 |
| **G3y** | b16e8cc | 09-18 20:04 | `--seed-per-view` 显式恢复 G1 的按视角(A/B 用) | 每视角 solve | 只用于 cs_yawseed |
| **发布管线**(RF-3DGS 教程 0.19) | — | — | 全局 seed 一次(`np.random.seed(1); tf.random.set_seed(1)`),每个 `compute_paths` 按 `scat_keep_prob` 随机保留 1–10% 的散射路径 → **相邻视角抽到不同的漫散射子集** | 每视角 solve | 按构造带有按视角不一致 |

归一化:`raw` = 生成器自己的全局 min–max(spec_min 被底/尾巴定,140–146 dB 跨度);`gpct` = `renormalize.py --pct 1 99.99`;`perview` = 逐图;MULTI 的功率通道按 hit 像素百分位。
通道数:MULTI 旧版 4(方位单通道),cs 起 5((cos, sin))。

## 数据集 → 版本

| 数据集 | 写入时间 | 版本 | 归一化 | 通道 |
| --- | --- | --- | --- | --- |
| 3dgs_MVDR_100 / _gpct / _perview,3dgs_CBF_100 / _gpct / _perview | 09-18 01:33–01:37 | **G1** | raw / gpct / perview | 1 |
| 3dgs_MVDR_txB / _gpct(Tx-A 范围) | 09-18 01:57 | **G1** | gpct(A 的范围) | 1 |
| 3dgs_MVDR_tut / _gpct(60 GHz 教程材料) | 09-18 03:43 | **G1** | gpct | 1 |
| 3dgs_MVDR_txB160 / _gpct | 09-18 03:52 | **G1** | gpct(A 的范围) | 1 |
| 3dgs_MVDR_24ghz_tut / _gpct,3dgs_CBF_24ghz_tut / _gpct / _perview | 09-18 03:56 / 07:46 | **G1** | gpct / perview | 1 |
| 3dgs_AOD3_24ghz_tut,3dgs_MULTI_24ghz_tut | 09-18 08:06–08:07 | **G1** | raw(投影族自带范围) | 3 / **4** |
| 3dgs_MULTI_24ghz_tut_s1 | 09-18 09:14 | **G1/G2 不明**(与提交同分钟;meta 未记 seed 方式) | raw | 4 |
| live_tx_*(交互平台) | 09-18 09:43 / 11:09 | G2 | gpct | 1 |
| 3dgs_MULTI_24ghz_tut_cs | 09-18 10:04 | **G2** | raw(hit 百分位) | 5 |
| 3dgs_MVDR_txC / _gpct | 09-18 11:16 | **G2** | gpct(自身) | 1 |
| 3dgs_MULTI_24ghz_tut_cs_s1 / _s6 / _s13 / _lin5 | 09-18 11:17–11:21 | **G2** | raw | 5 |
| 3dgs_MVDR_txD … txK / _gpct | 09-18 18:40–20:56 | G2(D 在 G3 提交前 20 min;MVDR 不受 G3 影响) | gpct(自身) | 1 |
| 3dgs_MULTI_24ghz_tut_cs_yawseed | 09-18 18:52 | **G3y**(按视角 seed + 截断) | raw | 5 |
| 3dgs_MVDR_100_posseed(round 14) | 待生成 | G3 | gpct | 1 |
| 发布数据 training-rf-spectrum/3dgs_MVDR_100 | 作者提供 | 发布管线 | 作者按类型两探针归一化 | 3(jet) |

## 实验 → 版本 → 结论状态

| 实验 / 结论 | 两臂的数据 | 版本 | 比较性质 | 结论状态 |
| --- | --- | --- | --- | --- |
| C0 复现(15.97 vs 16.02) | 发布数据 | 发布管线 | 同版本 | 成立(锚点) |
| E2 颜色函数:db 18.75/3.91 vs rgb 18.15/4.42 vs power(MVDR_100_gpct) | 同一数据集 | G1 | 同版本内,**但两臂对不一致的鲁棒性可能不同**(L1 在 dB 域的最优是物理中位数,在 jet RGB 域不是)→ db 优势可能被污染数据放大 | **需在 G3 数据上重做**(round 15:e2_mvdr_rgb_posseed vs e2_mvdr_db_posseed) |
| db 收敛 3.6× 更快(a_mvdr_db_iters2k 等) | 同上 | G1 | 同上 | 同上,随 E2 重做 |
| E1 归一化:gpct vs perview(MVDR rgb 18.15 → 11.97;CBF;2.4 GHz CBF) | 同一 G1 生成、不同归一化 | G1 | 同版本;两臂同样带按视角 seed;归一化效应与 seed 效应可能叠加但方向独立 | 成立作为 G1 内的结论;干净数据上的量值待定(低优先) |
| E3 材料、E4 2.4 GHz(db vs rgb) | 各自数据集 | G1 | 同版本 | 同 E2 的保留 |
| 几何解冻 −27%(a_mvdr_db_geom vs e2_mvdr_db) | MVDR_100_gpct | G1 | 同版本 | **round 14 重做**(posseed);若增益缩小,说明一部分几何在吸收 seed 不一致 |
| 子集解冻(针/碟/随机 1–11%)、N_eff、位移/形状诊断 | MVDR_100_gpct | G1 | 同版本 | 相对结论(类别无关、低维)保留;绝对值随解冻增益重立 |
| 24/76 分解(t_txB_cold / geomA_frozen / geom) | txB_gpct(G1),源 A 的适配在 G1 | G1 | 同版本 | **round 15 重做**(txB posseed + A posseed 适配 + 三条 run) |
| T1/T2(C→B 有害、power 解冻同增益) | txB G1;C 的适配在 **G2** | **跨版本**(源 C 是 G2,源 A 是 G1;目标 B 是 G1) | 目标一致,源不一致 | 定性保留;C 点在 G2 源上,A 点在 G1 源上——曲线里的 A 点与其它点源版本不同 |
| 转移曲线 10 点(2k) | 目标 txB G1;源 A(G1)、C/D/E/N/L/O/H/M/K(G2) | 混合 | y 轴一致(同一目标),源版本混合;区域二元结构不依赖 A 一个点 | 二元结论保留;A 点标注版本;干净重做需 txB(G3)+ 10 个源(G3):**不重做**,标注 |
| Tx 换位速度(160 位置,33 s + 32 s) | txB160 G1 | G1 | 计时,不依赖一致性 | 成立(计时另按契约) |
| 多通道 12.9(m_multi_24_tut,G1,4ch)vs 8.57(cs,G2,5ch) | **跨版本** | G1 vs G2 | 已由 cs_yawseed(G3y)归因:seed 4.1 dB,其余 ≤ 0.25 | 成立(已归因) |
| (cos, sin) vs 单通道(cs vs lin5)、σ 扫描(cs_s1/s6/s13)、同支撑集/功率加权 | 同版本 | G2 | 同版本 | 成立 |
| cs_yawseed vs cs(seed 归因) | G3y vs G2 | 跨版本但只差 seed 方式 + 截断(截断只动 hit 门限以下的像素,训练掩码之外) | 受控 A/B | 成立;截断差异标注 |
| E1 重跑(发布设定,CBF 2.4 GHz) | CBF_24ghz_tut | G1 | 同版本 | 同 E2 的保留 |
| smoke v2、多跳份额、AoA 仰角 | 直接 solve | G3 | — | 成立 |

## 需要在干净数据(G3,按位置 seed)上重做的清单(按优先级)

1. **round 14**:e2_mvdr_db_posseed、a_mvdr_db_geom_posseed——第三个一致性数字 + 解冻增益重立。
2. **round 15(条件:round 14 的效应 ≥ 0.1 dB RMSE 或 ≥ 0.3 dB PSNR)**:e2_mvdr_rgb_posseed(db/rgb 差距重做);txB posseed 数据集 + t_txB_cold/geomA_frozen/geom 三条(24/76 重做)。
3. 不重做、只标注:转移曲线(源版本混合,二元结论稳)、E1 归一化(同版本内成立)、多通道系列(G2 内成立)。

## 对审稿人的一句话

所有 MVDR/CBF 头条实验的数据来自 G1(按视角 seed),与发布管线的构造一致;G2/G3 修复后的重做在 round 14/15 里;两个版本内部的比较各自成立,跨版本的比较只在归因实验(cs_yawseed)里出现且是受控的。
