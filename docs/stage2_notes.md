# 阶段 2 实验笔记(边跑边记)

> 2026-09-18 夜间自动运行。每条都标注来源脚本/日志,结论以 `output/rrf/*/results.json` 为准。

## 新发现:教程里其实定义了材料(修正第一阶段的一个说法)

`RF-3DGS-tutorial.ipynb` cell 6(之前只看了 `sionna_nb_src.py` 的一部分)对每种材料显式给了 EM 参数、
散射系数和散射方向图,并把场景里 `itu_*` 材料按名字换成 `custom_*`:

| 材料 | εr | σ | scattering_coefficient(×global 4) | DirectivePattern α |
| --- | --- | --- | --- | --- |
| plastic | 2.3 | 0 | 0.05×4 = 0.2 | 6 |
| leather | 1.8 | 0 | 0.1×4 = 0.4 | 3 |
| cloth | 1.8 | 0 | 0.2×4 = 0.8 | 3 |
| plasterboard | 2.73 | ITU(f) | 0.4 | 5 |
| wood | 1.99 | ITU(f) | 0.8 | 3 |
| concrete | 5.24 | ITU(f) | 0.4 | 5 |
| metal | 1 | 1e7 | 0.1 | 10 |
| glass | 6.31 | ITU(f) | 0.1 | 10 |
| ceiling_board | 1.48 | ITU(f) | 0.4 | 5 |
| chipboard | 2.58 | ITU(f) | 0.4 | 5 |
| marble | 7.074 | ITU(f) | 0.2 | 10 |

所以 `sionna_port/README.md` 里"语义材料描述符未公开"的说法不对——它在 notebook 里。我们统一用的 0.7
是在"路径数 ≈ 30 万"上校准出来的等效值,落在教程的 0.1–0.8 区间内,但**逐材料不同 + 有向散射图**是
教程的设定。待办:给 `scene_common.load_radio_scene` / `generate_dataset` 加 `--materials tutorial`。
场景文件也不同:教程用 `NIST_lobby_V1.0_material_assigned.xml`,我们用 V1.1。

另外三个设置值得记录:
- 数据集生成的 cell(89/91/93/98/101)把 `scene.frequency` 设成 **2.4e9**,只有 MPC 的 cell 是 60e9;
  但材料电导率公式在 cell 6 里按 60 GHz 算好了。发布的 CBF/TCBF/MVDR/AoD/Delay 数据集很可能是 2.4 GHz 的。
- 阵列 `M = 10`,`tr38901` 方向图,`synthetic_array = True`;MVDR `time_interval = 0.1`。
- `compute_paths(max_depth=1, reflection=True, diffraction=False, scattering=True, scat_keep_prob=0.1,
  scat_random_phases=False, num_samples=1e6)`;MVDR 的全局范围来自两个探针位置
  `[6.905,-0.2,0.287]`(最强)和 `[-3.5,-9.3,0]`(最弱),各 4 个朝向。
- 相机:COLMAP 存的是 c2w 四元数 + `tvec_c2w = -R_c2w · rx_loc`。

## 数据集(sionna_port/generate_dataset.py,60 GHz,s=0.7,depth 1,1e6 samples,每视图换种子)

| 数据集 | 位置 | 生成速度 | 全局 min/max | 备注 |
| --- | --- | --- | --- | --- |
| regenerated/3dgs_MVDR_100 | 800×4 | 10.3 views/s(≈5 min) | −215.8 … −75.2 dB | 数值底噪把跨度撑到 140 dB |
| regenerated/3dgs_CBF_100 | 800×4 | 21.3 views/s | −179.2 … −33.6 dB | |
| *_gpct | 同上 | 重映射 | MVDR −175.8…−101.5(74 dB);CBF −118.3…−41.2(77 dB) | 1 / 99.99 百分位 |
| *_perview | 同上 | 重映射 | 每图自身 min/max | `per_view_ranges.csv` 留 oracle |

MVDR 浮点分布(采样 200 张):P1 −184.7,P50 −148.7,P99.9 −122.9,max −113.8;单图跨度中位数 34 dB。

划分:按位置留出 20%(160 个位置 × 4 朝向 = 640 张),与发布数据一致(发布的 test_index 正是每 4 张一组)。

## 训练器(rrf_gsplat/train_rrf.py,WSL rf-gsplat,gsplat 1.6.0)

- 几何冻结自 `blender_visual_trained/chkpnt30000.pth`(1,014,142 个高斯);SH 清零;只学 SH + opacity;
  Adam eps 1e-15,f_dc 0.0025 / f_rest 0.000125 / opacity 0.05;L1+SSIM λ=0.2;每步一张随机训练图。
- 三种 color function:`rgb`(jet RGB,3 通道,= RF-3DGS)、`db`(1 通道归一化 dB)、`power`
  (高斯携带线性功率,alpha 合成后取 10log10 再算损失)。
- 统一评估:预测 dB 对浮点真值的 RMSE/MAE;以及把预测经 jet 映射后对 PNG 的 PSNR/SSIM(与论文可比)。
  RGB 模式的 dB 由 jet 逆映射得到(4096 点最近邻 LUT,往返最大误差 1.4% 跨度)。
- 踩坑:checkpoint 里的张量是 `nn.Parameter`,`exp()/normalize()` 后成为图节点,第二步反传报
  "backward through the graph a second time"——必须 `detach()`。
- Windows 符号链接 WSL 不能跟随,`spectra_float` 改为 NTFS 硬链接。

## 结果(持续追加)

| run | 数据 | mode | PSNR(jet) | SSIM | RMSE dB | it/s | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c0_released_mvdr_rgb | 发布 MVDR | rgb | **15.97** | 0.727 | (无浮点真值) | 13.1 | 发布 checkpoint 16.02 / 0.731;10k 步 763 s;运行中子集 64 张到 16.36 |

C0 结论:**gsplat 上的 RRF 微调复现了 RF-3DGS**(640 张测试图 PSNR 15.97 vs 16.02,SSIM 0.727 vs 0.731),
之后所有 color function / 归一化 / 消融实验都在同一份训练器上做,差异只来自被测的变量。

## 原版流水线耗时(rrf_gsplat/time_tutorial_019.py,Sionna 0.19.1 + TF 2.15.1,WSL CPU/LLVM)

教程 notebook 自己的代码(cell 6 场景 + 材料,cell 93 阵列 M=10 tr38901,2.4 GHz,`compute_paths(max_depth=1,
scattering, scat_keep_prob=0.1, num_samples=1e6)` + `MVDR_spectrum(time_interval=0.1)`),2 个位置 × 4 朝向:

| | 每视图 | 3200 视图 |
| --- | --- | --- |
| compute_paths | 0.4–0.6 s(62k 条路径) | |
| MVDR_spectrum(numpy) | 0.5 s | |
| 合计 | ≈1.1 s | **≈1.0 h** |

比我之前猜的"数小时"快得多——0.19 的 CPU 路径求解并不慢;但它是 2.4 GHz、6 万条路径。我们的 2.1/GPU
版本 60 GHz、31 万条路径、每视图 0.097 s(10.3 views/s),3200 视图 5 分钟。同频同设置的对比待补(见下)。
场景加载:0.19 无法读 V1.1 的 Sionna 2 格式材料,用 `NIST_lobby_V1.0_material_assigned_ascii.xml`(网格名按
`ascii_rename_map.json` 转写)。TF 在 WSL 下无 GPU,OptiX 也没有,所以这是 CPU 数字;作者的 GPU 机器会更快。

## Tx 移动一次的成本(2026-09-18 02:02)

| 流水线 | 每视图 | 3200 视图 | RRF 微调 10k 步 |
| --- | --- | --- | --- |
| 教程(Sionna 0.19 + TF,CPU,2.4 GHz,62k 路径) | 1.1 s | ≈1.0 h | INRIA 光栅器 272 s(37 it/s,RTX 3060) |
| 本移植(Sionna 2.1 + OptiX GPU,2.4 GHz 同设置) | 0.068 s(14.8 views/s) | 3.6 min | 同上 |
| 本移植,60 GHz,312k 路径 | 0.097 s(10.3 views/s) | 5.2 min | gsplat 763 s(13 it/s) |
| 本移植,Tx 移到 (8.2, −5.4, 2.0),60 GHz | 0.102 s(9.8 views/s) | 5.4 min | 冷/热启动实验见下 |

- 同设置(2.4 GHz)下数据集生成快 **16×**;60 GHz 且路径数 5× 时仍快 11×。
- **意外**:gsplat 版训练器 13 it/s,比 INRIA 光栅器的 37 it/s 慢 3×。原因待剖析(torch 侧对 100 万高斯逐步算
  SH、python 开销);INRIA 的 SH 在 CUDA 里。INRIA `--eval` 报告的测试 PSNR 16.82(其内部评估,非 render+metrics)。
- Tx-B 数据集的全局范围是 −228.5…−69.9 dB;重映射到 Tx-A 的 −175.75…−101.52,这样热启动看到同一套 dB 映射。

## 训练器提速(rrf_gsplat/profile_step.py,1,014,142 高斯,884k 在视野内)

| 一步(fwd+bwd) | rgb | db |
| --- | --- | --- |
| 原始:torch `eval_sh` 走 autograd | 101 ms | 36 ms |
| SH 基在 no_grad 下算好、颜色 = 系数的线性映射 | 65 ms | 23 ms |
| rgb 改用 gsplat 内置 CUDA SH(系数 [N,K,3] 布局) | 49 ms | 22 ms |

INRIA 光栅器 27 ms/步(37 it/s)仍比 gsplat 的 rgb 路径快;gsplat 的价值在通道数,不在速度。db/power 模式 22 ms。
`sh_basis` 与 INRIA `eval_sh` 逐项一致(最大差 1e-6)。

## 结果:实验矩阵(2026-09-18 02:10–03:07,快路径训练器,每组 10k 步,640 张按位置留出的测试图)

数据 = 重生成的 MVDR(60 GHz,全局 1/99.99 百分位范围 −175.75…−101.52 dB),除非注明。

### E2 color function

| mode | PSNR(jet) | SSIM | RMSE dB | MAE dB | it/s | 到 PSNR 17 的秒数 |
| --- | --- | --- | --- | --- | --- | --- |
| rgb(= RF-3DGS,jet RGB 三通道) | 18.15 | 0.802 | 4.42 | 3.22 | 47.5 | 69 |
| **db(单通道归一化 dB)** | **18.75** | **0.818** | **3.91** | **2.79** | 59.5 | **19** |
| power(线性功率合成,dB 域损失) | 18.63 | 0.812 | 3.97 | 2.85 | 58.1 | 20 |

- 把训练目标从伪彩图换成谱本身:RMSE 降 0.5 dB(11%),PSNR 升 0.6 dB,到同一质量快 3.6×(1k 步时 db 17.5 vs rgb 12.8)。
- power 与 db 几乎相同(略差 0.06 dB)。物理上"功率相加"的合成规则没有带来额外收益——alpha 合成的权重
  本来就是凸组合,在 dB 与线性域之间差别被 SH 的表达力吸收了。需要再看:更大跨度/更强 LoS 峰的场景。

### E1 归一化(rgb 模式,同一份浮点谱)

| 数据 | 归一化 | PSNR(jet) | SSIM | RMSE dB(oracle 范围) |
| --- | --- | --- | --- | --- |
| CBF | 全局 | **13.68** | 0.592 | **7.54** |
| CBF | 逐图 min/max(教程的 CBF/TCBF 做法) | 13.09 | 0.580 | 11.03 |
| MVDR | 全局 | **18.15** | 0.802 | **4.42** |
| MVDR | 逐图 min/max | 11.97 | 0.582 | 7.61 |

- 逐图归一化在 MVDR 上损失 6.2 dB PSNR、3.2 dB RMSE;CBF 上 0.6 dB PSNR、3.5 dB RMSE——而且这还是给了
  oracle 范围的最好情况,实际使用时绝对电平根本不在图里。**论文里 CBF/TCBF 偏弱的一个原因就是这个,已证实。**
- CBF 本身比 MVDR 难(13.7 vs 18.2):主瓣宽、旁瓣多,视角相关性更强。

### 消融(db 模式)

| 变量 | PSNR(jet) | SSIM | RMSE dB | it/s | 训练 s |
| --- | --- | --- | --- | --- | --- |
| 基准 SH3,opacity 训练,2560 张,10k 步 | 18.75 | 0.818 | 3.91 | 59.5 | 168 |
| SH0(无视角依赖) | 16.75 | 0.753 | 4.91 | 179.7 | 56 |
| SH1 | 17.43 | 0.786 | 4.58 | 130.9 | 76 |
| opacity 冻结 | 18.53 | 0.816 | 3.99 | 51.2 | 195 |
| 2k 步 | 17.45 | 0.781 | 4.70 | 55.0 | **36** |
| 640 张训练图(160 个位置 × 4 朝向) | 18.69 | 0.812 | 3.98 | 28* | 351 |
| 160 张训练图(40 个位置 × 4 朝向) | 17.42 | 0.761 | 4.71 | 25* | 402 |

\* 与 Tx 移动组并行跑,it/s 偏低。第一版 views 消融按"每 4 张取 1 张"抽样,结果只取到同一朝向(13.3 dB),
是抽样错误不是数据量效应;改成按位置整组保留后重跑。
- **数据效率**:位置数减到 1/4(160 个位置)只掉 0.06 dB PSNR / 0.07 dB RMSE;减到 1/20(40 个位置)掉 1.3 dB。
  Tx 移动后重建时,数据集可以只生成 160 个位置——生成时间从 5.2 min 降到约 1 min。

- 视角依赖是主要的表达力来源:SH0→SH3 差 2 dB PSNR、1 dB RMSE。射频谱的镜面峰确实高度视角相关(第 2.2 节的假设 1)。
- opacity 训练只值 0.2 dB;几何来自视觉、"哪里有东西"基本不用改。
- 36 秒的 2k 步已到 17.45 dB——Tx 移动后的"可用"模型不需要 10k 步。

### 基线复验

| run | PSNR(jet) | SSIM | 备注 |
| --- | --- | --- | --- |
| c0_released_mvdr_rgb(torch SH) | 15.97 | 0.727 | 763 s |
| c0b_released_mvdr_rgb(gsplat CUDA SH) | 15.97 | 0.727 | 226 s,逐位相同 |
| 发布 checkpoint | 16.02 | 0.731 | |
| INRIA train.py 同设置(`--eval` 内部评估) | 16.82 | | 272 s |

发布数据 15.97 vs 我们重生成数据 18.15(同为 rgb):不是同一份数据,不能直接比,但说明重生成的数据更"可拟合"
(全局范围从数据来、逐视图换种子)。

### Tx 移动:冷启动 vs 热启动(Tx-B = (8.2, −5.4, 2.0),数据用 Tx-A 的色标范围,15% 像素低于下限)

| run | mode | 初始化 | PSNR(jet) | RMSE dB | RMSE(范围内) | 到 PSNR 17 的步数 |
| --- | --- | --- | --- | --- | --- | --- |
| t_txB_cold | db | SH 清零(RF-3DGS 的做法) | 17.86 | 7.35 | 4.82 | 1500 |
| t_txB_warm | db | Tx-A 的 db 模型 | 17.99 | 7.23 | 4.67 | 1250 |
| t_txB_cold_rgb | rgb | SH 清零 | 17.70 | 7.76 | 5.19 | 2000 |
| t_txB_warm_rgb | rgb | Tx-A 的 rgb 模型 | 17.47 | 7.80 | 5.24 | **3250** |

- 250 步时 warm 11.4 vs cold 8.6,1000 步 16.6 vs 16.0:**热启动在 db 模式下省约 250 步(17%)**,终值相同。
- **rgb 模式热启动反而有害**(到 17 dB 从 2000 步变 3250 步,终值低 0.23 dB):Tx-A 的伪彩合成对新场是误导性初值。
  这是"颜色 ≠ 物理量"的又一个后果。
- 未裁剪 RMSE 7.3 dB 大半来自 15% 低于共享下限的像素(范围内 4.8 dB);Tx-B 在 2 m 高、贴东墙,阴影区更多,
  本身也比 Tx-A(3.9)更难。

### "Tx 动了要多久"——把数字串起来(RTX 3060)

| 步骤 | RF-3DGS 原流水线 | 本工作 |
| --- | --- | --- |
| 数据集(800 位置 × 4 视图) | ≈1.0 h(Sionna 0.19,CPU;2.4 GHz) | 5.4 min(Sionna 2.1 OptiX;60 GHz,5× 路径) |
| 数据集(160 位置,质量 −0.06 dB) | — | ≈1.1 min |
| 微调到 PSNR 17 | INRIA 光栅器 37 it/s;rgb 需 ≈2000 步 ≈ 54 s | db 56 it/s,1500 步 ≈ 27 s(热启动 1250 步 ≈ 22 s) |
| 微调 10k 步 | 272 s | 168 s(db)/ 211 s(rgb) |
| **合计,Tx 移动一次** | **≈1 h** | **≈6 min;用 160 个位置 ≈1.5 min** |

原流水线的数据集时间是本机 CPU 的实测(作者的 GPU 机器会更快,但 0.19 也不能用 OptiX 之外的 GPU 路径……
实际上 0.19 支持 CUDA 变体,作者机器上可能是几分钟量级;这里只对比同一台机器)。

## 教程材料在 Sionna 2.1 上的复现(sionna_port/tutorial_materials.py,03:40)

按名字把每个物体的材料换成教程 cell 6 的定义(33 个合并后的物体,9 个材料类),同一 Tx/Rx 对、depth 1、1e6 samples:

| 材料设定 | 路径数 | 总功率 | 最强路径 |
| --- | --- | --- | --- |
| 统一 s=0.7(第一阶段的校准) | 311,372 | −79.9 dB | −80.8 dB |
| 教程 as-is | 6,753 | −80.8 dB | −80.8 dB |
| 教程,频率单位修正 | 104,443 | −79.7 dB | −80.8 dB |

- **教程的电导率公式有个单位错误**:`0.0085*(scene_freq/1e-9)**0.9395` 把 60 GHz 除以 1e-9 而不是 1e9,
  石膏板/混凝土/玻璃/木头/天花板的 σ 变成 1e16–1e24 S/m——全是完美导体。as-is 复现只剩 6.7k 条路径。
  修成 f(GHz) 后 σ 回到 ITU-R P.2040 的 0.09–3.7 S/m。
- 修正后的教程材料给 10 万条路径,总功率与统一 0.7 几乎一样(LoS 主导);路径数是 0.7 校准值的 1/3。
  论文的"> 30 万"是 0.19 采样器(scat_keep_prob)的口径,与 2.1 的采样不能直接对齐。
- 生成器新增 `--materials {uniform,tutorial,tutorial-asis}`,材料消融见下。

### E2 补充:CBF 上的 db 目标(e2_cbf_db,03:38)

| 数据 | mode | PSNR(jet) | SSIM | RMSE dB |
| --- | --- | --- | --- | --- |
| CBF 全局范围 | rgb | 13.68 | 0.592 | 7.54 |
| CBF 全局范围 | **db** | **13.95** | **0.605** | **7.04** |

dB 目标在 CBF 上同样有效(RMSE −7%,PSNR +0.27),幅度比 MVDR 小;CBF 的主要困难在谱本身(宽主瓣、强视角相关)。

### E3 材料消融(教程逐材料定义,频率单位修正;数据集 3dgs_MVDR_tut_gpct,范围 −187.2…−104.1 dB,跨度 83 dB)

| 材料 | mode | PSNR(jet) | SSIM | RMSE dB | MAE dB |
| --- | --- | --- | --- | --- | --- |
| 统一 s=0.7 | rgb | 18.15 | 0.802 | 4.42 | 3.22 |
| 统一 s=0.7 | db | 18.75 | 0.818 | 3.91 | 2.79 |
| 教程逐材料 | rgb | 17.29 | 0.772 | 5.33 | 4.00 |
| 教程逐材料 | **db** | **18.04** | **0.791** | **4.68** | **3.50** |

- color function 的结论在作者的材料模型下成立,而且差距更大:db 比 rgb 高 0.75 dB PSNR、RMSE 低 12%(统一材料下 0.6 dB / 11%)。
- 逐材料 + 有向散射图的场更难拟合(两种 mode 都低 0.7 dB 左右):跨度更大(83 vs 74 dB),镜面成分更强、视角相关性更高。
- 生成速度不变(10.2 views/s)。

### E4 与发布数据同设置(2.4 GHz、教程材料、M=10 tr38901;数据集 3dgs_MVDR_24ghz_tut_gpct,范围来自 1/99.99 百分位)

| 数据 | mode | PSNR(jet) | SSIM | RMSE dB |
| --- | --- | --- | --- | --- |
| 发布 MVDR 数据集(作者生成) | rgb(发布 checkpoint) | 16.02 | 0.731 | — |
| 发布 MVDR 数据集 | rgb(本训练器 c0) | 15.97 | 0.727 | — |
| 本生成器,2.4 GHz + 教程材料 | rgb | 16.87 | 0.750 | 5.57 |
| 本生成器,2.4 GHz + 教程材料 | **db** | **17.66** | **0.773** | **4.87** |

- 在作者的频率与材料设定下重生成的数据,rgb 训练得到 16.87 dB,与发布数据上的 15.97 相差 0.9 dB——
  两条流水线产出的数据"可拟合程度"接近,剩余差距来自全局范围(从数据来 vs 两个探针)、逐视图种子和采样器差异。
- db 目标在这套数据上再 +0.79 dB / RMSE −13%。三种材料/频率设定下 db 都赢,结论稳健。
- 2.4 GHz 生成 14.4 views/s(3.7 min)。

### Tx 移动实测:只生成 160 个位置(t_txB160_*,04:10)

数据集 3dgs_MVDR_txB160:160 个位置 × 4 = 640 视图,**生成 33 s**(20.5 views/s);位姿与 800 位置版逐位相同(同一 rng 抖动),
测试用的 640 张留出视图从完整 Tx-B 数据集复制,与之前的 Tx-B 实验同一套真值。训练集 492 张(160 位置中去掉留出的)。

| run | 训练位置 | 初始化 | PSNR(jet) | RMSE(范围内) | 到 PSNR 17 | 训练总时 10k 步 |
| --- | --- | --- | --- | --- | --- | --- |
| t_txB_cold | 640 | 清零 | 17.86 | 4.82 | 1500 步 | 468 s* |
| t_txB160_cold | 160 | 清零 | 17.45 | 5.11 | 1500 步 / **32 s** | 184 s |
| t_txB160_warm | 160 | Tx-A 的 db 模型 | 17.62 | 4.90 | 1250 步 / **23 s** | 174 s |

\* 与其他任务并行,it/s 偏低。

**一次 Tx 移动的实测成本(RTX 3060,60 GHz,312k 路径/视图)**:

| | 数据集 | 到 PSNR 17 | 合计 |
| --- | --- | --- | --- |
| 800 个位置,冷启动 | 5.4 min | 27 s(1500 步 @56 it/s) | ≈6 min |
| **160 个位置,冷启动** | **33 s** | **32 s** | **65 s** |
| 160 个位置,热启动 | 33 s | 23 s | 56 s |
| RF-3DGS 教程流水线(本机 CPU) | ≈1 h | 54 s(INRIA,2000 步) | ≈1 h |

160 个位置的代价是终值低 0.4 dB(17.45 vs 17.86)。

## 光栅器速度 vs 分辨率(rrf_gsplat/bench_resolution.py,同一 1.01M 高斯,一步 = 渲染 + L1 + 反传到 SH 与 opacity)

| 分辨率 | INRIA(Windows,rf-3dgs) | gsplat 1.6(WSL) |
| --- | --- | --- |
| 300×200 | 6.8 ms(148 it/s) | **4.9 ms**(202 it/s) |
| 600×400 | 12.2 ms | **5.5 ms** |
| 1200×800 | 33.4 ms | **9.9 ms** |
| 2400×1600 | 120.4 ms | **27.2 ms** |

- 之前"INRIA 37 it/s 比 gsplat 快"的说法是错的:那是两条完整训练循环的对比(含 Adam、SSIM、数据取样),
  光栅器本身 gsplat 在每个分辨率上都更快,分辨率越高差距越大(2400×1600 时 4.4×)。
- 由此也看出我们训练循环的开销:光栅器一步 5 ms,而 db 训练一步 22 ms、rgb 49 ms——大头在 SH 张量的
  cat/反传拷贝和 Adam 更新(4900 万参数);rgb 路径还多一次 [N,K,3] 拼接。可优化,但与结论无关。

## 频率核实:发布数据集是 2.4 GHz(sionna_port/generate_dataset.py --poses-from,output/freqcheck/)

在发布 CBF/MVDR 数据集**完全相同的 48 个位姿**(从 images.txt 反算:rx = −Rᵀt,yaw 匹配)上,用教程材料分别在 2.4 GHz
和 60 GHz 生成谱,与发布数据对比:

| 证据 | 2.4 GHz | 60 GHz |
| --- | --- | --- |
| 发布 CBF 目录的 `cbf_power.csv`(均值 −27.4 dB,std 13.3)vs 我们 CBF 谱峰值 | 均值 −29.1,**偏差 +1.6 dB(std 4.2)** | 均值 −61.6,偏差 +34.2 dB |
| 逐视图相关(csv vs 我们的峰值) | 0.962 | 0.968 |
| 与发布 PNG 的结构相关(逐图归一化后),CBF | 0.552 | 0.553 |
| 与发布 PNG 的结构相关,MVDR | 0.820 | 0.854 |

- 结构相关两个频率几乎一样(阵列响应按波长归一化,谱的形状对频率不敏感),**绝对电平是决定性的**:
  csv 里记录的 CBF 功率与 2.4 GHz 的计算值只差 1.6 dB,与 60 GHz 差 34 dB(自由空间 28 dB + 材料差异)。
- csv 与图片在同一个生成循环里写出,所以发布的 CBF/TCBF/MVDR/AoD/Delay 数据集是 **2.4 GHz** 的,与 notebook
  的 `scene.frequency = 2.4e9` 一致;只有 MPC 的 cell 是 60 GHz。论文正文和 README 说的 60 GHz / mmWave 与发布数据不符。
- 我们第一阶段"s=0.7 在 60 GHz 复现 30 万条路径"因此只是与论文数字的巧合,不是与发布数据的对齐。
- MVDR 与发布数据的结构相关 0.82–0.85,说明移植的 MVDR 谱与作者的基本一致;CBF 只有 0.55(裸导向矢量 vs 流形矢量、
  逐图归一化等细节还有差异)。

## gsplat 相机模型(10 分钟核查)

`gsplat.rasterization(camera_model=...)` 支持 `pinhole | ortho | fisheye | ftheta`,**没有 equirect**。实测(rrf_gsplat 里的
一次性脚本):fisheye(等距投影 r = f·θ)无论把 f 设成多大 FoV,图像内最大 θ = 89.8°——它在 z ≤ 0 处剔除,**单张最多一个半球**;
ftheta 需要 `FThetaCameraDistortionParameters` torch 类和 `with_ut=True`。结论:一次性消掉四面拼接的三个问题做不到,
但**两张 180° fisheye(前/后半球)**把接缝从 4 条减到 1 条(赤道大圆)、消掉极冠、且等距投影的立体角 Jacobian 光滑(sinθ/θ)。
代价:生成端要按 fisheye 角网格算谱(CBF/MVDR 的导向矢量网格可以任意定义,投影谱本来就是等距柱状重采样),训练端加
`camera_model="fisheye"`(建议 `with_ut=True`)。

### E1 重跑:发布数据的设定(CBF,2.4 GHz,教程材料;数据集 3dgs_CBF_24ghz_tut_*,08:01)

| 归一化 | mode | PSNR(jet) | SSIM | RMSE dB(逐图用 oracle 范围) |
| --- | --- | --- | --- | --- |
| 全局 1/99.99 百分位 | rgb | 13.00 | 0.511 | 9.32 |
| 逐图 min/max(教程对 CBF 的做法) | rgb | 12.49 | 0.503 | **13.35** |
| 全局 | **db** | **13.39** | **0.542** | **8.54** |

- 在发布数据的频率与材料下结论不变:逐图归一化让 dB 误差 +4.0 dB(+43%),PSNR −0.5;db 目标 RMSE −8%。
- 2.4 GHz 的 CBF 比 60 GHz 版(13.68 / 7.54)更难一点;全局范围 −160.8…−1.0 dB 里跨度更大。

## 第四轮:几何、MCMC、多通道(08:07–08:43)

### 几何解冻与 MCMC 致密化(db 模式,重生成 MVDR 60 GHz)

| run | 几何 | 致密化 | PSNR(jet) | SSIM | RMSE dB | 高斯数 | 训练 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e2_mvdr_db(基准) | 冻结(RF-3DGS 的设定) | 无 | 18.75 | 0.818 | 3.91 | 1,014,142 | 168 |
| **a_mvdr_db_geom** | **解冻**(means 1.6e-5×scale,scales 5e-3,quats 1e-3) | 无 | **21.53** | **0.868** | **2.85** | 1,014,142 | 302 |
| a_mvdr_db_mcmc | 解冻 | MCMC,cap = N | 4k 步起 loss = NaN(1k 步 14.7 → 6k 步 5.4),随后重定位采样 `_multinomial_sample` 收到 NaN 概率崩溃 | | | | |
| a_mvdr_db_mcmc_grow | 解冻 | MCMC,cap 1.3M | 5.25(1k 步 14.7 → 8k 步 5.3,发散) | 0.417 | 28.2 | 1,300,000 | 579 |

- **让几何跟着射频走是本阶段最大的单项收益**:RMSE −27%、PSNR +2.8 dB。RF-3DGS 冻结几何的设计假设"视觉几何就是射频几何",
  这个假设值 1 dB RMSE。视觉里的一张桌面在 60 GHz 是一面镜子,高斯该挪到反射发生的地方。
- MCMC 用 gsplat 默认超参(noise_lr 5e5、每 100 步重定位、从 500 步开始)在这个"从收敛几何微调"的场景下**发散**:
  噪声注入和重定位把已经对的高斯打散,PSNR 从 14.7 一路掉到 5.3;两种 cap 都在 4k 步前后出现 NaN 损失(cap = N 的版本因此在重定位采样里崩溃)。
  需要的是"小噪声、晚开始、只重定位不新增"的配置——待调,当前结论是负的,不是"MCMC 没用"。

### 多通道目标(2.4 GHz,教程材料;MULTI = 功率 / AoD 方位 / AoD 天顶 / 时延各一通道;AOD3 = 教程的角度×幅度 RGB 编码)

逐通道误差(有路径的像素上,640 张留出图):

| 表示 | 功率 RMSE(范围内) | AoD 方位 RMSE | AoD 天顶 RMSE | 时延 RMSE |
| --- | --- | --- | --- | --- |
| MULTI(`evaluate_multi`,逐图 RMSE 再平均) | 12.9 dB | 16.1° | 11.3° | 7.7 ns |
| MULTI(`eval_encoding`,全像素合并) | | **19.0°** | **11.6°** | **8.2 ns** |
| AOD3 解码(`10^((v_c − v_B)/10)`) | — | 48.6° | 56.6° | 无时延通道 |

- **分通道之后角度误差是角度×幅度编码的 1/3–1/5**。教程那套"把角度乘进幅度再取对数"的编码,解码要做两个学到的通道相除,
  误差被放大;多通道下它的必要性没有了——这是 gsplat 任意通道数的第一个直接兑现。
- 功率通道的 12.9 dB 看着大,是因为 MPC 谱(逐路径散射)跨度 114 dB、且高斯核散射出的谱比 MVDR 尖得多;
  它和 MVDR 的 3.9 dB 不是同一个量。
- 未裁剪的 `rmse_db`(40 dB)包含 −200 dB 的空像素,没有意义;看 `rmse_db_in_range` 和逐通道值。
