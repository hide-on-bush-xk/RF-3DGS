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
