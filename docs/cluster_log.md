# 集群日志:VT ARC Falcon 上的 RF-3DGS

集群这边的结果只写在这里(`docs/cluster_handover.md` §1);新脚本只放 `rrf_gsplat/cluster/`。本机把结果并进 `stage2_notes.md`。

## 0. 环境(2026-09-25)

以后的作业照这里用。

| 项 | 值 |
| --- | --- |
| 集群 / 登录节点 | Falcon,`falcon1`(Rocky Linux 9.7) |
| Slurm 账号 | `ai_wireless_hu_lab`(Ke 名下唯一的账号;owner rosehu,额度 2,000,000,开始时已用 0)。Ke 给的 allocation ID 7746 在命令行查不到,应是 ColdFront 编号,`-A` 用账号名 |
| 分区 | `a30_normal_q`(默认分区):`fal001–032`,每节点 4 × A30、64 核、512 GB。QoS 按时长自动分配(`fal_a30_normal_short` ≤ 1 天 / `base` ≤ 7 天 / `long` ≤ 14 天;`int` 最多同时 6 个作业),不用写 `--qos` |
| 其他 GPU 分区 | `l40s_normal_q`(4 × L40S)、`v100_normal_q`(2 × V100)、`t4_normal_q`(1 × T4);各有 `*_preemptable_q` |
| conda | `module load Miniforge3/25.11.0-1`,只用 conda-forge(Ke:不用 Anaconda defaults 频道)。环境 `~/envs/rf-gsplat`,python 3.11 |
| CUDA | `module load CUDA/12.8.0`(只在编译 gsplat 时需要);主机编译器用系统 GCC 11.5.0。**不是交接文件推的 12.6**:gsplat 28e794ca 调用 `::cuda::ceil_div`,CUDA 12.6 自带的 CCCL 没有它,第一次装环境(作业 599529)在这里编译失败;12.8 的 CCCL 有(`nvcc -E` 核实),gsplat 官方 GPU 测试用 12.9 |
| torch | 2.9.1(与本机同版本),cu128 轮子(与 nvcc 12.8 对应;torch 2.9.1 没有 cu129 轮子;本机是 cu130) |
| gsplat | 官方 `28e794ca`(1.6.0),源码在 `~/src/gsplat`(含子模块);`Config.h` 默认通道列表与 `train_rrf.GSPLAT_CHANNELS` 逐项一致,不设 `NUM_CHANNELS`;编译架构 `8.0;8.9+PTX` |
| 装环境 | `sbatch rrf_gsplat/cluster/setup_env.sh`(A30 节点上编译;计算节点可以联网)。作业 599549(fal019):共 69 分钟,其中 gsplat 编译 64 分钟(3DGUT 的 FromWorld 光栅化单个文件就 30 多分钟;容量基准用不到 3DGUT,以后重装可加 `BUILD_3DGUT=0`) |
| 装完的检查 | NVIDIA A30 24 GB,驱动 595.71.05;nvcc 12.8(V12.8.61);gcc 11.5.0;torch `2.9.1+cu128`,`cuda.is_available()` True;gsplat 1.6.0;numpy 2.4.6、scipy 1.17.1、pillow 12.3.0、matplotlib 3.11.2;5 通道 300 × 200 渲染 + 反向:有限、梯度非零,OK |
| 运行时 | 作业里直接用 `~/envs/rf-gsplat/bin/python`,不需要加载模块 |
| 日志 | Slurm 输出在 `output/cluster/logs/`(`output/` 在 `.gitignore` 里) |

### 数据

`~/rf3dgs_data_v1.tar`:sha256 `f730132d…eb881` 校验通过;6411 个条目(6406 个文件 + 5 个目录),全是相对路径,不覆盖任何已有文件,
解出的路径都在 `.gitignore` 里。`spectra_float` 3200 个(= 800 个位置 × 4 面,与 `generation_meta.json` 的 `num_images` 一致)。

## 1. 集群冒烟:主峰方向 loss(2026-09-25,作业 599550)

**做了什么**:`rrf_gsplat/cluster/dir_loss_smoke.sh`,复现本机的 `rounds/win_dir_loss_smoke.sh`(stage2_notes §17),配置完全相同:
`3dgs_APS_60_gp100`、protocol_v1、24 个位置(`--max-train-views 96`,路线两端都在,四个面齐)、power、SH3、`--faces-per-step 4`、
3000 步、seed 0;base / eg / ce 三臂。唯一不同(Ke:多用几张卡):一个作业 3 张 A30,三臂同时跑,每臂独占一张卡、16 核。

**判据**(跑之前写下,cluster_handover.md §5):S1 监测量全程有限;S2 eg / ce 的监测量(最后 300 步)比 base 低 ≥ 0.5 dB;
S3 训练视图 RMSE 不超过 base + 1.0 dB;C1 base 训练视图 RMSE 在 2.21 ± 0.1 dB;C2 三臂监测量在本机 1.84 / 0.83 / 1.29 dB 的 ± 0.3 dB 内;
C3 distinct ≤ 1° 在本机 46.5 / 49.3 / 52.1 % 的 ± 5 个点内。

| 臂 | 监测量,最后 300 步(本机) | 训练视图 RMSE(本机) | distinct ≤ 1°(本机;71 张 distinct) | 主峰方向中位(本机) | 真峰处功率误差(本机) |
| --- | --- | --- | --- | --- | --- |
| base | 1.82 dB(1.84) | 2.21 dB(2.21) | 46.5 %(46.5;33 / 71) | 1.19°(1.19) | −2.18 dB(−2.16) |
| eg | 0.83 dB(0.83) | 2.22 dB(2.22) | 49.3 %(49.3;35 / 71) | 1.07°(1.07) | −0.82 dB(−0.68) |
| ce | 1.29 dB(1.29) | 2.22 dB(2.22) | 52.1 %(52.1;37 / 71) | 0.66°(0.57) | −4.55 dB(−4.56) |

**结论**:S1、S2、S3、C1、C2、C3 **全部通过**。集群环境(官方 gsplat 28e794ca + torch 2.9.1 cu128)复现了本机(gsplat_win + cu130)的冒烟:
distinct 的计数逐个相同,监测量差 ≤ 0.02 dB,RMSE 差 ≤ 0.01 dB。只有 ce 的主峰方向中位不同(0.66° 对 0.57°),不在判据内,记下。
只验通路,不判优劣(1 个 seed、24 个位置)。最终评估用的是验证集(480 张 = val_random 236 + val_segment 244),测试集没有读。

**计时**(按契约):NVIDIA A30,独占(开跑时三张卡利用率 0 %、无其他进程;每臂一张卡;三臂同时跑,共用节点的 CPU 和主机内存带宽)。
分辨率 300 × 200,每步 4 面(一个位置的整圈),排除前 200 步预热,报 50 步窗口的中位数。

| 臂 | 数据生成 | 载入 | 训练(其中实时评估) | 最终评估(480 张验证视图) | 训练拟合诊断(96 张) | train_rrf.py 墙钟 | 合计(墙钟) | 步时中位(56 个窗口) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | MISSING(本机生成) | 7.2 s | 44.9 s(0.4 s) | 0.7 s | 6.8 s | 64.8 s | 71.6 s | 12.81 ms |
| eg | MISSING | 7.2 s | 44.9 s(0.4 s) | 0.7 s | 6.8 s | 64.9 s | 71.7 s | 12.94 ms |
| ce | MISSING | 7.2 s | 44.9 s(0.4 s) | 0.7 s | 6.4 s | 65.0 s | 71.4 s | 12.79 ms |

train_rrf.py 墙钟比"载入 + 训练 + 最终评估"多约 12 s:Python / CUDA 初始化和保存模型,不单列。平均 66.9 it/s(含预热),预热后约 78 it/s
(本机 3060 冒烟约 41 it/s,但那次计时作废:GPU 被占 44 %)。300 × 200 这个尺度下步时由 kernel launch 和 Python 编排支配,不外推为光栅器速度。
**冒烟不能回答**:尾部行为、规模相关的数值稳定性、显存与吞吐。

**两个对照**(仓库 CLAUDE.md;作业 599573):`check_dir_loss.py` 原样运行(`rrf_gsplat/cluster/check_dir_loss_falcon.py` 包一层:
原脚本从本机的 `output/rrf/m3/rr/plain_s0/results.json` 读数据的 dB 范围,那个文件不在数据包里;没有 `--db-range` 时这个范围就是
`generation_meta.json` 的 `spec_min_db / spec_max_db`,包装只替换这一次读取)。期望(跑之前写下):U1 通过;U2 的已知失败一半与解析值
相差 ≤ 0.01 dB(≈ 74.46 dB);U3、U4、U5b、U5c、U6 通过;U2 前半、U5、U7 与本机一样按原判据不过(已知:平滑峰、探索问题、浮点)。

| 检查 | 集群 | 本机 |
| --- | --- | --- |
| U1 解析解(合成) | 预测 = 真值 −0.0000 dB;峰放在低 30 dB 处 30.000 dB;通过 | 0.000 / 30.000,通过 |
| U2 已知失败(四个面转一格) | 74.4608 dB,解析值 74.4608 | 74.4608 |
| U2 前半(预测 = 真值,T = 0.05 dB) | 0.0165 dB(判据 ≤ 0.01,按原判据不过) | 0.0165 |
| U3 / U4 / U6 | 通过(0.00e+00 dB;0.263455 对 0.263455;KL 0 / 61.587) | 通过 |
| U5(不退火) | 梯度 4.7e-7,loss 不动,不过 | 5e-7,不过 |
| U5b(expgain 退火)/ U5c(ce) | 74.461 → −0.000 dB / → 0.016 dB,都找回真峰,通过 | 通过 |
| U7(平坦真值) | L = 1.9e-8(判据"恰好为 0",浮点),梯度有限 | −6.5e-9 |

**结论**:与期望逐项一致,两个对照都对上。冒烟通过,开始第 6 节的容量基准。
