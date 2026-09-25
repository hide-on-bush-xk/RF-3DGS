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
