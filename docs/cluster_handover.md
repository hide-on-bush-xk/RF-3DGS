# 集群交接:在 VT ARC Falcon 上继续 RF-3DGS(2026-09-25)

写给 Falcon 上的 Claude Code 和 Ke。本机(Windows,RTX 3060)的会话从这里交给集群:本机以后只跑几分钟的冒烟,
容量基准和全量运行都在 Falcon 上跑。先读完这份文件和仓库根目录的 `CLAUDE.md`,再动手。

## 1. 规矩(本机会话里 Ke 定下的,集群上同样适用)

**Ke 的全局规矩(本机 `~/.claude/CLAUDE.md`,不会跟着 git 过来,照这里执行):**
- Ke 要求的改动做完并验证过(能跑、测试通过)就提交,不用等他说;只暂存与这次改动有关的文件,提交信息写清改了什么、为什么。
- 在当前分支(main)上提交,不要新建分支。
- **提交后自动 push**(Ke,2026-09-25:仓库只有他一个用户)。push 前先 `git pull --rebase`(本机也在 push);rebase 冲突或 push
  被拒就停下告诉 Ke。永远不要 force push。
- 改动没通过验证就不提交,告诉 Ke 哪里坏了。
- **第一次启动时**,把下面这段原样写进集群上的 `~/.claude/CLAUDE.md`(Ke 要求所有 session 都记住;这是本机全局设置的原文),
  这样集群上以后所有项目、所有 session 都照此执行:

```markdown
# Git workflow

When I have asked for a change and you have finished it and verified it (it builds,
compiles, or the tests pass), commit it without waiting to be asked. Stage only the
files that change was about, and write a commit message describing what changed and why.

Commit on the branch that is already checked out. Do not create a new branch for this,
even on `main` — I work directly on `main` in my writing repos.

After each commit, push it to the branch's remote without waiting to be asked: I am the only user of my
repositories (2026-09-25). First `git pull --rebase`, because another machine may have pushed (e.g. my GPU
cluster). If the rebase conflicts or the push is rejected, stop and tell me. Never force-push.

If the change does not build or verify, do not commit it — tell me what broke instead.
```

**仓库 `CLAUDE.md`**:计时契约(全链计时、声明 GPU 与是否独占、排除预热、报中位数)、基线公平性、冒烟优先(判据事先写下,
结果落在范围外就报告、不改判据;分层抽样、边界样本、退化样本、平凡输入的解析解、已知失败样本)。

**本机会话积累的约定(原在本机记忆里):**
- 只用虚拟环境(conda env),不用系统 Python。
- 只在自己的空间操作:`/home/kexu`(及 Ke 自己的 `/projects`、`/scratch` 目录)。不碰别人的目录。
- 登录节点只做轻活;训练与编译用 `sbatch` / `interact`(`salloc`)交给 GPU 节点。
- 协议只用 `rrf_gsplat/protocol_v1`:训练 467 个位置;验证 = val_random(内插)+ val_segment(外推);测试集(test_interp / test_region)
  封存,只有第 4 阶段加 `--final-test` 才能读,读一次就记录一次。
- 主指标是物理的:distinct ≤ 1°(真主峰比 ≥ 3° 外的对手高 ≥ 1 dB 的视图里,预测主峰在 1° 内的比例)和真实信道上的波束增益损失,
  与查表(NN / IDW)比。不用 PSNR 下结论。
- 比较前至少 3 个 seed(容量基准的种子间差异 20–33 %);按"每张视图访问次数"对齐训练量;判收敛看峰指标,不看 loss。
- 数据集用不截峰的 `3dgs_APS_60_gp100`(训练脚本有截峰检查,会拒绝截峰的数据)。
- 新组件训练后要查"还活着"(梯度非零),不是只查第一步。
- 意外的数字先查是不是测量假象,再报告。

**git 上避免和本机冲突:** 本机那边会继续改 `docs/stage2_notes.md` 等文件。集群这边的结果只写进 `docs/cluster_log.md`(新建),
新脚本只放 `rrf_gsplat/cluster/`;不要改 `docs/stage2_notes.md`。两边都自动 push,本机 pull 下来再把结果并进笔记。

## 2. 现在在哪(细节见 `docs/stage2_notes.md` 的 "协议 v1" §11–§17)

- 目标:在新接收位置上,用训练好的场赢过查表(NN / IDW)。至今没赢:验证集 distinct ≤ 1° 约 27–31 %,查表 42 %;
  波束增益损失 4.2–4.9 dB,查表 1.1–1.4 dB(§16)。
- G1(§14):训练位置的发射点已经覆盖新位置的主峰来源(10 cm 内 100 %),**放置不是问题**。
- 关键发现(§16):模型在**自己的训练位置上**也只拟合对约 50 % 的 distinct 主峰,低于"抄最近训练位置的谱"在新位置得到的 63 %。
  **第一瓶颈是在有监督的地方把峰的方向拟合准**;像素 loss 已经平了,峰还是错的。
- 于是做了主峰方向 loss(§17,`rrf_gsplat/dir_loss.py`,`train_rrf.py --dir-loss`),本机冒烟通过:
  - `expgain`:π = softmax(预测 dB / T + log 立体角)下的期望波束增益损失(dB);T 必须从大退火到小,否则梯度消失(`check_dir_loss.py` U5);
  - `ce`:KL(q‖π),q 为真值自己的 softmax,不需要退火;
  - 冒烟(24 个位置、3000 步、1 个 seed):训练位置 distinct ≤ 1° 46.5 %(只监测)→ 49.3 %(expgain)/ 52.1 %(ce),像素拟合不变。只验通路,不判优劣。

## 3. 第一件事:装环境(在 GPU 节点的交互作业里)

下面的模块名、QoS 名是按 ARC 文档推的,**先核实**(`module spider CUDA`、`module spider Miniconda3`、`sacctmgr show assoc user=$USER`)。

```bash
chmod 700 ~                                   # home 只给自己读写(Claude Code 的凭证在 ~/.claude)
cd ~ && git clone https://github.com/hide-on-bush-xk/RF-3DGS.git && cd RF-3DGS
# 交互 GPU 作业(账号 <ACCOUNT> 问 Ke;A30 与本机 3060 同代,结果最好对照)
salloc -A <ACCOUNT> -p a30_normal_q --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=16 --gres-flags=enforce-binding -t 2:00:00
module load Miniconda3
conda create -y -p ~/envs/rf-gsplat python=3.11 && source activate ~/envs/rf-gsplat
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
module load CUDA/<与 torch.version.cuda 同一个 12.x>      # 编译 gsplat 要 nvcc;GCC 版本要 nvcc 支持
pip install numpy scipy pillow matplotlib ninja
MAX_JOBS=16 pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@28e794ca"
```

- gsplat 用官方的 28e794ca(本机用的是它的 Windows 移植版 `gsplat_win`,只多了 Windows 构建和几个默认关闭的环境变量开关)。
  默认编译的通道数列表与 `train_rrf.py` 里的 `GSPLAT_CHANNELS` 一致。容量基准用不到 3DGUT(lidar / 命中距离)。
- 装完的检查(要贴给 Ke):`python -c "import gsplat; print(gsplat.__version__)"`,以及下面第 4 节的冒烟。

## 4. 数据

Ke 从本机传一个打包文件(1.57 GB,6411 个文件,路径是相对仓库根目录的):

```
rf3dgs_data_v1.tar   sha256 f730132d825b47678846521a6aa40872fcde96e0f243f250beb5d228c5eeb881
  RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100/      APS 数据集(不截峰)
  RF-3DGS_dataset/blender_visual_trained/chkpnt30000.pth   视觉 3DGS checkpoint(几何来源)
  output/rrf/beam_maps_val_subset.npz / .json         验证子集(120 张)的实时通信指标用
```

```bash
cd ~/RF-3DGS && echo "f730132d825b47678846521a6aa40872fcde96e0f243f250beb5d228c5eeb881  $HOME/rf3dgs_data_v1.tar" | sha256sum -c \
  && tar -xf ~/rf3dgs_data_v1.tar && ls RF-3DGS_dataset/regenerated/3dgs_APS_60_gp100/spectra_float | wc -l
```

## 5. 第二件事:集群冒烟(复现本机的主峰方向 loss 冒烟)

按 `rrf_gsplat/rounds/win_dir_loss_smoke.sh` 写一个 Linux 版(放 `rrf_gsplat/cluster/`,路径用仓库相对路径、解释器用 conda 环境),
配置完全相同(24 个位置 = `--max-train-views 96`,3000 步,seed 0,base / eg / ce 三臂),作为一个 sbatch 作业跑。

判据(事先写下,2026-09-25):
- 本机冒烟的三条判据照样通过(S1 有限、S2 eg / ce 的监测量比 base 低 ≥ 0.5 dB、S3 训练视图 RMSE 不超过 base + 1.0 dB);
- 与本机数值一致(不同 GPU 不会逐位相同):base 的训练视图 RMSE 在 2.21 ± 0.1 dB 内;三臂监测量(最后 300 步)在本机
  1.84 / 0.83 / 1.29 dB 的 ± 0.3 dB 内;distinct ≤ 1° 在本机 46.5 / 49.3 / 52.1 % 的 ± 5 个点内;
- 落在范围外 = 报告并查原因,不改判据,不往下跑全量。

计时:集群作业独占 GPU,计时有效,按契约报(GPU 型号、独占、分辨率 300 × 200、每步 4 面、排除预热、中位数)。

## 6. 第三件事:主峰方向 loss 的容量基准(冒烟通过后)

配置同 rounds 51–56 的重跑(`rrf_gsplat/rounds/win_rerun_r51_56.sh`):160 个训练位置(`--max-train-views 640`)、power、SH3、
`--faces-per-step 4 --lr-scale 2 --sh-backend gsplat --eval-group`、最多每张视图 250 次(`--visits-per-view 250`)、
按训练子集早停(`--early-stop-on train --early-stop-patience 10 --early-stop-min-steps 5000`)、`--save-renders 0`,
每个 run 之后 `diag_train_fit.py --run <dir> --train-subset 640`,主指标 distinct ≤ 1°。

臂(每臂 seed 0 / 1 / 2,一个 run 一个作业):
- base:`--dir-loss expgain --dir-weight 0`(只监测;在集群上重跑,同一硬件条件下比较)
- eg:`--dir-loss expgain --dir-weight 0.1 --dir-temp-start-db 20 --dir-temp-db 1 --dir-anneal-steps 1500 --dir-start 200`
- eg3:同上,`--dir-weight 0.3`
- ce:`--dir-loss ce --dir-weight 0.001 --dir-temp-db 1 --dir-start 200`
- ce3:同上,`--dir-weight 0.003`

读法(事先写下,2026-09-25):
- 一个臂"有帮助":3 个 seed 的均值比 base 高出的幅度超过两臂各自的种子范围(max − min);否则"没有差别"(与 §12 同一规则)。
- 目标线:训练位置 distinct ≤ 1° ≥ 63 %(查表在内插段的水平)。达到或有臂明显逼近,再上验证集
  (全部 467 个训练位置、按验证子集早停、与查表比,配置见 `rrf_gsplat/rounds/win_g1_g2.sh`),那一步先回来和 Ke 商量。
- 同时报像素拟合(训练视图 RMSE)和真峰处功率误差,看方向变准是否以电平为代价(本机冒烟里 ce 的真峰功率偏低 4.6 dB)。

## 7. 回报

每做完一步写进 `docs/cluster_log.md`:做了什么、判据、结果表(每个 seed 的数 + 均值 [范围])、按判据的结论、计时(按契约)、
遇到的坑。提交并 push,然后告诉 Ke。环境信息(模块名、torch / gsplat / CUDA 版本、GPU 型号、账号与分区)也写进去,
以后的作业照着用。
