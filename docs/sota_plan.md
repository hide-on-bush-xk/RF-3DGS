# SOTA 对标计划(2026-09-21)

> Ke 的方向:SOTA 优先;指标不一致就加对方的指标,坐标不同就改坐标;对方没有的指标,创建我们占优的指标并对比。
> 本文件是执行清单:每条对比先写"同坐标"是什么、对手的数从哪来、判据是什么,再跑。数字来源一律可重跑。

## 0. 可以站到同一坐标上的对手,和站不上的

| 对手 | 坐标(数据 / 划分 / 指标) | 我们能否上同一坐标 | 依据 |
| --- | --- | --- | --- |
| **RF-3DGS**(发布 checkpoint + 发布数据 + 官方 640 张测试划分 + 作者的 `metrics.py`) | 六种谱各一个模型;PSNR / SSIM / LPIPS(40k 步);另有 20/40/60/80% 训练子集划分 | **能,完全同坐标**:同数据、同划分、同一份 `metrics.py`(逐像素、逐图平均、VGG-LPIPS) | 发布模型的 PSNR/SSIM 已逐位复现(`output/demo/metrics_all.json`) |
| RF-PGS(2508.16849) | 自己生成的数据,多配置平均;PSNR 20.61 / SSIM 0.661 / LPIPS 0.395;RF-3DGS 基线 14.22 | 不能:数据与代码未公开 | literature_check §3 |
| NeRF²(2305.06118,代码 MIT)、WRF-GS/+(2412.04832,代码公开)、GSRF、RxGS、BiWGS | NeRF² 公开基准:RFID 空间谱 / BLE RSSI / MIMO CSI,固定网关、移动 Tx,无视觉几何;PSNR/SSIM/LPIPS(谱)、RSSI MAE(dB)、CSI 误差 | **半能**:(a)把 NeRF²、WRF-GS 的公开代码跑在 RF-3DGS 发布数据上(要改输入:他们是 Tx 动网关不动,RF-3DGS 是 Rx 动);(b)我们的方法跑不了他们的基准——没有视觉几何 | 验证记录续 |
| 城市 Tx 布置(2604.28153,次模贪心) | 城市 3D 地图 RT;均速率 ≈ 2×、边缘速率 2–8× | 不能直接:室外多基站;能借它的**目标(均速率、边缘速率)和贪心基线**到室内 | literature_check_planning §C |

结论:第一优先级是 **RF-3DGS 发布基准上的完全同坐标对比**(Track A),因为它今天就能跑、坐标零争议。第二是把 NeRF²/WRF-GS 的公开代码搬到 RF-3DGS 数据上(Track B)。规划(Track C)与"我们的指标"(Track D)在 A 之后。

## Track A:RF-3DGS 发布基准(同数据、同划分、同 metrics.py)

行(按 `CLAUDE.md` 的基线公平性:as published / as published + 工程修正 两行都要):

| 行 | 训练器 | 目标 | 几何 | 步数 |
| --- | --- | --- | --- | --- |
| A0 RF-3DGS as published | 发布 checkpoint | jet RGB | 冻结 | 40k(作者训的) |
| A1 RF-3DGS,我们用作者 `train.py` 重训 | INRIA 光栅器 | jet RGB | 冻结 | 40k;补 TCBF(作者没发模型) |
| A2 ours-rgb | gsplat | jet RGB | 冻结 | 10k(锚点 15.97)与 40k |
| A3 ours-db | gsplat | 从 PNG 逆 jet 得到的浮点值(发布数据没有浮点谱,范围未知,按 [0,1] 归一化) | 冻结 | 10k |
| A4 ours-rgb + 解冻几何 | gsplat | jet RGB | 解冻 | 10k |
| A5 ours-db + 解冻几何 | gsplat | 逆 jet 浮点 | 解冻 | 10k |

列:六种谱(AoD、CBF、Delay、MPC、MVDR、TCBF)。指标:作者 `metrics.py` 的 PSNR / SSIM / LPIPS(640 张测试图,逐图平均),对每一行都从渲染 PNG 算,不用训练器内部的数。
第二张表:20 / 40 / 60 / 80 / 100% 训练子集(发布的划分文件),MVDR 一列,A0/A1(作者是否发了子集模型?没有则 A1)vs A5。

**可以宣称什么**:若 A5 在六种谱上 PSNR/SSIM/LPIPS 全部优于 A0——"在 RF-3DGS 的发布基准、发布划分、作者的度量下优于发布模型 X dB",并按契约分解:目标改动(A3−A2)、几何解冻(A4−A2)、两者(A5−A2)、光栅器(A2−A1,应 ≈ 0)。RF-PGS 的 20.61 只引不比。

**跑之前写下的判据(smoke,MVDR,500 步,16 张)**:
1. 通路同一性:同一批渲染,作者 `metrics.py` 的 PSNR 与训练器内部 `psnr_rgb` 差 ≤ 0.05 dB(同公式、同图);SSIM 差 ≤ 0.01。
2. 已知答案:发布 checkpoint 的 640 张渲染经 `metrics.py` 得 16.0188 / 0.7309 / 0.3911(已复现前两项;LPIPS 待核);c0b 零步重渲 640 张得 15.97 / 0.727。
3. db-from-jet 目标:500 步 PSNR(jet) > 10 dB 且逐步上升;LPIPS 落在 (0, 1)。
4. 解冻几何在发布数据上 500 步不发散(loss 单调、无 NaN)。
不满足即报告,不调判据。全量:6 谱 × (A1 40k ≈ 18 min + A2–A5 各 ≈ 4 min)≈ 3.5 h,RTX 3060 独占,计时按契约。

## Track B:把 NeRF² / WRF-GS 的公开代码跑到 RF-3DGS 发布数据上

- 输入改造:NeRF² 以网关为"相机"、Tx 位置为条件;RF-3DGS 以 Rx 为相机、Tx 固定。用互易性把 Rx 位姿当条件、针孔面当谱,或把他们的球面谱采样改成我们的四面针孔。工作量:先读两份代码的数据加载器再估;不承诺时间。
- 判据:他们的模型在自己的 RFID 基准上先复现 README 数字(已知答案),再上我们的数据。
- 产出:同一张 Track A 的表多两行(NeRF²、WRF-GS+),同 metrics.py。

## Track C:室内 Tx 布置(借城市那篇的目标与基线)

- 目标:覆盖率之外加**均速率**与**边缘(5%)速率**(Shannon 速率按 SNR 算,噪声与带宽写死并报)。
- 基线:精细网格穷举、随机重启、CMA-ES 或贝叶斯优化、**次模贪心**(多 Tx,K = 1–3);我们的梯度法;KPI 加 RT 评估次数与 wall-clock。
- 场景:大堂 + 走廊两个室内场景,depth 1 与逐接收端 depth 3 复核。
- 材料反演不进这张表(inverse crime)。

## Track D:我们占优的指标,对方也算一遍

- 解码方位 / 天顶 / 时延、拷贝地板、波束 top-k:对 RF-3DGS 发布的 AoD / Delay 模型的输出按同一解码器打分(逆 jet → 教程编码的两通道相除),与我们在**同位姿、同频率、同材料**重生成的 MULTI 场比(`--poses-from`,E4 设置)。
- 深度对 mesh:发布视觉 checkpoint 就是我们用的,已有数。
- 跨视角一致性代价:对发布数据不适用(采样在作者那边),只在重生成数据上报。

## 执行顺序

A(今天起)→ D(A 的渲染就够,零 GPU)→ B(先读代码)→ C。
