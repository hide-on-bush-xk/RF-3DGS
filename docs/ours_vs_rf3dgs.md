# 我们的方案相对 RF-3DGS 改了什么

> 写给 Ke 自己看的技术账本:代码层、数学层、feature 与性能对比,以及尚未做和值得想的地方。
> 所有数字都来自本仓库里可以重跑的脚本(路径都给了),没有来自论文以外的估计。
> VS Code 里 `Ctrl+Shift+V` 打开预览可以渲染公式。

---

## 0. 一页总结

RF-3DGS 的本质是:**把 Sionna 算出来的"射频图片"(角功率谱的 jet 伪彩图)当成普通照片,用 INRIA 原版 3DGS 的渲染器和 L1+SSIM 损失,在冻结的视觉几何上重新拟合每个高斯的颜色和不透明度。** 它回答的问题是"Tx 固定时,任意 Rx 位姿看到的谱图长什么样",输出是 8-bit 图像,评价指标是 PSNR/SSIM。

我们在三个层面动了它:

| 层 | RF-3DGS | 我们 | 一句话 |
| --- | --- | --- | --- |
| **生成端** | Sionna 0.19 + TF,CPU,逐路径 Python 循环,8-bit PNG 唯一输出,两种谱按图归一化 | Sionna 2.1 + PyTorch + Dr.Jit/OptiX GPU,`index_add_` 一次散射 30 万条路径,float 谱 + PNG,六种谱统一全局归一化 | 同一 pipeline,快约两个数量级,并修掉了两个会污染训练目标的缺陷 |
| **评价端** | PSNR / SSIM / LPIPS(图像域) | 信道域指标(K 因子、RMS 时延扩展、相干带宽、角扩展、相干增益、容量、CFR)+ 消融 + 对比表 dashboard | 用信道说话而不是用图片说话 |
| **规划端(新)** | 不存在;Tx 动了要重跑整套 Sionna + 重训 | 可微 RT:Tx 布设梯度优化、材料反演、主动测量选点;Aerial 风格结构化 CIR 输出 | 从"渲染 Rx 视角"变成"规划 Tx 位置并学习孪生" |

**还没做的**(第 5 节详述):RRF 本身的训练还在 INRIA 原版光栅器上,gsplat 环境建好但没接入;规划端的材料反演用的是合成真值,不是实测;所有实验只在 NIST 大厅、60 GHz、depth 1。

---

## 1. RF-3DGS 在代码层到底是什么

仓库是 INRIA 3DGS 的一个 fork(基于 commit `b17ded92`)。**射频相关的改动只有两处**,光栅器一行没动:

### 1.1 `train.py`:从视觉 checkpoint 出发,只学颜色和不透明度

[train.py:44-52](../train.py#L44-L52)

```python
# RF_3dgs_retraining
gaussians._xyz.requires_grad = False       # 位置冻结
gaussians._scaling.requires_grad = False   # 尺度冻结
gaussians._rotation.requires_grad = False  # 朝向冻结
gaussians._opacity.requires_grad = True    # 不透明度重新学
gaussians._features_dc.data.fill_(0.0)     # SH 0 阶清零
gaussians._features_rest.data.fill_(0.0)   # SH 高阶清零
```

也就是说:先用 Blender 渲染的照片按原版 3DGS 训练 30k 步得到几何(`_xyz, _scaling, _rotation`),然后加载这个 checkpoint,把每个高斯的颜色(球谐系数)清零,**只训练 SH 和 opacity** 去拟合射频谱图,再跑 10k 步(30k → 40k)。这就是论文里"3 min 训练"的含义——它不包括视觉 3DGS 的 30k 步,也不包括 Sionna 生成数据集的时间。"2 ms 渲染"是光栅化一张 300×200 图的时间,也就是原版 3DGS 的渲染速度。

损失函数是原版的,[train.py:103-105](../train.py#L103-L105):

$$\mathcal{L} = (1-\lambda)\,\|I - I_{gt}\|_1 + \lambda\,(1 - \mathrm{SSIM}(I, I_{gt})),\quad \lambda = 0.2$$

其中 $I_{gt}$ 是 jet 伪彩的 **RGB 图**。射频信息(dB 值)在进入训练之前已经被 colormap 量化成 8-bit 三通道了。

### 1.2 `scene/dataset_readers.py`:按索引文件划分训练/测试

[dataset_readers.py:195-211](../scene/dataset_readers.py#L195-L211) 读 `train_index.txt` / `test_index.txt`。仅此而已。

### 1.3 光栅器:原封不动

[config.h:15](../submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h#L15) 仍是 `#define NUM_CHANNELS 3 // Default 3, RGB`。谱图是 3 通道 RGB,不是 1 通道 dB,也不是 N 通道(比如每个时延 bin 一通道)。

### 1.4 数据集:Sionna 0.19 教程

Google Drive 上的 `RF-3DGS-tutorial.ipynb`:`scene.compute_paths(max_depth=1, diffraction=False, scattering=True, scat_keep_prob=0.1, num_samples=1e6)`,每个 Rx 位置 4 个 yaw(0/90/180/270°)的 90° 针孔视图,300×200,六种谱(AoD / CBF / Delay / MPC / MVDR / TCBF),800 个位置 × 4 = 3200 张图。

---

## 2. 数学原理:RF-3DGS 把什么当成了什么

### 2.1 3DGS 渲染方程

对像素 $u$,沿视线把高斯按深度排序后 alpha 合成:

$$C(u) = \sum_{i} c_i(\mathbf{d})\,\alpha_i(u) \prod_{j<i}\bigl(1-\alpha_j(u)\bigr),\qquad \alpha_i(u) = o_i \exp\!\Bigl(-\tfrac12 (u-\mu_i')^\top \Sigma_i'^{-1}(u-\mu_i')\Bigr)$$

$c_i(\mathbf{d})$ 是用球谐系数表示的、随观察方向 $\mathbf{d}$ 变化的颜色,$o_i$ 是不透明度,$\mu_i', \Sigma_i'$ 是 3D 高斯投影到像平面后的 2D 均值和协方差。

### 2.2 RRF 的隐含假设

RF-3DGS 的核心类比:**Rx 处的角功率谱 $P(\theta,\phi)$ 是"从 Rx 看向各个方向的亮度"**,和相机看到的辐射亮度是同一种数学对象——都是位置和方向的函数 $L(\mathbf{x}, \mathbf{d})$。于是:

- 每个高斯变成一个"射频发光体",SH 系数 $c_i(\mathbf{d})$ 编码"从方向 $\mathbf{d}$ 看这个高斯时它贡献的谱值";
- alpha 合成的遮挡项 $\prod(1-\alpha_j)$ 近似射频的阻挡;
- 冻结的几何保证了"哪里有物体"来自视觉,只有"物体在射频里多亮"是新学的。

这个类比成立的前提有三条,每一条都是可以攻击的点:

1. **光度一致性**:同一个高斯从不同视角看应该给出一致的值(SH 是低阶光滑的)。射频谱的镜面反射峰是高度视角相关的,SH 3 阶(16 个系数)未必表达得了。
2. **每张图的取值尺度一致**:多视角拟合的前提。教程里 CBF/TCBF 是**逐图归一化**的(见 3.1),这条被打破了。
3. **谱是可加的、可遮挡的**:CBF 谱 $|\mathbf{a}^H \mathbf{R}\mathbf{a}|$ 对路径不是线性可加的(有交叉项),MVDR 更不是;高斯 alpha 合成是线性可加的。

它天然做不到的:**相位**(谱是功率)、**频率选择性**(一张图一个载频)、**Tx 移动**(Tx 是场景的一部分,动了就是新场景)。

### 2.3 六种谱的定义(生成端的数学)

Sionna 给出每条路径 $k$ 的复增益 $a_k$、时延 $\tau_k$、到达角 $(\theta_k^r,\phi_k^r)$、离开角 $(\theta_k^t,\phi_k^t)$。$M\times M$ 平面阵列的导向矢量 $\mathbf{a}(\theta,\phi)\in\mathbb{C}^{M^2}$。把路径按时延分到时间栅格得到 $\mathbf{h}[n] = \sum_{k:\tau_k\in \text{bin } n} a_k\,\mathbf{a}(\theta_k^r,\phi_k^r)$,协方差 $\mathbf{R} = \sum_n \mathbf{h}[n]\mathbf{h}[n]^H$。

| 谱 | 定义 | 备注 |
| --- | --- | --- |
| CBF | $P(\theta,\phi) = \mathbf{a}^H \mathbf{R}\,\mathbf{a}$ | 用裸导向矢量 |
| TCBF | 同上,$\mathbf{a}$ 乘 Hann 窗 | 旁瓣低、主瓣宽 |
| MVDR | $P = 1/(\mathbf{a}^H \mathbf{R}^{-1}\mathbf{a})$ | 需要 $\mathbf{R}$ 满秩,即时延 bin 数 ≥ $M^2$;用带 TR 38.901 单元增益的流形矢量 |
| MPC | 每条路径按 $(\theta_k^r,\phi_k^r)$ 以功率 $|a_k|^2$ 高斯核散射到等距柱状网格 | 不做波束形成 |
| Delay | 同上,颜色取归一化时延 | |
| AoD | 同上,颜色取离开角 | |

然后把等距柱状网格按针孔模型重采样到 300×200,四个 yaw。**这三个投影谱在教程里是 Python 逐路径循环**,30 万条路径要几分钟;我们用一次 `index_add_`([rf_spectra.py](../sionna_port/rf_spectra.py) `equirect_splat`)。

一个所有版本共有的物理限制:阵列在 y-z 平面,导向矢量只依赖 $(\sin\theta\sin\phi, \cos\theta)$,**分不清 $\phi$ 和 $180°-\phi$**。每个视图只有 ±45° 方位角,所以镜像峰落在相邻视图里当"鬼影",压制它靠单元方向图的前后比——而 CBF 用裸导向矢量、MVDR 用带单元增益的流形矢量,这个不对称是教程带来的。

---

## 3. 我们改了什么(按层)

### 3.1 生成端:`sionna_port/`

**平台**:Sionna 0.19/TensorFlow → Sionna 2.1/PyTorch,Dr.Jit + Mitsuba 3,Windows 上 OptiX GPU 光线追踪。`PathSolver`、`paths.cir(out_type="torch")`、`Paths.a/tau/theta_*/phi_*` 全程 torch;完整 API 映射表在 [sionna_port/README.md](../sionna_port/README.md#api-mapping)。

**修掉的第一个缺陷——路径数校准**([calibrate_paths.py](../sionna_port/calibrate_paths.py))。Sionna 的 ITU 材料默认 `scattering_coefficient = 0`,所以 `diffuse_reflection=True` 什么都不产生,depth 1 只有 **8 条**路径——那不是谱,是阵列的点扩散函数。`samples_per_src` 从 10 万调到 1000 万路径数一条不变。

| scattering_coefficient | depth 1 | depth 2 | depth 3 |
| --- | --- | --- | --- |
| 0.0(Sionna 默认) | 8 | 20 | 43 |
| 0.3 | 57,619 | 94,220 | 130,269 |
| 0.5 | 159,684 | 262,935 | 363,155 |
| **0.7** | **312,683** | 522,365 | 722,643 |

0.7 复现论文的"> 300,000 MPC / Tx-Rx 对"。**更正(2026-09-18)**:教程 notebook 的 cell 6 其实逐材料定义了散射系数(0.1–0.8,全局系数 4)、`DirectivePattern` 方向图和 EM 参数,并把 `itu_*` 按名字换成 `custom_*`;见 [stage2_notes.md](stage2_notes.md)。我们的统一 0.7 是等效校准值,不是作者的设定;逐材料版本待接入。

**修掉的第二个缺陷——归一化不一致**。教程里 MVDR/AoD/Delay/MPC 用两个手选探针位置建立全局 dB 范围;CBF/TCBF 走 `jet_colormap_convert`,**逐图按自身 min/max 缩放**(`per_view_normalization` 参数被接收但从未读取)。逐图归一化破坏了 2.2 的假设 2。论文里 CBF/TCBF 恰好是最差的两类并归因于干扰——这是一个可检验的混淆(confound):用全局范围重生成 CBF 再训一次。我们的生成器对六种谱用**同一种**从数据来的全局范围。

**保留 float 谱**:`spectra_float/NNNNN.npy` 与 PNG 并排,`generation_meta.json` 记录 colormap 的 dB 范围,量化可逆。消融数据(`output/ablation.json`,s=0.7 d=1)说明为什么这重要:全局跨度 72.5 dB,8-bit 步长 0.28 dB,量化 RMSE 0.08 dB——这部分不大;但**每张图平均只用了 8-bit 范围的 47%**(逐图跨度均值 34 dB),也就是说全局归一化下有一半的量化分辨率是浪费的,而逐图归一化又破坏一致性。这是"float 域损失 + 多通道光栅器"的直接动机(第 6 节)。

**其它**:导向矢量/流形只依赖相机模型,`ArrayGrid.build` 算一次而不是 3200 次;MVDR 对奇异协方差显式拒绝(`torch.linalg.inv` 遇到秩亏返回 inf/nan 不报错),提供对角加载;`v_tr38901_pattern` 在 2.x 返回单个 `Complex2f` 而非 0.19 的 pair,按 pair 解包会静默取实部虚部。

**移动晶格(moving lattice)**:`per_view_seed` 每个视图换一次求解器种子。低采样数留在谱里的采样结构,如果种子固定,在每个视图里一模一样——这是多视角拟合唯一平均不掉的误差,因为它看起来像场的真实特征。[test_sampling_accumulation.py](../sionna_port/test_sampling_accumulation.py) 对 400 万采样参考、每视图 5 万采样:

| 平均视图数 | 晶格移动 | 晶格固定 |
| --- | --- | --- |
| 1 | 1.690 dB | 1.691 dB |
| 4 | 1.577 dB | 1.691 dB |
| 8 | **1.472 dB** | 1.691 dB |

固定列纹丝不动——机制得到确认。但移动只带来 1.15×,独立噪声应给 $\sqrt{8}=2.83\times$:**1.69 dB 里大部分是偏差(bias),不是方差**——低采样系统性地漏掉弱路径,换种子救不了。这个技巧买回了方差那一半,偏差那一半还得花采样数。

**吞吐**:RTX 3060,20 个位置(80 视图,MVDR,每视图 31 万路径)8 秒,约 10 视图/秒,800 位置全集约 **5 分钟**。单次 path solve 只要 27 ms,时间花在谱计算上。

**验证**:`test_rf_spectra.py` 六项检查通过(角网格、导向矢量模、时延分箱、两种波束形成器对合成单路径的峰值);`smoke_sionna.py` 对真实场景通过;投影谱(MPC/Delay/AoD)与发布的真值在结构上几乎逐像素对齐(柱子和墙的轮廓吻合);MVDR 定性一致但**尚未数值上可比**——归一化范围、`time_interval_ns`、单元方向图、`synthetic_array` 还需逐一对齐。

### 3.2 评价端:信道指标、消融、dashboard

**为什么 PSNR 不够**:它看的是伪彩图,看不见时延扩展、相干带宽、K 因子、阵列增益——这些才是模型号称要表示的信道性质。[rf_metrics.py](../sionna_port/rf_metrics.py) 从求解出的路径直接算:

$$\bar\tau = \frac{\sum_k |a_k|^2 \tau_k}{\sum_k |a_k|^2},\qquad \sigma_\tau = \sqrt{\frac{\sum_k |a_k|^2 (\tau_k-\bar\tau)^2}{\sum_k |a_k|^2}},\qquad B_c \approx \frac{1}{5\sigma_\tau}$$

$$K = \frac{\max_k |a_k|^2}{\sum_k |a_k|^2 - \max_k |a_k|^2},\qquad G_{coh} = 10\log_{10}\frac{|\sum_k a_k|^2}{\sum_k |a_k|^2}$$

外加方位/天顶角扩展、SNR、容量,以及 2.x 新增的 `paths.cfr()` 频率响应。(最早版本的"阵列增益"是个恒等式,永远 20 dB,已换成上面的相干增益。)

消融(`run_ablation.py`,s ∈ {0, 0.3, 0.5, 0.7} × depth ∈ {1,2,3})的核心读数:**散射系数是唯一重要的旋钮**。s 从 0 到 0.7,K 因子从 +6.3 dB 落到 −2.2 dB(从 LoS 主导变成富散射),RMS 时延扩展从 7.0 ns 升到 8.5 ns,相干带宽约 28 MHz;depth 从 1 到 3 在 s=0.7 下路径数翻倍但 K 因子只从 −2.2 到 +1.8。

**复现发布的模型**(`output/demo/quick_metrics.py`,640 张测试图):

| 谱 | PSNR | SSIM | 论文 PSNR | 论文 SSIM |
| --- | --- | --- | --- | --- |
| AoD | 19.34 | 0.681 | 19.34 | 0.681 |
| CBF | 14.13 | 0.723 | 14.13 | 0.723 |
| Delay | 17.73 | 0.691 | 17.73 | 0.691 |
| MPC | 15.51 | 0.628 | 15.51 | 0.628 |
| MVDR | 16.02 | 0.731 | 16.02 | 0.731 |

逐位一致,说明环境(CUDA 13.4 + VS 2026 + torch 2.9)和数据读取是对的。

**dashboard**(`output/dashboard.html`):交叉表——行 = 真值 / RF-3DGS 模型输出 / 我们,列 = optical + 六种谱,同一位姿;每格标注参数;下方消融图、指标表、CFR、以及规划面板(3.3)。

### 3.3 规划端:`tx_planning/`(RF-3DGS 里不存在的东西)

出发点:RF-3DGS 的 Tx 是场景的一部分,Tx 动一下就要重跑 Sionna + 重训。室内规划要问的恰恰是"Tx 放哪"。Sionna 2.x 是可微的,我们直接用它的梯度,并借用 Aerial Omniverse Digital Twin 的数据模型(静态 RU × 移动 UE 网格,输出结构化 CIR 而不是图片)。

**阶段 0:可微性验证**([probe_differentiability.py](../tx_planning/probe_differentiability.py))。$\partial P_{rx}/\partial \mathbf{p}_{tx}$ 与有限差分在 5% 内一致;$\partial P_{rx}/\partial s$(材料散射系数)同号同量级(有限差分在随机漫散射采样器上本身就噪)。需要两个文档里没有的开关:`solver.loop_mode = "evaluated"`,`dr.set_flag(dr.JitFlag.SpillToSharedMemory, False)`。

**阶段 1:暴力真值**([tx_sweep.py](../tx_planning/tx_sweep.py))。候选 Tx × 室内 Rx 网格,每对存复增益/时延/AoD/AoA(Aerial `CIRResultsRequest` 的内容)。两个工程发现:

- 室内掩码:向上打到天花板 **且** 向下打到地板。包围盒网格一半在楼外(零路径);4 m 冒烟网格上这个判据与"至少一条路径"逐点一致。
- **把所有 Rx 放进同一次 solve**:Sionna 的开销按源计,不按接收机计。与逐对求解相差 0.0 dB,16 个 Rx 时 ~12 Tx/s 对 0.9 pairs/s,约 200×。**33 个候选 Tx × 239 个室内 Rx 点共 7 秒**。

**阶段 2:梯度布设**([optimize_tx.py](../tx_planning/optimize_tx.py))。目标是软覆盖率:

$$J(\mathbf{p}_{tx}) = \frac{1}{N}\sum_{i=1}^{N} \sigma\!\left(\frac{P_i(\mathbf{p}_{tx}) - T}{w}\right),\qquad P_i = 10\log_{10}\Bigl(\sum_k |a_{ik}|^2 + n_0\Bigr)$$

$T=-85$ dB,$w=5$ dB,$n_0$ 是 −130 dB 的底噪(让够不到的点付有限代价而不是 −300 dB)。梯度 $\partial J/\partial \mathbf{p}_{tx}$ 从 Sionna 反传,Adam 上升。三个发现:

- **蒙特卡洛噪声不是问题**:固定 Tx 时 $J$ 跨种子只波动 ±0.0007(2 万采样)。优化中 0.04 的跳变是真的:60 GHz 下 Tx 挪 0.3 m 就翻转几个点的 LoS,**景观是分段光滑的**。所以正确用法是 sweep 找盆地、梯度精修。
- **梯度不知道墙**:无约束时把 Tx 推穿了 x = 8.48 m 的东墙(墙外零路径 → drjit 空轴归约崩溃)。现在每步要求室内且离最近表面 ≥ 0.2 m,被挡时先去掉墙法向分量沿墙滑。
- 结果:从 sweep 最优格点起步,2 m 网格 59 点,25 步 27 秒,>−85 dB 硬覆盖 **45.8% → 59.3%**。1 m 网格 239 点:46% → 49%,但软目标略降(0.4236 → 0.4195)——两者在这个分辨率上不完全一致,dashboard 上两条曲线都放了。
- 40 万采样、59 个 Rx 时求解出错(单点 std 达 14 dB),保持 ≤ 10 万。

**阶段 3:材料反演**([fit_materials.py](../tx_planning/fit_materials.py))。29 种在用材料各藏一个 $s_n^\star \in [0.2, 0.9]$ 的真值;从 Tx A 在 59 个室内点"测"时延分箱 PDP(16 × 10 ns,+1 dB 高斯噪声);全部从 0.5 起步;到从未见过的 Tx B 打分。损失:

$$\mathcal{L}(\mathbf{s}) = \frac{1}{B N}\sum_{b=1}^{B}\sum_{i=1}^{N}\Bigl(\mathrm{PDP}_{ib}(\mathbf{s}) - \widehat{\mathrm{PDP}}_{ib}\Bigr)^2,\qquad \mathrm{PDP}_{ib} = 10\log_{10}\Bigl(\sum_{k:\tau_{ik}\in b}|a_{ik}(\mathbf{s})|^2 + n_0\Bigr)$$

时延不依赖材料,所以分箱掩码是常数,箱内求和保持可微。

| | Tx A RMSE | 留出 Tx B RMSE |
| --- | --- | --- |
| 全部材料 0.5 | 1.53 dB | 1.09 dB |
| 从 Tx A 拟合后 | 0.98 dB(= 1 dB 噪声底) | **0.47 dB** |

三个发现:

- **总功率是错误的观测量**。用总功率也能把 Tx B 压到 0.44 dB,但材料估计是胡来的(0.80 的玻璃拟成 0.01,0.89 的木头拟成 0.12):镜面/漫散射的分配几乎不改总功率,但改 PDP 的"尖峰 vs 拖尾"。
- **Adam 是错误的优化器**。它让每个参数等速移动,数据约束不到的材料随机漂。改成归一化梯度下降 $\Delta s_n = -\eta\, g_n / \max_m |g_m|$ 后,梯度 ≥ 最大值 2% 的 7 种材料平均误差 **0.161 → 0.025**(混凝土 0.71→0.74,金属 0.22→0.21,玻璃 0.80→0.85),其余 22 种留在先验附近。也试了 ε 绑到最大梯度的 Adam 变体(`adam-rel`):不漂,但在噪声梯度上过冲(0.051),保留为选项。
- **可辨识性本身是产出**。12 种材料从 Tx A 收到的梯度恰为零(depth 1 下根本没打到)。

**阶段 3b:主动测量**([active_measurement.py](../tx_planning/active_measurement.py))。测量之前,孪生就能说候选位置 C 会揭示什么——用 Hutchinson 估计每种材料的灵敏度:

$$\mathbb{E}_{\mathbf{w}\sim\mathcal N(0,I)}\bigl[(\mathbf{J}_n^\top \mathbf{w})^2\bigr] = \|\mathbf{J}_n\|^2,\qquad \mathbf{J}_n = \frac{\partial\,\mathrm{PDP}(C)}{\partial s_n}$$

4 个随机投影、4 次反传,每个候选约 1 s。打分 $\sum_n \log\bigl(1 + \|\mathbf{J}_n(C)\|^2 / (\|\mathbf{J}_n(A)\|^2 + \epsilon)\bigr)$:A 没看到的材料权重大,A 已钉死的饱和。

| 拟合数据来源 | 3 个留出 Tx 平均 RMSE | 可辨识材料 |
| --- | --- | --- |
| 先验(全 0.5) | 1.24 dB | 0 |
| 仅 A | 0.30 dB | 7 |
| A + 中位分候选 (8, −3) | 0.25 dB | 7 |
| **A + 算法选的 (0, 1)** | **0.17 dB** | 8 |

算法选的位置把留出误差减半,随便选的减 1/6。这是规划器的闭环:拟合 → 问孪生哪里是盲的 → 去那里测。RadioSight 的逐 AP 孪生给不出这个问题的答案。

### 3.4 工具链(不算科研贡献,但决定了什么能跑)

- CUDA 13.4(conda-forge,无管理员)+ VS 2026 MSVC 14.51:`NVCC_APPEND_FLAGS=-Xcompiler /Zc:preprocessor`;CUDA 12.8 在这套编译器下 `cudafe++` 崩溃。`submodules/*/setup.py` 加了 Windows 守卫。
- `torch.load(weights_only=False)`:torch ≥ 2.6 默认拒绝 checkpoint 里的 numpy 标量。
- gsplat 1.6.0 在 VS 2026 下编不过(`std::isfinite` 设备端),在 WSL 里从 main(`28e794ca`)构建成功;Sionna 2.1 在 WSL 的 CPU 后端因 Dr.Jit 1.5 捆绑的 LLVM 15 不能降低 `fmaximum` 而崩溃,WSL 里也没有 OptiX。**结论:Sionna 在 Windows 跑,gsplat 在 WSL 跑**。四个环境:`rf-3dgs`(py3.10,原版训练)、`rf-sionna-win`(py3.12,Sionna 2.1 GPU)、WSL `rf-gsplat` / `rf-sionna2`。
- 场景文件:Blender 的 `.001` 后缀让 Sionna 查不到 ITU 材料;`custom_*` 不是 ITU 材料;plywood/brick 在 60 GHz 无定义;网格文件名 UTF-8 被当 CP437 解码;Windows 上 Mitsuba 不接受非 ASCII 路径;shape id 含 `.` 被渲染器拒绝。`fix_scene_xml.py` + `ascii_meshes.py` 处理。

---

## 4. Feature 与性能对比表

| 能力 | RF-3DGS | 我们 | 数字 |
| --- | --- | --- | --- |
| 射线追踪后端 | Sionna 0.19 / TF / CPU | Sionna 2.1 / PyTorch / Dr.Jit OptiX GPU | 单次 solve 27 ms;80 视图 8 s |
| 数据集生成 | 教程逐路径 Python 循环 | `index_add_` 一次散射 | 800 位置 ≈ 5 min |
| 路径数 | 论文 > 300k;pipeline 默认实际 8 | 校准 s=0.7 | 312,683 @ depth 1 |
| 归一化 | 四种全局两探针,两种逐图 | 六种统一全局、从数据来 | **已验证**:逐图归一化让 MVDR 掉 6.2 dB PSNR、CBF 的 dB 误差 +3.5 dB |
| 训练目标精度 | 8-bit jet PNG | float `.npy` + PNG + 范围元数据 | 逐图仅用 47% 的 8-bit 范围 |
| 谱种类 | 6 | 6(全部移植,同一相机) | 投影谱与真值逐像素对齐 |
| 采样策略 | 固定种子 | 逐视图换种子 | 8 视图 1.69→1.47 dB;偏差仍在 |
| 评价 | PSNR / SSIM / LPIPS | + K 因子、$\sigma_\tau$、$B_c$、角扩展、相干增益、容量、CFR | 消融:s 是唯一重要旋钮 |
| Tx 移动 | 重跑 Sionna + 重训 | 可微 RT,直接对 $\mathbf{p}_{tx}$ 求梯度 | 梯度 vs 有限差分 5% 内 |
| 多 Tx 评估 | 无 | 批量 Rx 单次 solve | 33 Tx × 239 Rx = 7 s;200× |
| Tx 布设 | 无 | sweep + 带墙约束的梯度精修 | 覆盖 45.8% → 59.3% |
| 材料学习 | 无(材料来自未公开描述符) | PDP 反演,留出 Tx 验证 | 1.09 → 0.47 dB;可辨识材料误差 0.025 |
| 下一步测哪 | 无 | 梯度灵敏度选点 | 0.30 → 0.17 dB(随便选 0.25) |
| 输出形态 | 图片 | 图片 + 结构化 CIR(npz)+ dashboard | Aerial 风格 |
| RRF 训练本身 | INRIA 光栅器,RGB 伪彩 | gsplat,单通道 dB 目标 | 复现 15.97 vs 16.02;dB 目标 RMSE −11%,到同质量快 3.6× |
| Tx 移动后重建 | ≈1 h 数据(CPU)+ 54 s 微调 | **实测 65 s**:160 个位置 33 s 生成 + 32 s 微调到 PSNR 17(热启动 56 s) | 终值比 800 位置低 0.4 dB;rgb 热启动反而有害 |
| 几何 | 冻结(视觉几何 = 射频几何) | 可解冻;适配后的几何跨 Tx 可沿用 | 解冻 RMSE −27%;跨 Tx 沿用 −5%,位置只动毫米级 |
| 输出解码 | 无(只有图片) | 多通道:功率 / AoD 方位 / AoD 天顶 / 时延 | 19.0° / 11.6° / 8.2 ns;教程编码 48.6° / 56.6° |

---

## 5. 诚实的边界:还没做、还不能说

1. ~~RRF 的训练一行没改~~ **已做(阶段 2)**:`rrf_gsplat/` 在 gsplat 上复现了微调并做了 color function / 归一化 / 消融 / Tx 移动实验,见第 7 节。2DGS、MCMC 致密化仍未用。
2. **重生成的 MVDR 谱与发布的谱还不能数值对比**:范围、`time_interval_ns`、单元方向图、`synthetic_array` 待逐一对齐;发布的谱峰更尖。
3. **材料反演用的是合成真值**(同一场景藏一组系数),不是实测。真实的失配来源(几何误差、天线方向图、频率相关 EM 参数、depth > 1 的高阶交互)一个都没进来。
4. **所有规划实验:一个场景、60 GHz、depth 1、单极化各向同性天线、2 m/1 m 网格、25–30 步。** 2 m 网格上的 59.3% 与 1 m 网格上的 49% 不是同一个数,阈值 −85 dB 是拍的。
5. **`custom_*` 材料是占位映射**(plastic→chipboard,leather→wood,cloth→ceiling_board),不是作者的值——作者的值在教程 cell 6 里(εr 2.3 / 1.8 / 1.8,σ = 0,散射 0.2 / 0.4 / 0.8),尚未接入。
6. ~~CBF/TCBF 归一化混淆是假设~~ **已验证**:同一份浮点谱,逐图归一化比全局归一化在 MVDR 上低 6.2 dB PSNR,在 CBF 上 dB 误差高 3.5 dB(第 7 节)。
7. 平面阵列的前后向模糊在所有版本里都存在;单元方向图在 CBF 与 MVDR 里用法不对称,继承自教程,未改。

---

## 6. 供你思考的改进方向(问题,不是结论)

按"离现有代码有多远"排序。

**近(生成端 → 训练端,把已有的东西用起来)**

- **float 域损失 + 多通道光栅器**。gsplat 支持任意通道数:训练目标可以是 dB 浮点单通道,或者每个时延 bin 一通道(把 PDP 直接当"颜色")。这直接回应 2.2 假设 2 和 3.1 的量化浪费。要回答的问题:SH 的低阶光滑性对 dB 谱够不够?需不需要换成方向编码 MLP?
- **决定性实验:归一化混淆**。用我们的全局范围 CBF 数据重训 RF-3DGS,和发布的 CBF PSNR 比。一天的工作量,直接检验论文的一个解释。
- **2DGS 几何 → Sionna 网格**。RF-3DGS 从照片得到的几何只用于 splat;2DGS 能导出网格,网格可以直接喂 Sionna 做射线追踪。这是"从真实照片到可微射频孪生"的路,也是阶段 4。要回答:重建网格上的路径数、K 因子、时延扩展和 Blender 真值差多少?

**中(改 RRF 的表示)**

- **Tx 位置作为条件**。RF-3DGS 的高斯颜色是 $c_i(\mathbf{d})$;能否变成 $c_i(\mathbf{d}, \mathbf{p}_{tx})$?这让 RRF 从"一个 Tx 的场"变成"任意 Tx 的场",规划端就可以用 RRF 而不是 Sionna 做前向——速度从秒级到毫秒级。代价是训练集要覆盖多个 Tx,而我们的 batched sweep 恰好 7 秒出一个 Tx。
- **物理可加的谱作为目标**。MPC/Delay/AoD 是路径的线性散射,和 alpha 合成的可加性匹配;CBF/MVDR 有交叉项,不匹配。先学"可加"的量(路径功率的角分布),再在渲染后做波束形成——把非线性移到可微的后处理里。
- **相位与频率**:从 `paths.cfr()` 出发,让每个高斯携带复增益和时延而不是功率;渲染得到 CIR 而不是图。这是 RF-3DGS 明确做不到的两件事,也是通信仿真真正需要的。

**远(规划闭环走向真实)**

- **实测替代合成真值**:哪怕 10 个点的实测 PDP,就能回答"材料反演在真实失配下还剩多少"。
- **多 AP 与 RAN 级目标**:Aerial 的思路——覆盖率只是最简单的目标,SINR、吞吐、切换才是网络仿真要的;目标函数换了,梯度 pipeline 不用换。
- **不确定性**:材料反演给出的是点估计;可辨识性已经暗示了后验的形状(零梯度 = 后验等于先验)。把它做成真正的贝叶斯设计(信息增益选点)是自然的下一步,现在的 $\log(1+\cdot)$ 打分是它的粗糙版。
- **RadioSight 对照**:它有 RGB/RF/语义/深度多模态和逐 AP 孪生,但单次反弹、无相位、每个环境冷启动;我们的材料反演 + 主动测量正好是"热启动"和"跨 AP 泛化"的部分。

---

## 7. 阶段 2 结果(2026-09-18,详见 [stage2_notes.md](stage2_notes.md) 与 [rrf_gsplat/README.md](../rrf_gsplat/README.md))

- **gsplat 移植复现 RF-3DGS**:发布 MVDR 数据上 15.97 dB / 0.727(发布 checkpoint 16.02 / 0.731)。
- **color function**(重生成 MVDR):rgb 18.15 dB / RMSE 4.42 dB → **db 18.75 / 3.91**,power 18.63 / 3.97;到 PSNR 17 从 69 s 到 19 s。
  线性功率合成没有比 dB 合成更好;RGB 合成会落到 jet 曲线之外(dashboard 上的对比条)。
- **归一化**:逐图 min/max(教程的 CBF/TCBF 做法)在 MVDR 上 18.15 → 11.97 dB,CBF 上 dB 误差 7.54 → 11.03——即使评估时给了 oracle 范围。
- **消融**:SH0 16.75 → SH3 18.75(视角依赖是主要表达力);opacity 冻结 −0.22 dB;2k 步 36 s 17.45 dB;160 个位置 −0.06 dB。
- **Tx 移动**:冷启动 1500 步到 17 dB(≈27 s),热启动 1250 步;rgb 热启动反而 2000 → 3250 步。
  一次 Tx 移动的总成本(实测):原流水线本机 ≈1 h;本工作 800 个位置 ≈6 min,160 个位置 **65 s**(33 s 生成 + 32 s 训练),热启动 56 s。
- **稳健性**:教程逐材料定义(修正单位错误后)下 db 18.04 vs rgb 17.29;与发布数据同设置(2.4 GHz + 教程材料)下
  rgb 16.87(发布 checkpoint 16.02)、db 17.66;CBF 上 db 13.95 vs rgb 13.68。三套数据 db 都赢。
- **教程材料的单位错误**:cell 6 的电导率公式把频率除以 1e-9,ITU 类材料全成了 1e16–1e24 S/m 的完美导体;
  `sionna_port/tutorial_materials.py` 提供原样与修正两种变体。
- **工程**:torch 侧 SH 走 autograd 一步 101 ms;改成线性映射 + gsplat CUDA SH 后 49 ms(rgb)/ 22 ms(db);INRIA 光栅器 27 ms 仍更快。
- **几何解冻**:db 模型 18.75 / 3.91 dB → 21.53 / 2.85 dB,本阶段最大单项收益。位移诊断:位置中位数只动 6 mm、最大 21 cm,
  变的是尺度与朝向;Tx-A 上适配的几何搬到 Tx-B 冻结使用,范围内 RMSE 4.82 → 4.58(略优于视觉几何),在 Tx-B 上再解冻 3.83。
  **"Tx 不变基底"的论据因此改写而非推翻**:基底 = 高斯位置(不动);局部形状可在任一 Tx 上适配一次并沿用。
- **解码精度是主指标**:多通道模型解码出 AoD 方位 19.0° / 天顶 11.6° / 时延 8.2 ns;真值路径表直接解码显示目标本身
  (σ=3 的核 + 凸组合 + 重采样)就带 12.6° 方位误差,σ=1 降到 4.9°;PSNR 只是 jet 折算的兼容指标。
- **发布数据是 2.4 GHz**(`cbf_power.csv` 与 2.4 GHz 计算值差 1.6 dB,与 60 GHz 差 34 dB),论文的 60 GHz 与数据不符。
- **同位置四个 yaw 共用 seed、投影谱一次 solve 派生四面**(与逐图归一化同类的一致性缺陷,已修;投影谱生成快 4×)。
- **未做**:MCMC 需重新配置(默认超参从收敛几何出发会发散);端到端可微(渲染残差 → 材料)写进 roadmap 不动工;
  fisheye 两半球渲染;所有结果仍是单场景、合成数据(两个频率、两套材料)。

## 附:复现路径

```
# 生成端与评价端(Windows,rf-sionna-win)
sionna_port/generate_dataset.py, run_ablation.py, comparison_grid.py, dashboard.py
# 规划端(Windows,rf-sionna-win)
tx_planning/probe_differentiability.py → tx_sweep.py → optimize_tx.py → fit_materials.py → active_measurement.py
# 原版训练与复现(Windows,rf-3dgs)
train.py, render.py, output/demo/quick_metrics.py
# dashboard
python sionna_port/dashboard.py output/ablation.json --out output/dashboard.html \
       --reference-root . --comparison output/comparison.json --planning-dir output/tx_planning
```

各目录的 README 记录了每一步的细节和失败:[sionna_port/README.md](../sionna_port/README.md),[tx_planning/README.md](../tx_planning/README.md)。
