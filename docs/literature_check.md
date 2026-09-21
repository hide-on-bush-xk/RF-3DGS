# 文献核查(2026-09-18,按 Ke 第四份清单第 6 项;只查摘要/全文,不跑实验)

目的:我们的几条主张里,哪些已经有人做了、哪些是空位、哪些数字可以直接当基线。
按与我们设计决策的相关度排序;每条先写"他们做了什么",再写"对我们意味着什么"。

## 1. RxGS(2026-05,arXiv 2605.24290):"几何与接收端无关,方向性辐射与接收端有关"

- 两阶段:阶段 I 在**一个参考接收端**的 RF 测量上联合训练全部高斯属性,收敛后**冻结位置、协方差、透射率**;
  阶段 II 引入按接收端位置条件化的辐射系数。条件化 = 对每个高斯的 SH 系数做仿射调制:
  全局分支 `(α_l, β_l) = MLP_global([γ(r); c_l; e_l])`(γ 为接收端位置的可学习 Fourier 编码),
  局部分支 `(α_k, β_k) = MLP_local([v̂_k; d_k; T_k; ρ̄_k])`——输入是**高斯到接收端的方向、距离、遮挡透射率 T_k**。
- 数据:BLE RSSI(21 接收端)、Sionna 仿真的 RFID 空间谱(21 接收端)、WiFi CSI(8 接收端);基线 GSRF、WRF-GS+、NeRF2。
  同接收端上比逐接收端模型差 0.8–1.5 dB PSNR;**未见接收端** MAE 4.92 dBm,而逐接收端模型在未见接收端上差 3 倍;训练成本省 45×。
- 明说"假设静态场景,不能适应环境变化"。
- **对我们**:这是我们"高斯位置是 Tx 不变基底、颜色是 Tx 专属"论据的**接收端镜像**,时间在前,必须引用并定位。
  我们不同的地方是量出来的:让几何动(尺度/朝向,位置只动毫米)再拿 27% RMSE,其中 76% 是 Tx 专属——RxGS 的冻结几何把这块留在桌上。
  他们局部分支里的 T_k(遮挡透射率)就是 Ke 公式 `c = f(d)·V·ΣT Y(Ω_in)Y(Ω_view)` 里的 V;
  "仿射调制 SH 系数"是 degree-1 Ω_in 之外的另一种条件化写法,可以直接搬到 Tx 侧。

## 2. BiWGS(2025-10,arXiv 2510.26166):6D CKM,双向球谐

- 每个高斯椭球是虚拟散射簇,散射系数分解 `Γ_m = Z_m · V_m(θ,φ,θ',φ')`,
  角度部分是**入射 × 出射两组球谐的张量积** `V_m = Σ_i Σ_k a_{i,k,m} y_k(θ',φ') y_i(θ,φ)`(复数,实虚部分别拟合)。
  路径 Tx → 高斯 → Rx,Tx 侧与 Rx 侧各乘一串 `(1−α) e^{−j2π/λ γ}`(遮挡与相位),不是普通 alpha 合成。
- 数据:Sionna,6 GHz,4×4 UPA,三个合成房间,最大 3 次散射;6D 实验**9 个训练 Tx → 1 个未见 Tx**。
  未见 Tx 的功率增益 MAE 3.68 / 4.93 / 6.70 dB(会议室/卧室/办公室),MLP 基线 4.40 / 7.81 / 14.60 dB;
  3D 谱质量与 WRF-GS 相当(SSIM 0.68 vs 0.69,LPIPS 更好)。没报训练/推理时间。
- **对我们**:Ke 提的 Tx 条件化传输 `ΣT Y(Ω_in)Y(Ω_view)` **已经有人写成模型并在留出 Tx 上验了**——这就是我们计划的"6 个 Tx 训、第 7 个零重训"实验的既有结果。
  我们能加的不是这个模型,而是(a)**相关长度曲线**:单 Tx 适配的几何能搬多远(他们只报 9 → 1 的平均误差,不报随距离的衰减);
  (b)针孔/gsplat 流水线的速度和物理解码指标(他们用 SSIM/LPIPS);(c)第八轮的"解冻子集"结果说明几何适配不属于某一类形状——
  这对"双向 SH 放在哪种基元上"是直接的输入。留出 Tx 实验若做,基线数字用他们的 MAE。

## 3. RF-PGS(2025-08,arXiv 2508.16849):平面高斯 + "全结构化"辐射,同一个 NIST 大厅

- 平面高斯 = 把最小尺度压成薄盘(2DGS 式),法线由旋转四元数和最小尺度轴给出;摘要说先用视觉数据重建几何,再用**很稀疏的 RF 测量**建场
  (初始化的具体机制在全文页没核到,UNVERIFIED)。
  "全结构化" = 每条多径分解成 FSPL(总路径长)× 表面交互增益(按出射方向的 SH);Tx 侧谱由几何上算出的路径长度直接聚合。
  新 Tx **仍要重训**(站点专属模型)。
- 数据:Sionna,**2.4 GHz 和 60 GHz,~14 m × 15 m 室内大厅**(就是 RF-3DGS 的场景),训练集 10–700 样本,100 测试;另有 NIST 60 GHz 实测。
  PSNR:RF-PGS 20.61 / RF-3DGS 14.22 / NeRF2 14.13(Table I,**多配置平均**,已逐字核对);训练 3 min 53 s(RF-3DGS 2 min 41 s)。
- **对我们**:最近的直接竞品,同场景同仿真器。三点要注意:(1)他们的 RF-3DGS 基线 14.22 dB 比我们复现的 15.97/16.02 低——样本数和归一化都不同,
  数字不能横比,但我们的 db 冻结 18.75、解冻 21.5 是在 2560 张训练图上;写论文要按契约把数据规模列出来。
  (2)FSPL 因子 f(d) 就是他们的分解;Ke 的公式 = RF-PGS 的 f(d) + BiWGS 的双向 SH + RxGS 的 V。
  (3)他们选盘(surfel);第八轮说明只解冻 11% 的盘就拿到 98% 增益,但只解冻针也拿到 93%——盘不是被单独挑出来的,随机对照在跑。
  Tx 换位速度:他们全训 3 min 53 s,我们 33 s + 32 s(160 位置)——同一场景,可以并排列,但要标明各自的数据量与 GPU。

## 4. GRaF(2025-02,arXiv 2502.05708):跨场景泛化,"邻近 Tx 的谱可以互相近似"

- 前提是 RF 域的插值定理:一个 Tx 的空间谱可由地理上邻近的 Tx 的谱近似;几何感知 Transformer 编码邻近 Tx 的谱,再做神经射线追踪到接收端。
  跨场景(未见布局)达到 SOTA;摘要没给基线名和数字。
- **对我们**:他们的前提正是我们 T1 曲线要量的东西——**相关长度 d_c**。GRaF 假设"近就像",我们给出"多近算近"(适配几何搬到 1.8–12.8 m 外的 Tx 的范围内 RMSE 曲线)。

## 5. 高斯上的可微射线追踪(2026-05,arXiv 2605.07781):不学场,直接在视觉重建上做 RT

- 把高斯基元嵌进硬件加速的 RT 结构,任意两点间算多跳路径,从纯视觉重建里抽出物理上有意义的 CIR;摘要无数字、无时间。
- **对我们**:这是"Tx 换位速度"这条 headline 的真正对手——RT 数字孪生换 Tx **零重训**,代价是每次查询都要追踪。
  我们的定位要写清楚:学到的场 + 热启动(33 s)vs. 物理 RT(0 s 重训、每次查询 RT 的开销);交互平台里两者其实都在(coverage/optimise 走 Sionna RT)。

## 6. GS-CG 信道增益图(2026-07,arXiv 2607.21099):冻结参考高斯 + 一小组可调高斯做增量更新

- 增益分解成距离衰减 × 路径透射 × 有效散射;环境变化时**冻结参考高斯,加一小组可调高斯**吸收新的局部变化,做增量学习。
- **对我们**:和我们的热启动同思路,但更省——只加少量高斯而不动全部。第八轮"随机 1%/3% 也够不够"的对照如果成立,就直接支持这种做法。

## 7. 其余(相关但不改变设计)

- **OctCGS**(2026-05,2605.22961):八叉树上下文高斯 + **显式多阶传播**建 CKM——多跳被当成必要项;我们 depth-1 的 2.4 GHz 多跳功率占比数字(队列里)回答"我们的孪生漏了多少"。
- **XFreq-GS**(2026-05,2605.11432):跨频率重建——与我们"发布数据是 2.4 GHz、教程材料电导率单位错"那节相邻,可引。
- **WiNeRT**(ICLR 2023):神经射线追踪代理,可微,Tx/Rx 布置是输入,下游做网络规划;页面核到的数字是飞行时间误差 < 0.33 ns(此前写的 0.58 m 定位误差未核到,撤)。
  与我们的交互规划器是同一下游任务、不同路线(学代理 vs RT 在环)。
- **GS-IR / Relightable 3DGS**(CVPR 2024):3DGS 逆渲染:每高斯法线 + BRDF,split-sum 光照,烘焙遮挡做间接光。
  图形学模板告诉我们两件事:(1)入射方向条件化在图形学里靠**法线**——而我们的形状诊断说视觉几何以针为主(长轴躺在切平面里,没有法线),
  所以 BRDF 式模型要么上 2DGS/平面(RF-PGS 路线),要么用不需要法线的双向 SH(BiWGS 路线);(2)GS-IR 的法线要从深度导数正则出来,不能直接用高斯的短轴——与我们"最短轴 vs 法线"那条测不出方向性一致。
- **RadioSight**(2026-08,2608.29504)、**稀疏测量下传播一致的数字孪生**(2026-05,2605.22361):感知辅助/数字孪生做网络优化的近邻工作;"sensing-assisted beamforming"这个词本身没有搜到与 3DGS 直接相关的文章。

## 对论文定位的结论

1. "几何 Tx 不变、辐射 Tx 相关"的**定性主张已被 RxGS(接收端)和 BiWGS(6D)占位**;我们的贡献要落在**定量**上:24/76 分解、相关长度曲线、解冻子集不挑形状。
2. Tx 条件化传输的模型形式(双向 SH × 可见性 × f(d))已存在(BiWGS + RF-PGS + RxGS 的拼接);若做,以 BiWGS 的留出 Tx MAE 为基线,并加我们的解码指标和速度。
3. Tx 换位速度这条要与两类对手并排:RF-PGS 的全训 3 min 53 s(同场景)和高斯上 RT 的零重训;数字按 CLAUDE.md 契约列全链路与数据量。
4. 平面高斯(RF-PGS)是最直接的竞品;第八轮随机对照的结果决定我们能否说"盘不是必要的"。

## 来源

- RxGS: <https://arxiv.org/abs/2605.24290>
- BiWGS: <https://arxiv.org/abs/2510.26166>
- RF-PGS: <https://arxiv.org/abs/2508.16849>
- GRaF: <https://arxiv.org/abs/2502.05708>
- Differentiable Ray Tracing with Gaussians: <https://arxiv.org/abs/2605.07781>
- GS-CG: <https://arxiv.org/abs/2607.21099>
- OctCGS: <https://arxiv.org/abs/2605.22961>;XFreq-GS: <https://arxiv.org/abs/2605.11432>;GSRF: <https://arxiv.org/abs/2502.01826>
- WiNeRT: <https://openreview.net/pdf?id=tPKKXeW33YU>;GS-IR: <https://arxiv.org/abs/2311.16473>
- RadioSight: <https://arxiv.org/abs/2608.29504>;传播一致数字孪生: <https://arxiv.org/abs/2605.22361>
- 学习型无线电地图综述/清单: <https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized>

## 验证记录(2026-09-19,按 Ke 的协议:逐个打开 arxiv.org/abs 核对标题,再在全文页逐字核对数字;没有浏览器,用 WebFetch 抓页面)

| arXiv ID | 打开 | 标题一致 | 关键数字 | 状态 |
| --- | --- | --- | --- | --- |
| 2510.26166 BiWGS | 是(2025-10-30,Zhou, Hu, Wu, Ren, Hu, Zhang, Zhang, Xu) | 是 | Table III 逐字:3.68 / 4.93 / 6.70 dB;"a training set containing measurements from 9 distinct Tx positions, and a test set containing measurements from a different Tx position";6 GHz,Sionna,最多 3 次散射 | **VERIFIED** |
| 2508.16849 RF-PGS | 是(2025-08-23,Lihao Zhang, Zongtan Li, Haijian Sun) | 是 | Table I 逐字:RF-PGS 20.6108 / 0.6606 / 0.3945,3min 53s;RF-3DGS 14.2208 / 0.3680 / 0.4250,2min 41s;**该表是多配置平均,不是单一数据集**;Sionna,2.4 与 60 GHz,≈14 m × 15 m 室内大厅 | **VERIFIED**;"平面高斯由视觉数据初始化"这一句在全文页**没找到**,标 UNVERIFIED(只保留摘要里"先用视觉数据重建几何"的表述) |
| 2605.24290 RxGS | 是(2026-05-22,Kang Yang, Mani Srivastava) | 是 | 逐字:未见接收端 MAE 4.92 dBm;逐接收端基线 "9.7 to 11.6 dBm, roughly ×3 worse";训练 "7× to 45×";推理 "up to 7.6×";几何/辐射分组与 Stage II 冻结几何原句在 | **VERIFIED** |
| 2502.05708 GRaF | 是(2025-02-08,v3 2026-04-20,Kang Yang, Yuning Chen, Wan Du) | 是 | 摘要原句:"generalizes across scenes to synthesize spectra" | **VERIFIED**(数字未引用) |
| 2605.07781 高斯上的可微 RT | 是(2026-05-08,Vaara, Huynh, Sangi, Bordallo López, Heikkilä) | 是 | 摘要原句在;无数字 | **VERIFIED** |
| 2607.21099 GS-CG | 是(2026-07-23,Chen, Guo, Zhou, Xu, Zhang) | 是 | 摘要原句在;无数字 | **VERIFIED** |
| 2605.22961 OctCGS | 是(2026-05-21,Zhang, Gong, Wang, Stirling-Gallacher, Caire) | 是 | — | VERIFIED(标题级) |
| 2605.11432 XFreq-GS | 是(2026-05-12,Wang 等) | 是 | — | VERIFIED(标题级) |
| 2502.01826 GSRF | 是(2025-02-03,Kang Yang 等) | 是 | — | VERIFIED(标题级) |
| 2311.16473 GS-IR | 是(2023-11-26,Liang, Zhang, Feng, Shan, Jia) | 是 | — | VERIFIED(标题级) |
| 2608.29504 RadioSight | 是(2026-08-30,Lihao Zhang, Kudyba, An, Haijian Sun) | 是 | — | VERIFIED(标题级) |
| 2605.22361 WEDT | 是(2026-05-21,Ai 等,Shi Jin) | 是 | — | VERIFIED(标题级) |
| WiNeRT(OpenReview tPKKXeW33YU) | OpenReview 被机器人验证页挡住;iclr.cc/virtual/2023/poster/10694 打开,标题与作者一致(Orekondy, Pratik, Kadambi, Ye, Soriaga, Behboodi,ICLR 2023) | 是 | 页面给的是 "<0.33ns error in time-of-flight predictions";我之前写的"定位误差中位 0.58 m / 1.21 m"来自搜索摘要,**页面上没核到** | 标题 VERIFIED;**0.58 m / 1.21 m 标 UNVERIFIED,改用 <0.33 ns** |

补一条与本项目直接相关的事实:RF-PGS 和 RadioSight 的通讯作者 Haijian Sun 在 UGA——同校同题,论文定位要考虑这一点。

## RF-PGS 的代码与管线(Ke 的追问,2026-09-19)

- 代码:GitHub `SunLab-UGA/RF-PGS` 存在,README 写 "Code will be released upon paper acceptance";论文原句 "Dataset and codes of this paper will be available after paper acceptance"。**目前没有可查的实现。**
- 全文页核到的生成细节:Sionna;"we apply path loss thresholds of −160 dB for 2.4 GHz and −190 dB for 60 GHz"(阈值截断);查询方向对齐到"a panoramic camera, using an equirectangular projection"。
  **没有**:每接收端位置的采样数、随机 seed、散射设置、每视角是否单独 solve、归一化方式、RF-3DGS 基线是重跑还是引用。
- 结论:无法确认 RF-PGS 是否共享"逐视角 seed"这一违反;能确认的是他们用了截断阈值(与我们的 −150 dB 同类)和全景查询(没有四面针孔的"四面派生"问题)。
  能说的话:"发布的 RF-3DGS 管线按构造带有这一违反(教程逐视角随机采样散射路径);RF-PGS 的管线未公开,不能判断。"

## 定向核查:RF-PGS 是否编码时延(2026-09-19,全文页逐字)

- ToF 纯几何:"ToF can be accurately computed from the geometric distance, eliminating the need for normalized delay approximations used in RF-3DGS";
  "the ray–surface intersection point is first estimated. Based on this, the distances to both the transmitter and receiver can be accurately computed, allowing precise calculation of the FSPL term"。
- 监督:"Training relies solely on the practically available path loss spectra, while AoD is inferred from the retrieved full path geometry, and ToF is computed from distance."——**没有学习的时延通道,没有任何"解析 + 残差"分解**(分解只用于路损:FSPL × 交互增益 SH)。
- 指标:**没有报任何 ToF/时延误差**(只有 PSNR/SSIM/LPIPS/波束成形容量)。
- 位置由此明确:RF-PGS 的 ToF 是"全解析"(需要 Tx 位置 + 射线-表面求交);我们的是 alpha 合成的解析范围项 + 学习的散射时延残差,不需要 Tx 位置和求交,并且报时延误差(中位 0.95 ns,拷贝地板 0.86)。
  "解析 + 残差用到时延上"在 RF-PGS / BiWGS / RxGS 里都没有。

## 验证记录续(2026-09-21,SOTA 对标前的补核;WebFetch 打开 abs 页面 / GitHub 页面)

| 条目 | 打开 | 核到的内容 | 状态 |
| --- | --- | --- | --- |
| 2412.04832 WRF-GS | 是:"Neural Representation for Wireless Radiation Field Reconstruction: A 3D Gaussian Splatting Approach",Wen, Tong, Hu, Lin, Zhang;v1 2024-12-06,v4 2025-03-24;INFOCOM 2025 | 摘要 KPI:RSSI 与 CSI 预测,"surpassing existing methods by more than 0.7 dB and 3.36 dB";空间谱合成优于 RT 与其它深度方法;摘要不提 PSNR/SSIM/LPIPS;**代码公开**:github.com/wenchaozheng/WRF-GS,增强版 WRF-GS+ 在 github.com/wenchaozheng/WRF-GSplus | **VERIFIED**(此前标 UNVERIFIED 的 ID 现已核) |
| 2305.06118 NeRF² | 是:"NeRF2: Neural Radio-Frequency Radiance Fields",Zhao, An, Pan, Yang;MobiCom 2023 Best Paper Runner-Up | 从 Tx 位置预测任意位置的信号(谱 / RSSI / CSI);turbo-learning;**代码 MIT 公开**:github.com/XPengZhao/NeRF2,数据集(RFID 谱、BLE RSSI、MIMO CSI)与预训练模型经 OneDrive 分发 | **VERIFIED** |
| 2502.01826 GSRF | 是:"GSRF: Complex-Valued 3D Gaussian Splatting for Efficient Radio-Frequency Data Synthesis",Yang, Dong, Ji, Du, Srivastava;v3 2025-11-06 | 复值高斯、正交投影、复值射线追踪;RSSI 合成;摘要无数字、无代码链接 | VERIFIED(摘要级) |

- 这三家(NeRF²、WRF-GS/+、GSRF)与 RxGS、BiWGS 共用 **NeRF² 的公开基准**(RFID 空间谱、BLE RSSI、MIMO CSI):固定网关 + 移动发射端,没有视觉几何。RF-3DGS 的基准是反过来的(固定 Tx + 移动接收端 + 视觉重建的几何)。两条基准之间没有人交叉跑过。
- RF-PGS(2508.16849):代码 "upon acceptance",数据未公开——在它的坐标上无法对比,只能引用。
